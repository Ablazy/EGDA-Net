#!/usr/bin/env python3
"""独立推理所需的图像处理、ONNX 运行时与在线事件体素化工具。"""
from pathlib import Path

import h5py
import hdf5plugin  # 注册校正事件 HDF5 数据可能使用的压缩过滤器。
import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent


def load_image(path, height=256, width=256):
    """按 RGB 顺序读取并归一化，不自动缩放以免破坏与事件的空间对齐。"""
    with Image.open(path) as image:
        array = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0
    if array.shape != (height, width, 3):
        raise ValueError(f'{path}: 图像必须为 {width}×{height}，实际为 {array.shape}')
    return array.transpose(2, 0, 1)


def pack_inputs(image_0, image_1, events, height=256, width=256):
    """沿用训练时通道顺序，事件体素和 IWE 必须已按原方法预处理。"""
    iwe = np.asarray(events['iwe'])
    if iwe.shape == (height, width):
        iwe = iwe[None]
    parts = []
    for name, array, shape in [
        ('image_0', image_0, (3, height, width)),
        ('image_1', image_1, (3, height, width)),
        ('eframes_t0', events['eframes_t0'], (15, height, width)),
        ('eframes_t1', events['eframes_t1'], (15, height, width)),
        ('iwe', iwe, (1, height, width)),
    ]:
        array = np.asarray(array, dtype=np.float32)
        if array.shape != shape:
            raise ValueError(f'{name} 形状应为 {shape}，实际为 {array.shape}')
        if name.startswith('image_') and (not np.isfinite(array).all() or
                                         array.min() < 0 or array.max() > 1):
            raise ValueError(f'{name} 必须是归一化到 [0,1] 的 RGB 数据')
        parts.append(array)
    weight = np.asarray(events['weight'], dtype=np.float32)
    if weight.size != 1 or not np.isfinite(weight).all() or not 0 <= weight.item() <= 1:
        raise ValueError('weight 必须是 [0,1] 范围内的单个数值')
    parts.append(np.full((1, height, width), weight.item(), dtype=np.float32))
    return np.concatenate(parts, axis=0)[None]


def create_session(model, device='cpu', threads=4):
    """两个推理入口共享运行时配置，并检查 GPU 后端是否真正初始化。"""
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    providers = ['CPUExecutionProvider']
    if device == 'cuda':
        ort.preload_dlls()
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            raise RuntimeError('未发现 CUDA 后端，请安装匹配 CUDA/cuDNN 的 onnxruntime-gpu')
        providers.insert(0, 'CUDAExecutionProvider')
    session = ort.InferenceSession(str(model), sess_options=options, providers=providers)
    if device == 'cuda' and 'CUDAExecutionProvider' not in session.get_providers():
        raise RuntimeError('CUDA 后端初始化失败，请检查 CUDA/cuDNN 动态库')
    return session


def input_shape(session):
    """从模型读取固定尺寸，不把导出时的空间尺寸写死在输入构造中。"""
    shape = session.get_inputs()[0].shape
    if len(shape) != 4 or shape[:2] != [1, 38] or not all(isinstance(v, int) for v in shape):
        raise ValueError(f'模型应为固定 [1,38,H,W] 输入，实际为 {shape}')
    return tuple(shape)


# 以下为在线事件读取与体素化工具，集中放在本文件，便于独立发布推理目录。

def voxelize(x, y, p, t, height, width, bins=15):
    """三线性体素化，仅对非零体素做均值/样本标准差归一化。"""
    grid = np.zeros((bins, height, width), dtype=np.float32)
    if len(t) <= 10 or t[-1] <= t[0]:
        return grid
    # 两步时间归一化与原 Dataset、VoxelGrid 的 float32 计算顺序保持一致。
    time = (t - t[0]).astype(np.float32)
    time /= time[-1]
    time = (bins - 1) * (time - time[0]) / (time[-1] - time[0])
    x, y, p = [np.asarray(v, dtype=np.float32) for v in (x, y, p)]
    x0, y0, t0 = [v.astype(np.int32) for v in (x, y, time)]
    flat = grid.reshape(-1)
    for xi in (x0, x0 + 1):
        for yi in (y0, y0 + 1):
            for ti in (t0, t0 + 1):
                valid = ((xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
                         & (ti >= 0) & (ti < bins))
                # 保留 float32 插值，避免 NumPy 的整数运算隐式提升到 float64。
                weights = ((2 * p - 1) * (1 - np.abs(xi.astype(np.float32) - x))
                           * (1 - np.abs(yi.astype(np.float32) - y))
                           * (1 - np.abs(ti.astype(np.float32) - time)))
                index = height * width * ti.astype(np.int64) + width * yi + xi
                np.add.at(flat, index[valid], weights[valid])
    nonzero = grid != 0
    values = grid[nonzero]
    if values.size:
        mean = values.mean()
        std = values.std(ddof=1) if values.size > 1 else np.float32(0)
        grid[nonzero] = (values - mean) / std if std > 0 else values - mean
    return grid


class RawSequence:
    """以文件名中的原始帧号关联图像、完整时间戳表和事件分段。"""

    def __init__(self, scene):
        self.scene = Path(scene)
        self.images = sorted((self.scene / 'images_processed' / 'left').glob('*.png'))
        if not self.images:
            raise FileNotFoundError(f'没有找到图像: {self.scene}/images_processed/left')
        self.frame_ids = [int(path.stem) for path in self.images]
        if any(b != a + 1 for a, b in zip(self.frame_ids, self.frame_ids[1:])):
            raise ValueError('图像帧号必须连续，不能把缺帧序列当成连续序列')
        with Image.open(self.images[0]) as image:
            self.width, self.height = image.size
        event_dir = self.scene / 'processed_events_rectified'
        with h5py.File(event_dir / 't_offset.h5', 'r') as data:
            offset = int(data['t_offset'][()])
        # 子集仍保留完整时间戳表，通过原始帧号索引，不能重新从 0 计数。
        self.timestamps = np.atleast_1d(np.loadtxt(
            self.scene / 'images_processed' / 'timestamps.txt', dtype=np.int64)) - offset
        if self.frame_ids[-1] >= len(self.timestamps):
            raise ValueError('时间戳表不足以覆盖图像帧号，请保留原场景完整时间戳表')
        if np.any(np.diff(self.timestamps) <= 0):
            raise ValueError('图像时间戳必须严格递增')
        self.event_paths = {int(p.stem): p for p in event_dir.glob('*.h5') if p.stem.isdigit()}

    def windows(self, skip=1, stride=1):
        """skip 表示端点间跳过的图像帧数，返回端点帧号。"""
        for start in range(0, len(self.frame_ids) - skip - 1, stride):
            end = start + skip + 1
            yield self.frame_ids[start], self.frame_ids[end]

    def insertion_times(self, first, last, insert_count):
        """按端点时间戳等间隔生成指定数量的插入时刻（微秒）。"""
        if insert_count <= 0:
            raise ValueError('insert_count 必须为正数')
        ts0, ts1 = self.timestamps[first], self.timestamps[last]
        return [float(ts0) + (float(ts1) - float(ts0)) * index / (insert_count + 1)
                for index in range(1, insert_count + 1)]

    def nearest_frame(self, timestamp):
        """返回时间上最接近给定时刻的原始图像帧号，供筛选与日志显示。"""
        index = int(np.searchsorted(self.timestamps, timestamp))
        index = min(max(index, 0), len(self.timestamps) - 1)
        if index and abs(self.timestamps[index - 1] - timestamp) <= abs(self.timestamps[index] - timestamp):
            index -= 1
        return index

    def build_events(self, first, target_time, last):
        """每次读取事件列表并按目标时间在线体素化，不读取或写入体素缓存。"""
        chunks = {key: [] for key in ('x', 'y', 'p', 't')}
        # 与原 Dataset 一致：多读起始帧之前的一段，并读到末端帧对应的事件段。
        for frame in range(max(0, first - 1), last + 1):
            if frame not in self.event_paths:
                raise FileNotFoundError(f'缺少事件段 {frame:06d}.h5，不能静默按零事件推理')
            with h5py.File(self.event_paths[frame], 'r') as data:
                for key in chunks:
                    chunks[key].append(np.asarray(data[key]))
        x, y, p, t = [np.concatenate(chunks[key]) for key in ('x', 'y', 'p', 't')]
        valid = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        x, y, p, t = [v[valid] for v in (x, y, p, t)]
        if np.any(t[1:] < t[:-1]):
            raise ValueError('事件段拼接后的时间戳无序，请检查事件分段或时间基准')
        if np.any((p != 0) & (p != 1)):
            raise ValueError('校正事件的极性必须为 0/1')
        ts0, ts1 = [self.timestamps[frame] for frame in (first, last)]
        tst = float(target_time)
        outputs = {}
        for key, endpoint in [('eframes_t0', ts0), ('eframes_t1', ts1)]:
            mask = (t >= min(tst, endpoint)) & (t <= max(tst, endpoint))
            vx, vy, vp, vt = [v[mask] for v in (x, y, p, t)]
            if tst > endpoint and len(vt):
                # 目标 -> 起始帧：同时反转时间、事件顺序和二值极性。
                vx, vy, vp, vt = vx[::-1], vy[::-1], 1 - vp[::-1], (vt[-1] - vt)[::-1]
            outputs[key] = voxelize(vx, vy, vp, vt, self.height, self.width)
        iwe = np.zeros((self.height, self.width), dtype=np.float32)
        mask = (t >= tst - 25000) & (t <= tst + 25000)
        np.add.at(iwe, (y[mask].astype(np.int64), x[mask].astype(np.int64)),
                  2 * p[mask].astype(np.float32) - 1)
        outputs['iwe'] = iwe
        outputs['weight'] = np.float32((tst - ts0) / (ts1 - ts0))
        return outputs

    def image_path(self, frame):
        return self.images[frame - self.frame_ids[0]]
