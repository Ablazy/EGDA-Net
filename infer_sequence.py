#!/usr/bin/env python3
"""从 DSEC 图像和校正事件列表在线构造输入并执行 ONNX 序列插帧。"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from utils import ROOT, RawSequence, create_session, input_shape, load_image, pack_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=ROOT / 'data/zurich_city_15_a')
    parser.add_argument('--model', type=Path, default=ROOT / 'DSEC.onnx')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/sequence')
    parser.add_argument('--skip', type=int, default=1, help='端点之间跳过的帧数')
    parser.add_argument('--insert-count', type=int, default=1, help='每个端点窗口生成的插帧数')
    parser.add_argument('--stride', type=int, default=1, help='端点窗口滑动步长')
    parser.add_argument('--limit', type=int, default=None, help='最多处理几个端点窗口，默认全部')
    parser.add_argument('--target-frame', type=int, default=None, help='仅运行最接近此原始帧号的插入时刻')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--save-npy', action='store_true', help='额外保存原始浮点预测')
    args = parser.parse_args()
    if min(args.skip, args.insert_count, args.stride, args.threads) <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error('skip、insert-count、stride、threads 和 limit 必须为正数')
    sequence = RawSequence(args.scene)
    session = create_session(args.model, args.device, args.threads)
    shape = input_shape(session)
    height, width = shape[-2:]
    if (sequence.height, sequence.width) != (height, width):
        raise ValueError(f'图像尺寸 {(sequence.height, sequence.width)} 与模型 {(height, width)} 不符')
    windows = list(sequence.windows(args.skip, args.stride))
    if args.limit is not None:
        windows = windows[:args.limit]
    samples = [(first, last, index, target_time)
               for first, last in windows
               for index, target_time in enumerate(
                   sequence.insertion_times(first, last, args.insert_count), start=1)]
    if args.target_frame is not None:
        samples = [item for item in samples
                   if sequence.nearest_frame(item[3]) == args.target_frame]
    if not samples:
        raise ValueError('没有满足条件的插帧窗口')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'scene': args.scene.name, 'input_shape': list(shape), 'skip': args.skip,
                'insert_count': args.insert_count,
                'stride': args.stride, 'online_voxelization': True,
                'providers': session.get_providers(), 'planned_samples': len(samples),
                'completed_samples': []}
    print(f'{len(sequence.images)} 张输入图像，{len(windows)} 个端点窗口，'
          f'准备生成 {len(samples)} 张插帧', flush=True)
    for sample_index, (first, last, insert_index, target_time) in enumerate(samples, start=1):
        start = time.perf_counter()
        events = sequence.build_events(first, target_time, last)
        x = pack_inputs(load_image(sequence.image_path(first), height, width),
                        load_image(sequence.image_path(last), height, width), events, height, width)
        if not np.isfinite(x).all():
            raise ValueError('在线构造的输入含非有限数值')
        preprocessing_seconds = time.perf_counter() - start
        start = time.perf_counter()
        prediction = session.run(['prediction'], {'input': x})[0]
        inference_seconds = time.perf_counter() - start
        if prediction.shape != (1, 3, height, width) or not np.isfinite(prediction).all():
            raise RuntimeError('模型输出形状或数值异常')
        target_frame = sequence.nearest_frame(target_time)
        stem = f's{first:06d}_t{target_time:.0f}_i{insert_index:03d}_e{last:06d}'
        image = np.rint(np.clip(prediction[0].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
        Image.fromarray(image).save(args.output_dir / f'{stem}.png')
        if args.save_npy:
            np.save(args.output_dir / f'{stem}.npy', prediction)
        item = {'start': first, 'target_nearest_frame': target_frame,
                'target_timestamp_us': target_time, 'end': last,
                'insert_index': insert_index, 'insert_count': args.insert_count,
                'weight': float(events['weight']),
                'prediction': f'{stem}.png', 'preprocessing_seconds': preprocessing_seconds,
                'inference_seconds': inference_seconds}
        manifest['completed_samples'].append(item)
        # 每个窗口落盘进度，长序列中断后也能查看已经完成的样本。
        (args.output_dir / 'run.json').write_text(
            json.dumps(manifest, indent=2), encoding='utf-8')
        print(f'[{sample_index}/{len(samples)}] {stem}: '
              f'在线预处理 {preprocessing_seconds:.2f}s，推理 {inference_seconds:.2f}s', flush=True)


if __name__ == '__main__':
    main()
