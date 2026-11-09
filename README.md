# EGDA-Net

[English](README_EN.md)

## 安装

建议 Python 3.10 或更新版本。在本目录执行：

```bash
python -m pip install -r requirements.txt
```

[预训练模型](https://1drv.ms/f/c/38c6657e5842214a/IgBFTgOKir41SIwPnGuyIB9jAaLGi3jZKzy1uj5RMdC5_7s?e=F0BZFK)和[样例数据](https://1drv.ms/f/c/38c6657e5842214a/IgDcifzMbYUSQIZQ8O-NaaSmAbdyQEXfXl2hCg3ieItTZrs?e=feT9fh)可以在链接中下载。

## 目录

```text
onnx_inference/
├── DSEC.onnx                     # 固定 480×640，batch=1，opset 22
├── infer_sequence.py             # 分开读取图像/事件，在线构造输入并推理
├── utils.py                      # ONNX 推理工具、在线事件读取和体素化
├── requirements.txt
└── data/zurich_city_15_a/
    ├── images_processed/timestamps.txt     # 保留完整原始时间戳表，以原始帧号索引
    ├── images_processed/left/
    │   ├── 000895.png
    │   ├── ...
    │   └── 000994.png            # 共 100 张图像
    └── processed_events_rectified/
        ├── t_offset.h5
        ├── 000894.h5             # 边界事件段
        ├── ...
        └── 000994.h5            # 共 101 个事件段
```


## 运行

```bash
# 指定场景
python infer_sequence.py --scene data/zurich_city_15_a \
  --skip 1 --insert-count 1 --stride 2 \
  --output-dir outputs/zurich_city_15_a
```

`--skip` 控制两个输入端点之间跳过多少张图像；端点帧距为 `skip + 1`。`--insert-count` 独立控制每个端点窗口输出多少张预测，插入时刻按两端真实时间戳等间隔分布。例如 `--skip 3 --insert-count 2` 使用相距 4 帧的端点，并在它们之间预测 2 张图像。`--stride` 控制相邻端点窗口的起点间隔，设置`stride`为`skip + 1`窗口内部不重叠，窗口共享边界帧。


