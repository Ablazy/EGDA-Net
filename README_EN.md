# EGDA-Net

[简体中文](README.md)

## Installation

Python 3.10 or later is recommended. Run the following command from this directory:

```bash
python -m pip install -r requirements.txt
```

Download the [pretrained model](https://1drv.ms/f/c/38c6657e5842214a/IgBFTgOKir41SIwPnGuyIB9jAaLGi3jZKzy1uj5RMdC5_7s?e=F0BZFK) and [sample data](https://1drv.ms/f/c/38c6657e5842214a/IgDcifzMbYUSQIZQ8O-NaaSmAbdyQEXfXl2hCg3ieItTZrs?e=feT9fh) from these links.

## Directory Structure

```text
onnx_inference/
├── DSEC.onnx                     # Fixed 480×640 input, batch size 1, opset 22
├── infer_sequence.py             # Reads images and events separately, builds inputs online, and runs inference
├── utils.py                      # ONNX inference utilities, online event loading, and voxelization
├── requirements.txt
└── data/zurich_city_15_a/
    ├── images_processed/timestamps.txt     # Full original timestamp table, indexed by original frame number
    ├── images_processed/left/
    │   ├── 000895.png
    │   ├── ...
    │   └── 000994.png            # 100 images in total
    └── processed_events_rectified/
        ├── t_offset.h5
        ├── 000894.h5             # Boundary event segment
        ├── ...
        └── 000994.h5             # 101 event segments in total
```

## Running Inference

```bash
# Run inference on a specific scene
python infer_sequence.py --scene data/zurich_city_15_a \
  --skip 1 --insert-count 1 --stride 2 \
  --output-dir outputs/zurich_city_15_a
```

`--skip` sets the number of images skipped between the two input endpoints; the endpoint frame distance is `skip + 1`. `--insert-count` independently sets the number of predictions generated within each endpoint window. Prediction timestamps are evenly spaced between the actual timestamps of the two endpoints. For example, `--skip 3 --insert-count 2` uses endpoints four frames apart and predicts two images between them. `--stride` sets the distance between the starting points of adjacent endpoint windows. Setting `stride` to `skip + 1` makes the windows non-overlapping internally while sharing their boundary frame.
