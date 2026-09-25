# LRDC2F

[English](README.md) | [简体中文](README_zh-CN.md)

Code for **LRD-C2F**, the coarse-to-fine oriented lunar rockfall detector in
[Detection of oriented lunar rockfalls and kinematic analysis in Tycho Crater](https://doi.org/10.1016/j.isprsjprs.2026.06.036),
ISPRS Journal of Photogrammetry and Remote Sensing, 2026.

This repository is based on [MMRotate v1.0.0rc1](https://github.com/open-mmlab/mmrotate/releases/tag/v1.0.0rc1)
and retains the supplied local modifications. The configuration directory contains
only **[configs/lrdc2f/lrdc2f.py](configs/lrdc2f/lrdc2f.py)**, a self-contained
single-class rockfall experiment.

## Model

- CSPNeXt backbone and feature pyramid for multiscale feature extraction.
- Oriented detection head with separate normalization at each feature level.
- Coarse-to-fine assignment: GJSD candidate selection, posterior quality
  re-ranking and semantic-only DGMM refinement.
- Adaptive positive-sample counts for small rockfalls.

The public model, head and assigner names are `LRDC2F`, `LRDC2FHead` and
`LRDC2FAssigner`. Their original layer layout and numerical implementation are
preserved. `DCFLAssigner` remains available as a compatibility alias.

## Installation

Use the original working MMRotate training environment when available. Commands
below run from this directory, which contains `setup.py`.

Install the framework dependencies first, following the
[MMRotate 1.x installation guide](https://mmrotate.readthedocs.io/en/1.x/get_started.html).
Choose MMCV/PyTorch/CUDA builds compatible with your machine. The intersection
of this checkout's dependency metadata and import-time version checks is:

| Dependency | Declared compatible range |
| --- | --- |
| MMRotate | 1.0.0rc1, this modified checkout |
| MMCV with compiled operators | >= 2.0.0rc4, < 2.1.0 |
| MMDetection | >= 3.0.0rc6, < 3.1.0 |
| MMEngine | >= 0.6.0, < 1.0.0 |

These ranges are not a tested environment lockfile. Record exact versions from
the original training machine for a reproducible release. The default configuration
retains the original Linux/CUDA runtime settings (`fork`, `nccl`).

Install this checkout into that environment:

```bash
pip install -v -e .
```

## Dataset

The default root is `data/lrd_rb/`. Edit `data_root` near the top of the
configuration to point to your dataset:

```text
data/lrd_rb/
├── train/
│   ├── images/       # PNG images
│   └── annfiles/     # One TXT annotation per image
└── test/
    ├── images/
    └── annfiles/
```

Each object is one DOTA-format annotation line:

```text
x1 y1 x2 y2 x3 y3 x4 y4 rockfall difficulty
```

For example, `train/images/example.png` corresponds to
`train/annfiles/example.txt`. Use difficulty `0` for ordinary annotations.
The dataset class is `LRDRBDataset`, with a single `rockfall` category.

Both validation and testing use `test/`, matching the supplied validation
configuration. If your benchmark has a separate validation split, change
`val_dataloader.dataset.ann_file` and `val_dataloader.dataset.data_prefix.img_path`
to `val/annfiles/` and `val/images/` before model selection.

## Training and evaluation

```bash
# Single GPU
python tools/train.py configs/lrdc2f/lrdc2f.py --work-dir work_dirs/lrdc2f

# Multiple GPUs (replace 3 with your device count)
bash tools/dist_train.sh configs/lrdc2f/lrdc2f.py 3 --work-dir work_dirs/lrdc2f

# Evaluate your checkpoint
python tools/test.py configs/lrdc2f/lrdc2f.py work_dirs/lrdc2f/epoch_36.pth

# Inference on an individual image
python demo/image_demo.py path/to/image.png configs/lrdc2f/lrdc2f.py work_dirs/lrdc2f/epoch_36.pth --out-file result.png
```

The supplied settings are 36 epochs, batch size 6 **per GPU**, AdamW with
learning rate 0.00025, 1024 × 1024 inputs, 40/25 candidate samples, QFL and GWD
losses. GPU count changes the total batch size; the learning rate is not
automatically rescaled by these commands.

To override the data path on the command line, pass all dataloader paths:

```bash
python tools/train.py configs/lrdc2f/lrdc2f.py --cfg-options train_dataloader.dataset.data_root=/path/to/lrd_rb/ val_dataloader.dataset.data_root=/path/to/lrd_rb/ test_dataloader.dataset.data_root=/path/to/lrd_rb/
```

## Code map

| Component | File |
| --- | --- |
| Model configuration | [configs/lrdc2f/lrdc2f.py](configs/lrdc2f/lrdc2f.py) |
| Detector entry | [mmrotate/models/detectors/lrdc2f.py](mmrotate/models/detectors/lrdc2f.py) |
| Detection head entry | [mmrotate/models/dense_heads/lrdc2f_head.py](mmrotate/models/dense_heads/lrdc2f_head.py) |
| Coarse-to-fine assigner | [mmrotate/models/task_modules/assigners/lrdc2f_assigner.py](mmrotate/models/task_modules/assigners/lrdc2f_assigner.py) |
| Semantic-only DGMM and Gaussian helpers | [mmrotate/structures/gaussian_tools.py](mmrotate/structures/gaussian_tools.py) |
| LRD-RB dataset | [mmrotate/datasets/lrd_rb.py](mmrotate/datasets/lrd_rb.py) |

Other framework modules are retained because they are imported by MMRotate.
The inherited documentation, tools and tests may describe upstream functionality;
the configuration above is the supported experiment in this distribution.

## Citation and acknowledgements

```bibtex
@article{miao2026lrdc2f,
  title   = {Detection of oriented lunar rockfalls and kinematic analysis in Tycho Crater},
  author  = {Miao, Dingruibo and Yan, Jianguo and Tu, Zhigang},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  volume  = {239},
  pages   = {963--975},
  year    = {2026},
  doi     = {10.1016/j.isprsjprs.2026.06.036}
}
```

This implementation builds on [MMRotate](https://github.com/open-mmlab/mmrotate),
[MMDetection](https://github.com/open-mmlab/mmdetection),
[MMCV](https://github.com/open-mmlab/mmcv), and
[MMEngine](https://github.com/open-mmlab/mmengine). Please also
acknowledge the upstream projects and methods when using their work.

The upstream [Apache-2.0 license](LICENSE) and copyright notices are retained.
