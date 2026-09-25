# LRDC2F

[lrdc2f.py](lrdc2f.py) is the only model configuration distributed in this
repository. It is self-contained: no external base configuration is required.

The model combines a CSPNeXt backbone, feature pyramid, oriented detection head,
coarse-to-fine assignment and semantic-only DGMM for the `rockfall` category.

| Setting | Supplied experiment |
| --- | --- |
| Image size | 1024 × 1024 |
| Angle representation | `le90` |
| Training epochs | 36 |
| Training batch size per device | 6 |
| Optimizer / learning rate | AdamW / 0.00025 |
| Coarse / medium candidates | 40 / 25 |
| DGMM threshold | 0.08 → 0.2 over 2000 assigner iterations |
| DGMM geometry / semantic weights | 0.0 / 1.0 |
| Classification / regression loss | Quality Focal Loss / GWD |
| Validation / test split | `test/` / `test/` |

Set `data_root` at the top of the file before training. All three dataloaders use
this value. Changing only `data_root` through `--cfg-options` does not recalculate
the dataloaders; override their nested `dataset.data_root` values instead.

See the [project guide](../../README.md), [中文指南](../../README_zh-CN.md) and
[release notes](../../docs/lrdc2f_release_notes.md). The release notes record
differences between the supplied experiment and the published paper.
