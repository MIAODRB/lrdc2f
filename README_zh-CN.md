# LRDC2F

[English](README.md) | [简体中文](README_zh-CN.md)

论文 **[Detection of oriented lunar rockfalls and kinematic analysis in Tycho Crater](https://doi.org/10.1016/j.isprsjprs.2026.06.036)** 的 LRD-C2F 月球岩崩旋转检测代码。

本项目基于 [MMRotate v1.0.0rc1](https://github.com/open-mmlab/mmrotate/releases/tag/v1.0.0rc1)，保留原工作目录中的本地修改。配置目录仅保留一个独立入口：

**[configs/lrdc2f/lrdc2f.py](configs/lrdc2f/lrdc2f.py)**

## 模型组成

- CSPNeXt 主干及多尺度特征融合。
- 具有分层归一化的旋转检测头。
- 基于 GJSD 的粗候选选择、预测质量重排序和 Semantic-only DGMM 精筛。
- 面向微小岩崩的自适应正样本数量保障。

对外注册名称统一为 `LRDC2F`、`LRDC2FHead`、`LRDC2FAssigner`，数据集名称为 `LRDRBDataset`。保留原网络层级和参数键名，`DCFLAssigner` 作为旧代码的兼容别名。

## 环境与安装

优先使用原来能正常训练的 MMRotate 环境。在包含 `setup.py` 的本目录执行：

```bash
pip install -v -e .
```

新建环境时，先按 [MMRotate 1.x 安装文档](https://mmrotate.readthedocs.io/en/1.x/get_started.html) 安装与 PyTorch/CUDA 匹配的框架依赖。当前源码版本检查与依赖声明的交集为：

| 依赖 | 声明兼容范围 |
| --- | --- |
| MMRotate | 本仓库修改后的 1.0.0rc1 |
| MMCV（含编译算子） | >= 2.0.0rc4，< 2.1.0 |
| MMDetection | >= 3.0.0rc6，< 3.1.0 |
| MMEngine | >= 0.6.0，< 1.0.0 |

这些范围不是经过训练验证的环境锁定文件，正式复现应记录原训练机器的实际依赖版本。配置保留原有 Linux/CUDA 设置（`fork`、`nccl`）。

## 数据准备

修改配置文件开头的 `data_root`，默认目录为：

```text
data/lrd_rb/
├── train/
│   ├── images/
│   └── annfiles/
└── test/
    ├── images/
    └── annfiles/
```

默认使用 PNG 图像，每张图像对应一个同名 TXT 文件。每个目标一行，采用 DOTA 四边形格式：

```text
x1 y1 x2 y2 x3 y3 x4 y4 rockfall difficulty
```

普通目标的 `difficulty` 可填 `0`，唯一类别为 `rockfall`。

验证和测试都指向 `test/`，沿用指定配置中的验证路径。若数据集有独立验证集，应将 `val_dataloader` 下的标注和图像路径改成 `val/annfiles/`、`val/images/`，再进行模型选择。

## 训练、测试与推理

以下命令均在包含 `setup.py` 的目录运行：

```bash
# 单卡训练
python tools/train.py configs/lrdc2f/lrdc2f.py --work-dir work_dirs/lrdc2f

# 多卡训练：3 为 GPU 数量
bash tools/dist_train.sh configs/lrdc2f/lrdc2f.py 3 --work-dir work_dirs/lrdc2f

# 测试已有权重
python tools/test.py configs/lrdc2f/lrdc2f.py work_dirs/lrdc2f/epoch_36.pth

# 单图推理
python demo/image_demo.py path/to/image.png configs/lrdc2f/lrdc2f.py work_dirs/lrdc2f/epoch_36.pth --out-file result.png
```

默认参数：36 个 epoch、每卡 batch size 为 6、AdamW、学习率 0.00025、输入 1024 × 1024、粗/中候选数 40/25、QFL 分类损失和 GWD 回归损失。改变 GPU 数量会改变总 batch size，上述命令不会自动缩放学习率。

直接修改配置开头的 `data_root` 会更新三个 dataloader。通过命令行修改路径时，需要覆盖嵌套字段，不能只写 `--cfg-options data_root=...`：

```bash
python tools/train.py configs/lrdc2f/lrdc2f.py --cfg-options train_dataloader.dataset.data_root=/path/to/lrd_rb/ val_dataloader.dataset.data_root=/path/to/lrd_rb/ test_dataloader.dataset.data_root=/path/to/lrd_rb/
```

## 核心文件

| 内容 | 路径 |
| --- | --- |
| 独立主配置 | [configs/lrdc2f/lrdc2f.py](configs/lrdc2f/lrdc2f.py) |
| 模型入口 | [mmrotate/models/detectors/lrdc2f.py](mmrotate/models/detectors/lrdc2f.py) |
| 检测头入口 | [mmrotate/models/dense_heads/lrdc2f_head.py](mmrotate/models/dense_heads/lrdc2f_head.py) |
| 粗到细样本分配 | [mmrotate/models/task_modules/assigners/lrdc2f_assigner.py](mmrotate/models/task_modules/assigners/lrdc2f_assigner.py) |
| Semantic-only DGMM | [mmrotate/structures/gaussian_tools.py](mmrotate/structures/gaussian_tools.py) |
| 岩崩数据集 | [mmrotate/datasets/lrd_rb.py](mmrotate/datasets/lrd_rb.py) |

其余底层模块因 MMRotate 的导入关系予以保留。原框架文档、工具和测试仍可能涉及上游模型，当前发行配置只提供 LRDC2F。

## 引用与致谢

论文 BibTeX 见 [English README](README.md#citation-and-acknowledgements)，GitHub 引用信息见 [CITATION.cff](CITATION.cff)。

本实现使用 MMRotate、MMDetection、MMCV、MMEngine，保留上游版权声明和 [Apache-2.0 许可证](LICENSE)。
