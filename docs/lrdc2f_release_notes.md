# LRDC2F 代码整理说明

## 整理范围

以用户指定的 `configs/rotated_rtmdet/rotated_rtmdet_l-3x-dota_ms_dcfl.py` 为基准，读取并展开它依赖的 `default_runtime.py`、`schedule_3x.py`、`dota_rr_ms.py`，合并成 `configs/lrdc2f/lrdc2f.py`。配置不再依赖其他文件。

- 移除其他模型和数据集的配置；配置目录原有 167 个文件，现在仅有 1 个 Python 配置、1 个说明文件和 1 个模型元数据文件。
- 将安装生成的 `mmrotate/.mim/` 副本归档到外层 `.cleanup-backup/generated-mim-before-cleanup/`，清除 Python 字节码缓存。归档保留了安装副本中与源目录不同的历史内容；再次安装时 `.mim` 会从当前配置和工具重新生成。
- 更新中英文 README、GitHub 引用信息、模型索引和安装元数据。
- 现有分块推理测试改用 LRDC2F 配置，避免引用已删除的配置。
- 保留底层 MMRotate 框架模块、原始许可证与版权注释，不对算法做重新实现。

当前代码是在 MMRotate v1.0.0rc1 基础上继续修改过的工作副本，并非未经修改的官方压缩包。整理前已与本地官方压缩包对比，保留已有修改。

## 名称对应与兼容性

| 原入口 | 新入口 | 处理方式 |
| --- | --- | --- |
| `rotated_rtmdet_l-3x-dota_ms_dcfl.py` | `configs/lrdc2f/lrdc2f.py` | 独立配置，展开继承 |
| `mmdet.RTMDet` | `LRDC2F` | 薄子类，只提供项目名称 |
| `RotatedRTMDetSepBNHead` | `LRDC2FHead` | 薄子类，沿用原网络和损失调用 |
| `DCFLAssigner` / `dcfl_assigner.py` | `LRDC2FAssigner` / `lrdc2f_assigner.py` | 实现迁移，保留旧注册名和导入兼容文件 |
| 配置中的 `DOTADataset` | `LRDRBDataset` | 继承相同标注读取逻辑，默认类别固定为 rockfall |

`LRDC2F` 和 `LRDC2FHead` 未增加网络层、参数或方法覆盖，原 `state_dict` 键名结构不变。实际权重加载还需在具备完整依赖的训练环境中验证，本次没有权重文件。

安装包名、导入命名空间及注册作用域仍为 `mmrotate`，版本保留 `1.0.0rc1`。CSPNeXt、GWD、QFL、DOTA 格式等组件/格式名称及预训练权重地址保留原名。上游 RTMDet 文件是实际依赖，不进行全仓库字符串替换。

目录中另有 `rotated_rtmdet_dcfl_head.py`，但指定的主配置没有使用它。本次保留该历史实验实现，不将其 PCB 模块写成当前模型已经启用的组件。`gmm_utils.py` 同样未被主配置的实际 assigner 导入；实际高斯工具位于 `structures/gaussian_tools.py`。

## 配置修正

原配置只覆写了 train/val dataloader。通过 MMEngine 实际加载可确认，原 `test_dataloader` 仍然指向 `data/split_ms_dota/trainval/`，且缺少 `rockfall` 类别信息。新配置显式设置三个 dataloader，测试使用 LRD-RB 的 `test/`。

个人机器上的绝对路径改为 `data/lrd_rb/`。仅对类别为 rockfall 的单类任务，移除旋转增强中的 DOTA 类别编号 `[9, 11]`，改为 `[]`；原编号对标签 0 不生效，增强行为保持一致。

验证仍使用原配置的 `test/`。若论文实验使用独立 val/test 划分，需要提供实际数据划分并调整验证目录；本次未推测或重排数据。

## 与论文描述的差异

论文：[Detection of oriented lunar rockfalls and kinematic analysis in Tycho Crater](https://doi.org/10.1016/j.isprsjprs.2026.06.036)。

| 项目 | 论文描述 | 用户指定的现有配置 |
| --- | --- | --- |
| 训练轮数（4.1.2 节） | 12 epochs | 36 epochs |
| Batch size（4.1.2 节） | 4 | 每卡 6 |
| CPS / MPS 数量（3.2.1 / 3.2.2 节） | 30 / 20 | 40 / 25 |
| 回归损失（3.3.2 节） | Rotated IoU Loss | GDLoss，loss_type=gwd |
| 不足最小正样本数时的补样（Algorithm 1） | 按 DGMM 分数排序 | 沿用 MPS 的预测质量排序，取前若干样本 |

整理保留代码现状，不根据论文文字擅自修改训练或补样逻辑。模型索引没有填入论文 AP50、Recall 或虚构的权重链接。论文所述设置与对应最终实验配置、权重之间的对应关系仍需核实。

## 验证范围

本次使用 MMEngine 0.6.0 加载整理前后的配置并对比。验证配置可独立解析、训练数值参数和网络参数没有意外改变、模型索引的路径存在，以及 Python 语法和本地导入路径。

本机用于整理的 Python 环境没有 PyTorch、MMCV 编译算子和 MMDetection，也未提供训练数据或权重，因此没有执行 GPU 训练、完整模型构建、实际权重加载或指标复现。新项目入口的兼容性结论基于配置对比和源码结构检查。

## GitHub 目录

将本目录（包含 `setup.py`、`configs/`、`mmrotate/` 和 README）的内容作为 GitHub 仓库根目录即可。外层文件夹名不影响 Python 包名。

整理前的完整备份位于工作区外层 `.cleanup-backup/`；临时验证依赖位于外层 `.cleanup-work/`，均被外层 `.gitignore` 排除。它们不是发布代码的一部分。

原有 CI 工作流和需要 GPU 的框架测试保留。配置解析与语法检查通过不代表训练复现通过。
