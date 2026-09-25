# Copyright (c) OpenMMLab. All rights reserved.
from mmrotate.registry import DATASETS
from .dota import DOTADataset


@DATASETS.register_module()
class LRDRBDataset(DOTADataset):
    """LRD-RB rockfalls stored as DOTA-format quadrilateral annotations."""

    METAINFO = dict(classes=('rockfall', ), palette=[(220, 20, 60)])
