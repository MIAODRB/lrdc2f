# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.detectors.rtmdet import RTMDet

from mmrotate.registry import MODELS


@MODELS.register_module()
class LRDC2F(RTMDet):
    """LRDC2F detector with the original MMDetection module layout.

    The project name is exposed through the registry without adding layers or
    changing checkpoint keys. The head, coarse-to-fine assigner and losses are
    specified in ``configs/lrdc2f/lrdc2f.py``.
    """
