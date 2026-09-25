# Copyright (c) OpenMMLab. All rights reserved.
from mmrotate.registry import MODELS
from .rotated_rtmdet_head import RotatedRTMDetSepBNHead


@MODELS.register_module()
class LRDC2FHead(RotatedRTMDetSepBNHead):
    """LRDC2F's shared-convolution head with separate normalization per level.

    Inherits the head used by the supplied experiment without changing its
    forward pass, loss computation or parameter names.
    """
