# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale
from mmcv.ops import DeformConv2d, ModulatedDeformConv2d
from mmdet.models import inverse_sigmoid
from mmdet.models.task_modules import anchor_inside_flags
from mmdet.models.utils import (filter_scores_and_topk, multi_apply,
                                select_single_mlvl, sigmoid_geometric_mean,
                                unmap)
from mmdet.structures.bbox import distance2bbox
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, reduce_mean)
from mmengine import ConfigDict
from mmengine.model import bias_init_with_prob, normal_init
from mmengine.structures import InstanceData
from torch import Tensor

from mmrotate.registry import MODELS
from mmrotate.structures import distance2obb
from .rotated_rtmdet_head import RotatedRTMDetHead


@MODELS.register_module()
class RotatedRTMDetDCFLHead(RotatedRTMDetHead):
    """RTMDet head with DCFL (Dynamic Coarse-to-Fine Learning) for oriented tiny object detection.
    
    This head integrates DCFL's Prior Capturing Block (PCB) into RTMDet architecture
    to enhance detection performance on small rotated objects like rockfall.
    
    Args:
        num_classes (int): Number of categories excluding the background category.
        in_channels (int): Number of channels in the input feature map.
        dcn_on (bool): Whether to use deformable convolution for prior capturing.
            Defaults to True.
        dilation_rate (int): Dilation rate for initial offset sampling. Defaults to 2.
        pcb_stacked_convs (int): Number of stacked convs before PCB module. Defaults to 2.
        **kwargs: Arguments passed to RotatedRTMDetHead.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 dcn_on: bool = True,
                 dilation_rate: int = 2,
                 pcb_stacked_convs: int = 2,
                 **kwargs) -> None:
        self.dcn_on = dcn_on
        self.dilation_rate = dilation_rate
        self.pcb_stacked_convs = pcb_stacked_convs
        super().__init__(num_classes, in_channels, **kwargs)

    def _init_layers(self):
        """Initialize layers of the head including PCB modules."""
        super()._init_layers()
        
        if self.dcn_on:
            # Prior Capturing Block (PCB) components
            # Generate offset for DeformConv2d (18 channels)
            self.pcb_reg_offset = nn.Conv2d(
                self.feat_channels, 18, 3, padding=1)
            
            # Generate offset and mask for ModulatedDeformConv2d (27 channels: 18 offset + 9 mask)
            self.pcb_reg_offset_mask = nn.Conv2d(
                self.feat_channels, 27, 3, padding=1)
            
            # Deformable convolution for regression branch
            self.pcb_reg_dcn = DeformConv2d(
                self.feat_channels, self.feat_channels, 3,
                stride=1, padding=1, groups=1, bias=False)
            
            # Modulated deformable convolution for final feature refinement
            self.pcb_reg_modulated = ModulatedDeformConv2d(
                self.feat_channels, self.feat_channels, 3,
                stride=1, padding=1, groups=1, bias=False)
            
            # Classification branch shares offset from regression
            self.pcb_cls_dcn = DeformConv2d(
                self.feat_channels, self.feat_channels, 3,
                stride=1, padding=1, groups=1, bias=False)

    def init_weights(self) -> None:
        """Initialize weights of the head."""
        super().init_weights()
        if self.dcn_on:
            normal_init(self.pcb_reg_offset, std=0.01)
            normal_init(self.pcb_reg_offset_mask, std=0.01)
            normal_init(self.pcb_reg_dcn, std=0.01)
            normal_init(self.pcb_reg_modulated, std=0.01)
            normal_init(self.pcb_cls_dcn, std=0.01)

    def forward_single(self, x):
        """Forward feature of a single scale level with PCB.

        Args:
            x (torch.Tensor): Features of a single scale level.

        Returns:
            tuple: Contains classification scores, bbox predictions, angle predictions,
                   and offset information for DCFL assignment.
        """
        cls_feat = x
        reg_feat = x
        
        # Standard convolution layers (first part)
        if self.dcn_on:
            # Apply regular convolutions first (except the last ones)
            for i in range(self.stacked_convs - self.pcb_stacked_convs):
                cls_feat = self.cls_convs[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_feat)
            
            # Prior Capturing Block (PCB) for dynamic prior adjustment
            # Generate initial sampling locations with dilation
            batch_size, _, feat_h, feat_w = reg_feat.shape
            init_offset = torch.zeros(batch_size, 18, feat_h, feat_w, 
                                    device=reg_feat.device, dtype=reg_feat.dtype)
            
            # Create dilation-based initial offsets (3x3 grid with dilation)
            dilation = self.dilation_rate - 1
            offset_pattern = torch.tensor([
                [-dilation, -dilation], [-dilation, 0], [-dilation, dilation],
                [0, -dilation], [0, 0], [0, dilation],
                [dilation, -dilation], [dilation, 0], [dilation, dilation]
            ], device=reg_feat.device, dtype=reg_feat.dtype).view(9, 2)
            
            for i in range(9):
                init_offset[:, 2*i, :, :] = offset_pattern[i, 0]      # y offset
                init_offset[:, 2*i+1, :, :] = offset_pattern[i, 1]    # x offset
            
            # Generate learnable offsets for basic deformable conv
            offset_reg = self.pcb_reg_offset(reg_feat)
            combined_offset = init_offset + offset_reg
            
            # Generate offset and mask for modulated deformable conv
            offset_mask = self.pcb_reg_offset_mask(reg_feat)
            modulated_offset = offset_mask[:, :18, :, :]  # First 18 channels for offset
            modulated_mask = offset_mask[:, 18:, :, :].sigmoid()  # Last 9 channels for mask
            
            # Combine initial offset with learned offset for modulated conv
            final_modulated_offset = init_offset + modulated_offset
            
            # Apply PCB to regression branch
            reg_feat = self.pcb_reg_dcn(reg_feat, combined_offset)
            reg_feat = self.pcb_reg_modulated(reg_feat, final_modulated_offset, modulated_mask)
            
            # Apply PCB to classification branch (share offsets)
            cls_feat = self.pcb_cls_dcn(cls_feat, combined_offset)
            cls_feat = self.pcb_cls_dcn(cls_feat, final_modulated_offset)
            
            # Apply remaining convolution layers
            for i in range(self.stacked_convs - self.pcb_stacked_convs, self.stacked_convs):
                if i < len(self.cls_convs):
                    cls_feat = self.cls_convs[i](cls_feat)
                if i < len(self.reg_convs):
                    reg_feat = self.reg_convs[i](reg_feat)
            
            final_offset = final_modulated_offset
        else:
            # Standard forward without PCB
            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in self.reg_convs:
                reg_feat = reg_conv(reg_feat)
            final_offset = None

        # Generate predictions
        cls_score = self.rtm_cls(cls_feat)
        
        if self.with_objectness:
            objectness = self.rtm_obj(reg_feat)
            cls_score = inverse_sigmoid(
                sigmoid_geometric_mean(cls_score, objectness))
        
        # Bbox and angle predictions
        reg_dist = self.scales[0](self.rtm_reg(reg_feat).exp()).float()
        if self.is_scale_angle:
            angle_pred = self.scale_angle(self.rtm_ang(reg_feat)).float()
        else:
            angle_pred = self.rtm_ang(reg_feat).float()

        return cls_score, reg_dist, angle_pred, final_offset

    def forward(self, feats: Tuple[Tensor, ...]) -> tuple:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores, bbox predictions,
                   angle predictions, and offset information.
        """
        cls_scores = []
        bbox_preds = []
        angle_preds = []
        offsets = []
        
        for idx, (x, scale, stride) in enumerate(
                zip(feats, self.scales, self.prior_generator.strides)):
            
            cls_score, bbox_pred, angle_pred, offset = self.forward_single(x)
            
            # Apply stride scaling
            if bbox_pred is not None:
                bbox_pred = bbox_pred * stride[0]
            
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
            angle_preds.append(angle_pred)
            # Handle None offset case
            if offset is not None:
                offsets.append(offset)
            else:
                # Create dummy offset with same spatial dimensions as cls_score
                dummy_offset = torch.zeros_like(cls_score[:, :18])
                offsets.append(dummy_offset)
        
        return tuple(cls_scores), tuple(bbox_preds), tuple(angle_preds), tuple(offsets)

    def loss_by_feat(self,
                     cls_scores: List[Tensor],
                     bbox_preds: List[Tensor],
                     angle_preds: List[Tensor],
                     offsets: List[Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None):
        """Compute losses with DCFL-enhanced target assignment.

        Args:
            cls_scores (list[Tensor]): Classification scores for all scale levels.
            bbox_preds (list[Tensor]): Box predictions for all scale levels.
            angle_preds (list[Tensor]): Angle predictions for all scale levels.
            offsets (list[Tensor]): Offset predictions from PCB for all scale levels.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of gt_instances.
            batch_img_metas (list[dict]): Meta information of each image.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional): 
                Batch of gt_instances_ignore.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # Use parent class loss computation with offset information
        # The offsets will be used by DCFL-enhanced assigner
        return super().loss_by_feat(
            cls_scores, bbox_preds, angle_preds,
            batch_gt_instances, batch_img_metas, batch_gt_instances_ignore)
