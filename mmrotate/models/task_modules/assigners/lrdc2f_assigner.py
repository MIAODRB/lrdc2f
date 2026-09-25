"""LRDC2F coarse-to-fine label assignment for lunar rockfall detection.

Optimizations for small, sparse targets:
1. Mahalanobis-based DGMM for numerical stability
2. Progressive thresholding for early training
3. Increased candidate pool for sparse distribution
4. Adaptive minimum positive samples
"""

import torch
import torch.nn.functional as F
from mmdet.models.task_modules.assigners import BaseAssigner, AssignResult
from mmrotate.registry import TASK_UTILS
from mmrotate.structures.bbox import rbbox_overlaps

try:
    from mmrotate.structures.gaussian_tools import (
        xy_wh_r_2_xy_sigma, compute_gjsd, DynamicGaussianMixtureModel,
        point_to_rotated_box
    )
except ImportError:
    raise ImportError("Please ensure gaussian_tools.py is in mmrotate/structures/")


@TASK_UTILS.register_module(name=['LRDC2FAssigner', 'DCFLAssigner'])
class LRDC2FAssigner(BaseAssigner):
    """Coarse-to-fine assigner optimized for small, sparse rockfalls.

    ``DCFLAssigner`` remains a registry alias for historical experiment configs.
    
    Args:
        num_cps (int): Coarse Positive Samples. Default: 30 (increased for sparse targets)
        num_mps (int): Medium Positive Samples. Default: 20 (increased)
        dgmm_threshold (float): DGMM threshold. Default: 0.15 (relaxed for small targets)
        dgmm_threshold_warmup_iters (int): Iterations for threshold warmup. Default: 1000
        dgmm_threshold_final (float): Final threshold after warmup. Default: 0.3
        gjsd_alpha (float): GJSD mixture weight. Default: 0.5
        dgmm_weight_geo (float): DGMM geometry weight. Default: 0.8 (favor geometry for small targets)
        dgmm_weight_sem (float): DGMM semantic weight. Default: 0.2
        min_pos_samples (int): Minimum positive samples per GT. Default: 5 (increased for small targets)
        adapt_min_pos_by_size (bool): Adapt min_pos based on GT size. Default: True
    """
    
    def __init__(self,
                 num_cps=30,
                 num_mps=20,
                 dgmm_threshold=0.15,
                 dgmm_threshold_warmup_iters=1000,
                 dgmm_threshold_final=0.3,
                 gjsd_alpha=0.5,
                 dgmm_weight_geo=0.8,
                 dgmm_weight_sem=0.2,
                 min_pos_samples=5,
                 adapt_min_pos_by_size=True,
                 debug=False):
        self.num_cps = num_cps
        self.num_mps = num_mps
        self.dgmm_threshold_init = dgmm_threshold
        self.dgmm_threshold_warmup_iters = dgmm_threshold_warmup_iters
        self.dgmm_threshold_final = dgmm_threshold_final
        self.gjsd_alpha = gjsd_alpha
        self.min_pos_samples = min_pos_samples
        self.adapt_min_pos_by_size = adapt_min_pos_by_size
        self.debug = debug
        
        self.iter_count = 0
        
        # Initialize DGMM with geometry preference for small targets
        self.dgmm = DynamicGaussianMixtureModel(
            weight_geometry=dgmm_weight_geo,
            weight_semantic=dgmm_weight_sem
        )
    
    def _get_adaptive_dgmm_threshold(self):
        """Progressive threshold: start lenient, gradually stricten."""
        if self.iter_count < self.dgmm_threshold_warmup_iters:
            # Linear warmup
            progress = self.iter_count / self.dgmm_threshold_warmup_iters
            threshold = self.dgmm_threshold_init + \
                       (self.dgmm_threshold_final - self.dgmm_threshold_init) * progress
        else:
            threshold = self.dgmm_threshold_final
        
        return threshold
    
    def _get_adaptive_min_pos(self, gt_bboxes):
        """Adapt minimum positive samples based on GT size.
        
        Smaller targets need more samples for robust training.
        """
        if not self.adapt_min_pos_by_size:
            return self.min_pos_samples
        
        # Compute GT areas
        gt_areas = gt_bboxes[:, 2] * gt_bboxes[:, 3]  # w * h
        
        # Area thresholds (adjust based on your data)
        # For NAC lunar images, small rockfalls might be 5-15 pixels
        small_area = 150    # e.g., 10×15 pixels
        medium_area = 500   # e.g., 20×25 pixels
        
        min_pos_per_gt = []
        for area in gt_areas:
            if area < small_area:
                min_pos_per_gt.append(8)  # Extra samples for tiny targets
            elif area < medium_area:
                min_pos_per_gt.append(6)
            else:
                min_pos_per_gt.append(4)
        
        return min_pos_per_gt
    
    def assign(self, pred_instances, gt_instances, gt_instances_ignore=None, **kwargs):
        """Assign gt to priors using DCFL strategy."""
        # Convert BaseBoxes to tensor if needed
        gt_bboxes = gt_instances.bboxes
        if hasattr(gt_bboxes, 'tensor'):
            gt_bboxes = gt_bboxes.tensor
        
        gt_labels = gt_instances.labels
        priors = pred_instances.priors
        pred_scores = pred_instances.scores
        
        pred_bboxes = pred_instances.bboxes
        if hasattr(pred_bboxes, 'tensor'):
            pred_bboxes = pred_bboxes.tensor
        
        num_priors = priors.shape[0]
        num_gts = gt_bboxes.shape[0]
        
        # No gt
        if num_gts == 0:
            assigned_gt_inds = priors.new_zeros((num_priors,), dtype=torch.long)
            max_overlaps = priors.new_zeros((num_priors,))
            assigned_labels = priors.new_full((num_priors,), -1, dtype=torch.long)
            return AssignResult(num_gts=num_gts, gt_inds=assigned_gt_inds,
                              max_overlaps=max_overlaps, labels=assigned_labels)
        
        # Get adaptive parameters
        dgmm_threshold = self._get_adaptive_dgmm_threshold()
        min_pos_per_gt = self._get_adaptive_min_pos(gt_bboxes)
        
        if self.debug and self.iter_count % 50 == 0:
            print(f"\n[LRDC2F Debug] Iter {self.iter_count}")
            print(f"  Adaptive DGMM threshold: {dgmm_threshold:.4f}")
            print(f"  GT sizes: {gt_bboxes[:, 2:4].tolist()}")
            print(f"  Min pos samples per GT: {min_pos_per_gt}")
        
        # Stage 1: CPS Selection
        cps_inds, cps_gt_inds = self._select_coarse_positive_samples(priors, gt_bboxes)
        
        if self.debug and self.iter_count % 50 == 0:
            print(f"  Stage 1 (CPS): {cps_inds.numel()} samples")
        
        if cps_inds.numel() == 0:
            assigned_gt_inds = priors.new_zeros((num_priors,), dtype=torch.long)
            max_overlaps = priors.new_zeros((num_priors,))
            assigned_labels = priors.new_full((num_priors,), -1, dtype=torch.long)
            return AssignResult(num_gts=num_gts, gt_inds=assigned_gt_inds,
                              max_overlaps=max_overlaps, labels=assigned_labels)
        
        # Stage 2: MPS Re-ranking
        mps_inds, mps_gt_inds = self._rerank_medium_positive_samples(
            cps_inds, cps_gt_inds, pred_scores, pred_bboxes, gt_bboxes, gt_labels
        )
        
        if self.debug and self.iter_count % 50 == 0:
            print(f"  Stage 2 (MPS): {mps_inds.numel()} samples")
        
        # Stage 3: FPS Filtering
        fps_inds, fps_gt_inds, dgmm_scores = self._filter_finer_positive_samples(
            mps_inds, mps_gt_inds, priors, gt_bboxes, dgmm_threshold
        )
        
        if self.debug and self.iter_count % 50 == 0:
            print(f"  Stage 3 (FPS): {fps_inds.numel()} samples")
            if dgmm_scores is not None:
                print(f"  DGMM scores: min={dgmm_scores.min():.4f}, "
                      f"max={dgmm_scores.max():.4f}, mean={dgmm_scores.mean():.4f}")
                print(f"  Samples above threshold: {(dgmm_scores > dgmm_threshold).sum().item()}")
        
        # Ensure minimum positive samples (adaptive)
        fps_inds, fps_gt_inds = self._ensure_min_pos_samples_adaptive(
            fps_inds, fps_gt_inds, mps_inds, mps_gt_inds, num_gts, min_pos_per_gt
        )
        
        if self.debug and self.iter_count % 50 == 0:
            print(f"  Final positive samples: {fps_inds.numel()}")
        
        self.iter_count += 1
        
        # Create assignment result
        assigned_gt_inds = priors.new_zeros((num_priors,), dtype=torch.long)
        assigned_gt_inds[fps_inds] = fps_gt_inds + 1
        
        max_overlaps = priors.new_zeros((num_priors,))
        if fps_inds.numel() > 0:
            ious = rbbox_overlaps(pred_bboxes[fps_inds], gt_bboxes[fps_gt_inds],
                                mode='iou', is_aligned=True)
            max_overlaps[fps_inds] = ious
        
        assigned_labels = priors.new_full((num_priors,), -1, dtype=torch.long)
        pos_inds = assigned_gt_inds > 0
        if pos_inds.any():
            assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        assigned_labels[assigned_gt_inds == 0] = 0
        
        return AssignResult(num_gts=num_gts, gt_inds=assigned_gt_inds,
                          max_overlaps=max_overlaps, labels=assigned_labels)
    
    def _select_coarse_positive_samples(self, priors, gt_bboxes):
        """Stage 1: CPS Selection via GJSD."""
        num_priors = priors.shape[0]
        num_gts = gt_bboxes.shape[0]
        prior_dim = priors.shape[-1]
        
        if prior_dim in [2, 4]:
            gt_sizes = gt_bboxes[:, 2:4]
            mean_w = gt_sizes[:, 0].mean().item()
            mean_h = gt_sizes[:, 1].mean().item()
            base_size = (max(mean_w * 0.5, 4.0), max(mean_h * 0.5, 4.0))  # Allow smaller sizes
            priors_boxes = point_to_rotated_box(priors, stride=None, base_sizes=base_size, angle=0.0)
        elif prior_dim == 5:
            priors_boxes = priors
        else:
            raise ValueError(f"Unsupported prior dimension: {prior_dim}")
        
        prior_center, prior_sigma = xy_wh_r_2_xy_sigma(priors_boxes)
        gt_center, gt_sigma = xy_wh_r_2_xy_sigma(gt_bboxes)
        gjsd_dist = compute_gjsd(prior_center, prior_sigma, gt_center, gt_sigma, alpha=self.gjsd_alpha)
        
        cps_inds_list = []
        cps_gt_inds_list = []
        
        for gt_idx in range(num_gts):
            gjsd_scores = gjsd_dist[:, gt_idx]
            
            if num_priors >= self.num_cps:
                _, topk_inds = torch.topk(-gjsd_scores, k=self.num_cps, largest=True)
            else:
                topk_inds = torch.arange(num_priors, device=priors.device)
            
            cps_inds_list.append(topk_inds)
            cps_gt_inds_list.append(topk_inds.new_full((topk_inds.shape[0],), gt_idx))
        
        return torch.cat(cps_inds_list), torch.cat(cps_gt_inds_list)
    
    def _rerank_medium_positive_samples(self, cps_inds, cps_gt_inds, pred_scores,
                                       pred_bboxes, gt_bboxes, gt_labels):
        """Stage 2: MPS Re-ranking via PT."""
        num_gts = gt_bboxes.shape[0]
        mps_inds_list = []
        mps_gt_inds_list = []
        
        for gt_idx in range(num_gts):
            gt_cps_mask = cps_gt_inds == gt_idx
            gt_cps_inds = cps_inds[gt_cps_mask]
            
            if gt_cps_inds.numel() == 0:
                continue
            
            gt_label = gt_labels[gt_idx]
            cls_scores = pred_scores[gt_cps_inds, gt_label]
            ious = rbbox_overlaps(pred_bboxes[gt_cps_inds],
                                gt_bboxes[gt_idx:gt_idx+1].expand(gt_cps_inds.shape[0], -1),
                                mode='iou', is_aligned=True)
            
            # For small targets, IoU is often low in early training
            # Give more weight to classification score
            pt_scores = 0.4 * cls_scores + 0.6 * ious
            
            num_candidates = min(self.num_mps, gt_cps_inds.shape[0])
            _, topq_inds = torch.topk(pt_scores, k=num_candidates, largest=True)
            
            mps_inds_list.append(gt_cps_inds[topq_inds])
            mps_gt_inds_list.append(topq_inds.new_full((topq_inds.shape[0],), gt_idx))
        
        if len(mps_inds_list) == 0:
            return cps_inds.new_zeros((0,)), cps_gt_inds.new_zeros((0,))
        
        return torch.cat(mps_inds_list), torch.cat(mps_gt_inds_list)
    
    def _filter_finer_positive_samples(self, mps_inds, mps_gt_inds, priors, gt_bboxes, threshold):
        """Stage 3: FPS Filtering via DGMM."""
        if mps_inds.numel() == 0:
            return mps_inds, mps_gt_inds, None
        
        num_gts = gt_bboxes.shape[0]
        prior_dim = priors.shape[-1]
        
        if prior_dim in [2]:
            sample_locations = priors[mps_inds, :]
        elif prior_dim in [4, 5]:
            sample_locations = priors[mps_inds, :2]
        else:
            raise ValueError(f"Unsupported prior dimension: {prior_dim}")
        
        mps_locations_list = []
        for gt_idx in range(num_gts):
            gt_mps_mask = mps_gt_inds == gt_idx
            gt_mps_inds = mps_inds[gt_mps_mask]
            if gt_mps_inds.numel() > 0:
                mps_locations_list.append(priors[gt_mps_inds, :2])
            else:
                mps_locations_list.append(gt_bboxes[gt_idx:gt_idx+1, :2])
        
        max_len = max(loc.shape[0] for loc in mps_locations_list)
        mps_locations_padded = []
        for loc in mps_locations_list:
            if loc.shape[0] < max_len:
                padding = loc.new_zeros((max_len - loc.shape[0], 2))
                padding[:] = loc[0]
                loc = torch.cat([loc, padding], dim=0)
            mps_locations_padded.append(loc)
        
        mps_locations = torch.stack(mps_locations_padded, dim=0)
        dgmm_scores = self.dgmm.compute_score(sample_locations, gt_bboxes, mps_locations)
        
        dgmm_scores_aligned = dgmm_scores[
            torch.arange(mps_inds.shape[0], device=mps_inds.device),
            mps_gt_inds
        ]
        
        fps_mask = dgmm_scores_aligned > threshold
        return mps_inds[fps_mask], mps_gt_inds[fps_mask], dgmm_scores_aligned
    
    def _ensure_min_pos_samples_adaptive(self, fps_inds, fps_gt_inds, mps_inds, mps_gt_inds,
                                        num_gts, min_pos_per_gt):
        """Ensure adaptive minimum positive samples per GT."""
        if mps_inds.numel() == 0:
            return fps_inds, fps_gt_inds
        
        # Check if each GT has enough samples
        fps_inds_list = []
        fps_gt_inds_list = []
        
        for gt_idx in range(num_gts):
            # Current FPS for this GT
            gt_fps_mask = fps_gt_inds == gt_idx
            gt_fps_count = gt_fps_mask.sum().item()
            
            min_required = min_pos_per_gt[gt_idx] if isinstance(min_pos_per_gt, list) else min_pos_per_gt
            
            if gt_fps_count >= min_required:
                # Already have enough
                fps_inds_list.append(fps_inds[gt_fps_mask])
                fps_gt_inds_list.append(fps_gt_inds[gt_fps_mask])
            else:
                # Need to add more from MPS
                gt_mps_mask = mps_gt_inds == gt_idx
                gt_mps_inds = mps_inds[gt_mps_mask]
                
                if gt_mps_inds.numel() == 0:
                    continue
                
                num_to_take = min(min_required, gt_mps_inds.shape[0])
                selected_inds = gt_mps_inds[:num_to_take]
                
                fps_inds_list.append(selected_inds)
                fps_gt_inds_list.append(selected_inds.new_full((num_to_take,), gt_idx))
        
        if len(fps_inds_list) == 0:
            return mps_inds.new_zeros((0,)), mps_gt_inds.new_zeros((0,))
        
        return torch.cat(fps_inds_list), torch.cat(fps_gt_inds_list)
