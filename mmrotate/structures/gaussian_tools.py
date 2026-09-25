"""Gaussian tools with semantic-only DGMM for rockfall detection.

This version ignores geometric center and relies entirely on semantic center
computed from MPS (Medium Positive Samples).
"""

import torch
import torch.nn.functional as F
import numpy as np


def point_to_rotated_box(points, stride=None, base_sizes=(64, 64), angle=0.0):
    """Convert point priors to rotated boxes."""
    N = points.shape[0]
    device = points.device
    
    if points.shape[-1] == 2:
        xy = points
    elif points.shape[-1] == 4:
        xy = points[:, :2]
        if stride is None and not isinstance(base_sizes, tuple):
            strides = points[:, 2:4]
            w = (strides[:, 0] * 2).unsqueeze(-1)
            h = (strides[:, 1] * 2).unsqueeze(-1)
        else:
            w = torch.full((N, 1), base_sizes[0], device=device, dtype=points.dtype)
            h = torch.full((N, 1), base_sizes[1], device=device, dtype=points.dtype)
    else:
        raise ValueError(f"Expected points shape (N, 2) or (N, 4), got {points.shape}")
    
    if not (points.shape[-1] == 4 and stride is None):
        if isinstance(base_sizes, tuple):
            w = torch.full((N, 1), base_sizes[0], device=device, dtype=points.dtype)
            h = torch.full((N, 1), base_sizes[1], device=device, dtype=points.dtype)
    
    theta = torch.full((N, 1), angle, device=device, dtype=points.dtype)
    boxes = torch.cat([xy, w, h, theta], dim=-1)
    return boxes


def xy_wh_r_2_xy_sigma(xywhr):
    """Convert rotated box to Gaussian parameters."""
    _shape = xywhr.shape
    assert _shape[-1] == 5
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = xywhr[..., 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = torch.diag_embed(wh.square() / 4)
    sigma = R @ S @ R.transpose(-1, -2)
    
    return xy.reshape(-1, 2), sigma.reshape(*_shape[:-1], 2, 2)


def compute_kl_divergence_aligned(mu_p, sigma_p, mu_q, sigma_q):
    """Compute aligned KL divergence."""
    N = mu_p.shape[0]
    det_p = torch.det(sigma_p).clamp(min=1e-7)
    det_q = torch.det(sigma_q).clamp(min=1e-7)
    
    eye_2d = torch.eye(2, device=sigma_q.device).unsqueeze(0).expand(N, -1, -1)
    sigma_q_inv = torch.inverse(sigma_q + eye_2d * 1e-6)
    
    trace_term = torch.einsum('nij,nji->n', sigma_q_inv, sigma_p)
    diff = (mu_q - mu_p).unsqueeze(-1)
    mahal_dist = torch.einsum('nij,njk,nk->n', 
                              diff.transpose(-1, -2), sigma_q_inv, diff.squeeze(-1))
    
    kl = 0.5 * (torch.log(det_q / det_p) + trace_term + mahal_dist - 2)
    return kl.clamp(min=0)


def compute_gjsd(mu_p, sigma_p, mu_q, sigma_q, alpha=0.5):
    """Compute GJSD."""
    N = mu_p.shape[0]
    M = mu_q.shape[0]
    
    mu_p_exp = mu_p.unsqueeze(1).expand(N, M, 2)
    sigma_p_exp = sigma_p.unsqueeze(1).expand(N, M, 2, 2)
    mu_q_exp = mu_q.unsqueeze(0).expand(N, M, 2)
    sigma_q_exp = sigma_q.unsqueeze(0).expand(N, M, 2, 2)
    
    eye_2d = torch.eye(2, device=sigma_p.device)
    sigma_p_inv = torch.inverse(sigma_p_exp + eye_2d * 1e-6)
    sigma_q_inv = torch.inverse(sigma_q_exp + eye_2d * 1e-6)
    
    sigma_alpha_inv = (1 - alpha) * sigma_p_inv + alpha * sigma_q_inv
    sigma_alpha = torch.inverse(sigma_alpha_inv + eye_2d * 1e-6)
    
    term_p = torch.einsum('nmij,nmj->nmi', sigma_p_inv, mu_p_exp)
    term_q = torch.einsum('nmij,nmj->nmi', sigma_q_inv, mu_q_exp)
    mu_alpha = torch.einsum('nmij,nmj->nmi', sigma_alpha, 
                           (1 - alpha) * term_p + alpha * term_q)
    
    kl_alpha_p = compute_kl_divergence_aligned(
        mu_alpha.reshape(-1, 2), sigma_alpha.reshape(-1, 2, 2),
        mu_p_exp.reshape(-1, 2), sigma_p_exp.reshape(-1, 2, 2)
    ).reshape(N, M)
    
    kl_alpha_q = compute_kl_divergence_aligned(
        mu_alpha.reshape(-1, 2), sigma_alpha.reshape(-1, 2, 2),
        mu_q_exp.reshape(-1, 2), sigma_q_exp.reshape(-1, 2, 2)
    ).reshape(N, M)
    
    gjsd = (1 - alpha) * kl_alpha_p + alpha * kl_alpha_q
    return gjsd.clamp(min=0)


class DynamicGaussianMixtureModel:
    """DGMM using ONLY semantic center for rockfall detection.
    
    Key change: Ignore geometric center (which is on track middle),
    only use semantic center from MPS (which should be on rock).
    """
    
    def __init__(self, weight_geometry=0.0, weight_semantic=1.0):
        """Initialize semantic-only DGMM.
        
        Args:
            weight_geometry: Weight for geometry center (set to 0 for rockfall)
            weight_semantic: Weight for semantic center (set to 1 for rockfall)
        """
        self.w_geo = weight_geometry
        self.w_sem = weight_semantic
        
        # Sanity check
        if self.w_geo > 0:
            print("WARNING: Using geometric center for rockfall detection may be problematic!")
    
    def compute_score(self, sample_locations, gt_boxes, mps_locations):
        """Compute DGMM score using only semantic center.
        
        For rockfall: Semantic center (average MPS) should be near rock,
        while geometric center is in track middle (wrong!).
        
        Args:
            sample_locations (Tensor): Shape (N, 2)
            gt_boxes (Tensor): Shape (M, 5)
            mps_locations (Tensor): Shape (M, K, 2) - semantic centers
        
        Returns:
            Tensor: DGMM scores, shape (N, M)
        """
        # Get GT Gaussian parameters (for covariance only)
        gt_center, gt_sigma = xy_wh_r_2_xy_sigma(gt_boxes)
        
        # Compute semantic center from MPS
        semantic_center = mps_locations.mean(dim=1)  # (M, 2)
        
        N = sample_locations.shape[0]
        M = gt_boxes.shape[0]
        
        samples = sample_locations.unsqueeze(1).expand(N, M, 2)
        sem_center = semantic_center.unsqueeze(0).expand(N, M, 2)
        sigma = gt_sigma.unsqueeze(0).expand(N, M, 2, 2)
        
        # Compute inverse covariance
        sigma_inv = torch.inverse(sigma + torch.eye(2, device=sigma.device) * 1e-6)
        
        # Semantic center score using Mahalanobis distance
        diff_sem = (samples - sem_center).unsqueeze(-1)
        mahal_sem = torch.einsum('nmij,nmjk,nmk->nm',
                                diff_sem.transpose(-1, -2), sigma_inv,
                                diff_sem.squeeze(-1))
        
        # DGMM score: purely semantic
        dgmm_score = torch.exp(-0.5 * mahal_sem)
        
        # If using mixed (not recommended for rockfall)
        if self.w_geo > 0:
            geo_center = gt_center.unsqueeze(0).expand(N, M, 2)
            diff_geo = (samples - geo_center).unsqueeze(-1)
            mahal_geo = torch.einsum('nmij,nmjk,nmk->nm',
                                    diff_geo.transpose(-1, -2), sigma_inv, 
                                    diff_geo.squeeze(-1))
            score_geo = torch.exp(-0.5 * mahal_geo)
            dgmm_score = self.w_geo * score_geo + self.w_sem * dgmm_score
        
        return dgmm_score