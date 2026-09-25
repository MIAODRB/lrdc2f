# mmrotate/models/task_modules/assigners/gmm_utils.py (Corrected Version 3 - Final)

import numpy as np
import torch

def cast_tensor(x, dtype=torch.float32, device=torch.device('cuda:0')):
    """Cast a tensor to a specified type and device."""
    return torch.from_numpy(x).to(device=device, dtype=dtype)

def obb2gaussian(obboxes):
    """Convert oriented bounding boxes to 2D Gaussian distributions.
    Args:
        obboxes (torch.Tensor): Oriented bounding boxes with shape (N, 5).
    Returns:
        g_obboxes (torch.Tensor): Gaussian representations of bounding boxes
            with shape (N, 5).
    """
    mu = obboxes[:, :2]
    w = obboxes[:, 2]
    h = obboxes[:, 3]
    theta = obboxes[:, 4]
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    v1_x = (w / 2) * cos_theta
    v1_y = (w / 2) * sin_theta
    v2_x = -(h / 2) * sin_theta
    v2_y = (h / 2) * cos_theta
    C11 = v1_x * v1_x + v2_x * v2_x
    C12 = v1_x * v1_y + v2_x * v2_y
    C22 = v1_y * v1_y + v2_y * v2_y

    lambda_val = (C11 + C22).sqrt().unsqueeze(1)
    theta_val = (torch.atan2(2 * C12, C11 - C22) / 2).unsqueeze(1)
    # Add a small epsilon to prevent sqrt(0) which can lead to NaN gradients
    d_val = (((C11 + C22).pow(2) - 4 * (C11 * C22 - C12 * C12)).clamp(min=1e-8).sqrt()).unsqueeze(1)

    g_obboxes = torch.cat((mu, lambda_val, theta_val, d_val), dim=1)
    return g_obboxes

def gjsd(g_bboxes1, g_bboxes2, alpha=0.5):
    """Calculate the Generalized Jensen-Shannon Divergence.
    Args:
        g_bboxes1 (torch.Tensor): Gaussian representations of bounding boxes
            with shape (N, 5).
        g_bboxes2 (torch.Tensor): Gaussian representations of bounding boxes
            with shape (M, 5).
        alpha (float): Weight of the two distributions.
    Returns:
        gjsd (torch.Tensor): Generalized Jensen-Shannon Divergence with
            shape (N, M).
    """
    mu1, lambda1, theta1, d1 = torch.split(g_bboxes1, [2, 1, 1, 1], dim=1)
    mu2, lambda2, theta2, d2 = torch.split(g_bboxes2, [2, 1, 1, 1], dim=1)

    # Broadcasting to shape [N, M, *]
    mu1 = mu1.unsqueeze(1)
    lambda1 = lambda1.unsqueeze(1)
    theta1 = theta1.unsqueeze(1)
    d1 = d1.unsqueeze(1)
    mu2 = mu2.unsqueeze(0)
    lambda2 = lambda2.unsqueeze(0)
    theta2 = theta2.unsqueeze(0)
    d2 = d2.unsqueeze(0)

    eps = 1e-8
    lambda_alpha = (1 - alpha) * lambda1 + alpha * lambda2
    delta_theta = theta1 - theta2
    d_alpha_2 = (1 - alpha)**2 * d1**2 + alpha**2 * d2**2 + \
                2 * (1 - alpha) * alpha * d1 * d2 * torch.cos(2 * delta_theta)
    d_alpha = d_alpha_2.sqrt()

    log_term1 = torch.log((lambda_alpha**2 - d_alpha_2).abs() + eps)
    log_term2 = torch.log((lambda1**2 - d1**2).abs().pow(1-alpha) + eps)
    log_term3 = torch.log((lambda2**2 - d2**2).abs().pow(alpha) + eps)
    v_alpha = 0.5 * (log_term1 - log_term2 - log_term3)

    # Re-implement mu_alpha and m_alpha calculation for clarity and correctness
    # Sigma_1 and Sigma_2 are covariance matrices for g_bboxes1 and g_bboxes2
    C1_11 = (lambda1+d1)*torch.cos(theta1)**2 + (lambda1-d1)*torch.sin(theta1)**2
    C1_12 = d1 * torch.sin(2*theta1)
    C1_22 = (lambda1-d1)*torch.cos(theta1)**2 + (lambda1+d1)*torch.sin(theta1)**2
    Sigma1 = torch.cat([C1_11, C1_12, C1_12, C1_22], dim=-1).reshape(-1, 1, 2, 2)

    C2_11 = (lambda2+d2)*torch.cos(theta2)**2 + (lambda2-d2)*torch.sin(theta2)**2
    C2_12 = d2 * torch.sin(2*theta2)
    C2_22 = (lambda2-d2)*torch.cos(theta2)**2 + (lambda2+d2)*torch.sin(theta2)**2
    Sigma2 = torch.cat([C2_11, C2_12, C2_12, C2_22], dim=-1).reshape(1, -1, 2, 2)

    # Inverse of covariance matrices
    det1 = (Sigma1[..., 0, 0] * Sigma1[..., 1, 1] - Sigma1[..., 0, 1] * Sigma1[..., 1, 0]).unsqueeze(-1).unsqueeze(-1)
    inv_Sigma1 = torch.cat([Sigma1[..., 1, 1, None], -Sigma1[..., 0, 1, None], -Sigma1[..., 1, 0, None], Sigma1[..., 0, 0, None]], dim=-1).reshape(-1, 1, 2, 2) / (det1 + eps)

    det2 = (Sigma2[..., 0, 0] * Sigma2[..., 1, 1] - Sigma2[..., 0, 1] * Sigma2[..., 1, 0]).unsqueeze(-1).unsqueeze(-1)
    inv_Sigma2 = torch.cat([Sigma2[..., 1, 1, None], -Sigma2[..., 0, 1, None], -Sigma2[..., 1, 0, None], Sigma2[..., 0, 0, None]], dim=-1).reshape(1, -1, 2, 2) / (det2 + eps)

    Sigma_alpha = torch.inverse((1 - alpha) * inv_Sigma1 + alpha * inv_Sigma2 + torch.eye(2).to(mu1.device) * eps)

    # Calculate mu_alpha
    mu1_T_inv_S1 = ((1 - alpha) * mu1.unsqueeze(-2) @ inv_Sigma1).squeeze(-2)
    mu2_T_inv_S2 = (alpha * mu2.unsqueeze(-2) @ inv_Sigma2).squeeze(-2)
    mu_alpha = ((mu1_T_inv_S1 + mu2_T_inv_S2).unsqueeze(-2) @ Sigma_alpha).squeeze(-2)

    # Calculate m_alpha
    m_alpha = 0.5 * (
        ((1 - alpha) * mu1.unsqueeze(-2) @ inv_Sigma1 @ mu1.unsqueeze(-1)).squeeze(-1).squeeze(-1) +
        ((alpha) * mu2.unsqueeze(-2) @ inv_Sigma2 @ mu2.unsqueeze(-1)).squeeze(-1).squeeze(-1) -
        (mu_alpha.unsqueeze(-2) @ torch.inverse(Sigma_alpha + torch.eye(2).to(mu1.device) * eps) @ mu_alpha.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    )

    gjsd_val = v_alpha.squeeze(-1) + m_alpha

    return gjsd_val.squeeze(-1)

def gmm_pdf(priors, g_gt, weight):
    """Calculate the probability density of GMM.
    Args:
        priors (torch.Tensor): Priors with shape (N, 2).
        g_gt (torch.Tensor): Gaussian representation of a gt bbox
            with shape (5,).
        weight (float): Weight of the GMM.
    Returns:
        prob (torch.Tensor): Probability density with shape (N,).
    """
    mu, lambda1, theta1, d1 = torch.split(g_gt, [2, 1, 1, 1], dim=0)
    delta_mu = priors - mu
    cos_theta = torch.cos(theta1)
    sin_theta = torch.sin(theta1)
    R_theta = torch.cat(
        (torch.cat((cos_theta, -sin_theta), dim=-1).unsqueeze(-1),
         torch.cat((sin_theta, cos_theta), dim=-1).unsqueeze(-1)),
        dim=-1)

    # Add epsilon for numerical stability
    eps = 1e-8
    C_inv = R_theta @ torch.cat(
        (torch.cat((1 / (lambda1 + d1 + eps), torch.tensor([0.0]).to(priors)),
                   dim=-1).unsqueeze(-1),
         torch.cat((torch.tensor([0.0]).to(priors), 1 / (lambda1 - d1 + eps)),
                   dim=-1).unsqueeze(-1)),
        dim=-1) @ R_theta.transpose(-1, -2)

    det_sqrt = ((lambda1**2 - d1**2).abs() + eps).sqrt()

    prob = weight * torch.exp(
        -0.5 * (delta_mu.unsqueeze(1) @ C_inv @ delta_mu.unsqueeze(-1))
    ) / (2 * np.pi * det_sqrt).squeeze()

    return prob