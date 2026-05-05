"""
Losses used by the OTTA paper reproduction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def source_heatmap_loss(pred_heatmap: torch.Tensor, gt_heatmap: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_heatmap, gt_heatmap)


def self_training_loss(pred_heatmap: torch.Tensor, pseudo_heatmap: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_heatmap, pseudo_heatmap)


def compute_quality_weight(student_heatmap: torch.Tensor, tau: float = 0.7) -> torch.Tensor:
    peaks = student_heatmap.flatten(2).amax(dim=-1)
    return F.softmax(peaks / tau, dim=1)


def masked_heatmap_consistency_loss(
    student_heatmap: torch.Tensor,
    teacher_heatmap: torch.Tensor,
    tau: float = 0.7,
) -> torch.Tensor:
    quality = compute_quality_weight(student_heatmap, tau=tau)
    mse_per_keypoint = (student_heatmap - teacher_heatmap).pow(2).mean(dim=(-1, -2))
    return (quality * mse_per_keypoint).mean()


def class_awareness_consistency_loss(
    student_cls: torch.Tensor,
    teacher_cls: torch.Tensor,
    source_prototype: torch.Tensor,
) -> torch.Tensor:
    target_feature = torch.cat([student_cls, teacher_cls], dim=-1)
    target_feature = F.normalize(target_feature, dim=-1)
    source_prototype = F.normalize(source_prototype, dim=0)
    cosine_similarity = (target_feature * source_prototype.unsqueeze(0)).sum(dim=-1)
    return 1.0 - cosine_similarity.mean()


def total_target_loss(
    loss_st: torch.Tensor,
    loss_mh: torch.Tensor,
    loss_ca: torch.Tensor,
    lambda_st: float = 10.0,
    lambda_ca: float = 0.01,
) -> torch.Tensor:
    return loss_mh + lambda_st * loss_st + lambda_ca * loss_ca

