from __future__ import annotations

import torch
import torch.nn.functional as F


def heatmap_mse_loss(
    pred_heatmap: torch.Tensor,
    gt_heatmap: torch.Tensor,
    positive_weight: float = 0.0,
    positive_threshold: float = 0.01,
) -> torch.Tensor:
    if positive_weight <= 0.0:
        return F.mse_loss(pred_heatmap, gt_heatmap)

    weight = torch.ones_like(gt_heatmap)
    weight = weight + positive_weight * (gt_heatmap > positive_threshold).to(gt_heatmap.dtype)
    return ((pred_heatmap - gt_heatmap).pow(2) * weight).mean()

