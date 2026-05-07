from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.crop_and_heatmap import load_camera
from dinov2_heatmap_otta.adapt.triggered_single_model_tta_dino_heatmap import (
    configure_trainable_parameters,
    confidence_weighted_regularization,
    diagnose_prediction,
    geometry_target_from_pose,
    load_source_model,
    make_dataset,
    tensor_bbox_to_tuple,
    tensor_image_name,
    weighted_heatmap_mse,
)
from dinov2_heatmap_otta.models.dino_pose_model import DinoHeatmapPoseModel


@dataclass
class FeatureMemoryEntry:
    image: torch.Tensor
    pseudo_heatmap: torch.Tensor
    feature: torch.Tensor
    quality: float
    image_name: str


class FeatureQualityMemoryBank:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.entries: List[FeatureMemoryEntry] = []

    def __len__(self) -> int:
        return len(self.entries)

    def push(
        self,
        image: torch.Tensor,
        pseudo_heatmap: torch.Tensor,
        feature: torch.Tensor,
        quality: float,
        image_name: str,
    ) -> None:
        entry = FeatureMemoryEntry(
            image=image.detach().cpu().clone(),
            pseudo_heatmap=pseudo_heatmap.detach().cpu().clone(),
            feature=F.normalize(feature.detach().cpu().flatten(), dim=0).clone(),
            quality=float(quality),
            image_name=image_name,
        )

        if len(self.entries) < self.capacity:
            self.entries.append(entry)
            return

        worst_index = int(np.argmin([item.quality for item in self.entries]))
        if entry.quality > self.entries[worst_index].quality:
            self.entries[worst_index] = entry

    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        if len(self.entries) == 0:
            raise RuntimeError("Feature memory is empty.")

        qualities = np.asarray([max(item.quality, 1e-6) for item in self.entries], dtype=np.float64)
        probs = qualities / qualities.sum()
        replace = len(self.entries) < batch_size
        indices = np.random.choice(np.arange(len(self.entries)), size=batch_size, replace=replace, p=probs)

        images = torch.stack([self.entries[int(index)].image for index in indices], dim=0).to(device)
        pseudo_heatmaps = torch.stack([self.entries[int(index)].pseudo_heatmap for index in indices], dim=0).to(device)
        weights = torch.tensor([self.entries[int(index)].quality for index in indices], dtype=torch.float32, device=device)
        weights = weights / weights.mean().clamp_min(1e-6)
        return images, pseudo_heatmaps, weights, indices

    def update_if_better(self, indices: np.ndarray, pred_heatmaps: torch.Tensor, qualities: torch.Tensor) -> None:
        pred_heatmaps = pred_heatmaps.detach().cpu()
        qualities = qualities.detach().cpu()
        for slot, memory_index in enumerate(indices):
            index = int(memory_index)
            quality = float(qualities[slot].item())
            if quality > self.entries[index].quality:
                self.entries[index].pseudo_heatmap = pred_heatmaps[slot].clone()
                self.entries[index].quality = quality

    def feature_distances(self, feature: torch.Tensor, source_prototype: torch.Tensor | None) -> torch.Tensor:
        feature = F.normalize(feature.detach().flatten(), dim=0).cpu()
        if source_prototype is None:
            source_distance = torch.tensor(0.0)
        else:
            source_distance = 1.0 - torch.sum(feature * F.normalize(source_prototype.cpu().flatten(), dim=0))

        if len(self.entries) == 0:
            memory_proto_distance = torch.tensor(1.0)
            nearest_memory_distance = torch.tensor(1.0)
        else:
            memory_features = torch.stack([item.feature for item in self.entries], dim=0)
            memory_proto = F.normalize(memory_features.mean(dim=0), dim=0)
            memory_proto_distance = 1.0 - torch.sum(feature * memory_proto)
            nearest_memory_distance = torch.min(1.0 - torch.matmul(memory_features, feature))

        return torch.stack([source_distance, memory_proto_distance, nearest_memory_distance]).float()

    def summary(self) -> Dict[str, float]:
        if len(self.entries) == 0:
            return {"memory_size": 0}
        qualities = np.asarray([item.quality for item in self.entries], dtype=np.float64)
        return {
            "memory_size": int(len(self.entries)),
            "memory_quality_mean": float(qualities.mean()),
            "memory_quality_min": float(qualities.min()),
            "memory_quality_max": float(qualities.max()),
        }


class DualBranchRiskGate(nn.Module):
    def __init__(self, geo_dim: int = 8, feat_dim: int = 3, hidden_dim: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.geo_branch = nn.Sequential(
            nn.Linear(geo_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.feat_branch = nn.Sequential(
            nn.Linear(feat_dim, max(8, hidden_dim // 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(max(8, hidden_dim // 2), 1),
        )
        self.fusion = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
        )

    def forward(self, geo_features: torch.Tensor, feat_features: torch.Tensor) -> torch.Tensor:
        geo_logit = self.geo_branch(geo_features)
        feat_logit = self.feat_branch(feat_features)
        return self.fusion(torch.cat([geo_logit, feat_logit], dim=-1)).squeeze(-1)


def safe_float(value, default: float = 0.0, cap: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default
    if not np.isfinite(result):
        result = default if cap is None else cap
    if cap is not None:
        result = min(result, cap)
    return result


def build_geo_features(diagnosis: Dict[str, object], args, device: torch.device) -> torch.Tensor:
    num_inliers = safe_float(diagnosis.get("num_ransac_inliers", 0.0))
    mean_reproj = safe_float(diagnosis.get("mean_reprojection_error", 0.0), cap=args.feature_reprojection_cap)
    max_reproj = safe_float(diagnosis.get("max_reprojection_error", mean_reproj), cap=args.feature_reprojection_cap)
    tvec_norm = safe_float(diagnosis.get("tvec_norm", 0.0), cap=args.feature_tvec_norm_cap)
    fallback = 1.0 if diagnosis.get("used_fallback_epnp", False) else 0.0
    pnp_failed = 1.0 if "pnp_failed" in diagnosis.get("trigger_reasons", []) else 0.0

    values = [
        safe_float(diagnosis.get("mean_confidence", 0.0)),
        num_inliers,
        num_inliers / max(float(args.num_keypoints), 1.0),
        np.log1p(mean_reproj) / np.log1p(args.feature_reprojection_cap),
        np.log1p(max_reproj) / np.log1p(args.feature_reprojection_cap),
        fallback,
        pnp_failed,
        tvec_norm / max(args.feature_tvec_norm_cap, 1e-6),
    ]
    return torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0)


def heuristic_risk_label(diagnosis: Dict[str, object], args, device: torch.device) -> torch.Tensor:
    mean_conf = safe_float(diagnosis.get("mean_confidence", 0.0))
    num_inliers = int(diagnosis.get("num_ransac_inliers", 0))
    mean_reproj = safe_float(diagnosis.get("mean_reprojection_error", 0.0), cap=args.feature_reprojection_cap)
    fallback = bool(diagnosis.get("used_fallback_epnp", False))
    pnp_failed = "pnp_failed" in diagnosis.get("trigger_reasons", [])
    label = (
        mean_conf < args.trigger_confidence
        or num_inliers < args.trigger_min_inliers
        or mean_reproj > args.trigger_reprojection_error
        or fallback
        or pnp_failed
    )
    return torch.tensor([1.0 if label else 0.0], dtype=torch.float32, device=device)


@torch.no_grad()
def build_source_prototype(model: DinoHeatmapPoseModel, args, device: torch.device) -> torch.Tensor | None:
    source_dataset = make_dataset(args, "train")
    source_loader = DataLoader(
        source_dataset,
        batch_size=args.prototype_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    features = []
    seen = 0
    model.eval()
    for batch in tqdm(source_loader, desc="build_source_prototype", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        _, cls_token = model(images, return_features=True)
        features.append(cls_token.detach().cpu())
        seen += images.shape[0]
        if args.prototype_max_samples is not None and seen >= args.prototype_max_samples:
            break

    if not features:
        return None
    prototype = torch.cat(features, dim=0).mean(dim=0)
    return F.normalize(prototype.flatten(), dim=0).cpu()


def gate_probability(
    gate: nn.Module,
    geo_features: torch.Tensor,
    feat_features: torch.Tensor,
) -> torch.Tensor:
    return torch.sigmoid(gate(geo_features, feat_features))


def choose_gate_decision(
    step: int,
    heuristic_label: torch.Tensor,
    risk_prob: torch.Tensor,
    args,
) -> Tuple[bool, float]:
    if step <= args.gate_warmup_steps:
        weight = float(heuristic_label.item())
        return bool(weight >= 0.5), weight

    prob = float(risk_prob.detach().item())
    if args.gate_usage == "hard":
        return prob >= args.gate_threshold, 1.0 if prob >= args.gate_threshold else 0.0
    if args.gate_usage in {"soft_loss", "soft_lr"}:
        return prob >= args.min_soft_gate_weight, max(prob, 0.0)
    raise ValueError(f"Unsupported gate_usage: {args.gate_usage}")


def scale_optimizer_lr(optimizer: AdamW, scale: float):
    old_lrs = [group["lr"] for group in optimizer.param_groups]
    for group, old_lr in zip(optimizer.param_groups, old_lrs):
        group["lr"] = old_lr * scale
    return old_lrs


def restore_optimizer_lr(optimizer: AdamW, old_lrs) -> None:
    for group, old_lr in zip(optimizer.param_groups, old_lrs):
        group["lr"] = old_lr


def adapt_with_gate(
    model: DinoHeatmapPoseModel,
    optimizer: AdamW,
    scaler: GradScaler,
    memory_bank: FeatureQualityMemoryBank,
    current_image: torch.Tensor,
    geometry_target: torch.Tensor | None,
    gate_weight: float,
    args,
    device: torch.device,
) -> Dict[str, float]:
    mem_images, mem_pseudo, mem_weights, mem_indices = memory_bank.sample(args.memory_sample_size, device)

    total_loss_value = 0.0
    loss_st_value = 0.0
    loss_geo_value = 0.0
    loss_reg_value = 0.0

    model.train()
    for _ in range(args.adapt_steps):
        optimizer.zero_grad(set_to_none=True)
        old_lrs = None
        if args.gate_usage == "soft_lr":
            old_lrs = scale_optimizer_lr(optimizer, max(gate_weight, args.min_lr_gate_scale))

        with autocast(enabled=device.type == "cuda"):
            mem_pred = model(mem_images)
            loss_st = weighted_heatmap_mse(mem_pred, mem_pseudo.detach(), weights=mem_weights)

            loss_geo = mem_pred.new_tensor(0.0)
            if geometry_target is not None and args.lambda_geo > 0.0:
                current_pred = model(current_image)
                loss_geo = F.mse_loss(current_pred, geometry_target.detach())

            loss_reg = mem_pred.new_tensor(0.0)
            if args.lambda_reg > 0.0:
                loss_reg = confidence_weighted_regularization(mem_pred, mem_pseudo, tau=args.tau)

            total_loss = (
                args.lambda_self_training * loss_st
                + args.lambda_geo * loss_geo
                + args.lambda_reg * loss_reg
            )

            if args.gate_usage == "soft_loss":
                total_loss = total_loss * float(gate_weight)

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group["params"]],
                args.grad_clip_norm,
            )
        scaler.step(optimizer)
        scaler.update()

        if old_lrs is not None:
            restore_optimizer_lr(optimizer, old_lrs)

        total_loss_value = float(total_loss.item())
        loss_st_value = float(loss_st.item())
        loss_geo_value = float(loss_geo.item())
        loss_reg_value = float(loss_reg.item())

    with torch.no_grad():
        updated_mem_pred = model(mem_images)
        updated_quality = updated_mem_pred.detach().flatten(2).amax(dim=-1).mean(dim=1)
        memory_bank.update_if_better(mem_indices, updated_mem_pred, updated_quality)

    return {
        "total_loss": total_loss_value,
        "loss_self_training": loss_st_value,
        "loss_geometry": loss_geo_value,
        "loss_regularization": loss_reg_value,
    }


def run_learnable_trigger_tta(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_dataset = make_dataset(args, args.target_split)
    target_loader = DataLoader(
        target_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    camera_matrix, dist_coeffs = load_camera(Path(args.data_root))
    model = load_source_model(args.source_checkpoint, device)
    source_prototype = build_source_prototype(model, args, device)

    trainable_params = configure_trainable_parameters(model, args.update_scope)
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=device.type == "cuda")

    gate = DualBranchRiskGate(hidden_dim=args.gate_hidden_dim, dropout=args.gate_dropout).to(device)
    gate_optimizer = AdamW(gate.parameters(), lr=args.gate_lr, weight_decay=args.gate_weight_decay)

    memory_bank = FeatureQualityMemoryBank(capacity=args.memory_capacity)

    history: List[Dict[str, object]] = []
    loss_history: List[Dict[str, object]] = []
    gate_history: List[Dict[str, object]] = []

    for step, batch in enumerate(tqdm(target_loader, desc="learnable_trigger_dual_branch"), start=1):
        if args.max_samples is not None and step > args.max_samples:
            break

        image = batch["image"].to(device, non_blocking=True)
        bbox = tensor_bbox_to_tuple(batch["bbox"])
        image_name = tensor_image_name(batch["image_name"])

        model.eval()
        with torch.no_grad(), autocast(enabled=device.type == "cuda"):
            heatmap, cls_token = model(image, return_features=True)

        diagnosis = diagnose_prediction(
            heatmap=heatmap,
            bbox=bbox,
            input_size=args.input_size,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            args=args,
        )

        geo_features = build_geo_features(diagnosis, args, device)
        feat_features = memory_bank.feature_distances(cls_token.squeeze(0), source_prototype).to(device).unsqueeze(0)
        risk_label = heuristic_risk_label(diagnosis, args, device)

        gate.train()
        gate_optimizer.zero_grad(set_to_none=True)
        risk_logit = gate(geo_features.detach(), feat_features.detach())
        gate_loss = F.binary_cross_entropy_with_logits(risk_logit, risk_label)
        gate_loss.backward()
        gate_optimizer.step()
        gate_loss_value = float(gate_loss.item())

        gate.eval()
        with torch.no_grad():
            risk_prob = gate_probability(gate, geo_features, feat_features)

        should_adapt, gate_weight = choose_gate_decision(step, risk_label, risk_prob, args)

        pushed_to_memory = False
        if (
            float(diagnosis["quality"]) >= args.memory_min_quality
            and float(diagnosis["mean_confidence"]) >= args.memory_min_confidence
            and int(diagnosis["num_ransac_inliers"]) >= args.memory_min_inliers
            and float(diagnosis["mean_reprojection_error"]) <= args.memory_max_reproj_error
        ):
            memory_bank.push(
                image=image.squeeze(0),
                pseudo_heatmap=heatmap.squeeze(0),
                feature=cls_token.squeeze(0),
                quality=float(diagnosis["quality"]),
                image_name=image_name,
            )
            pushed_to_memory = True

        adapted = False
        losses = {
            "total_loss": 0.0,
            "loss_self_training": 0.0,
            "loss_geometry": 0.0,
            "loss_regularization": 0.0,
        }

        if should_adapt and len(memory_bank) >= args.min_memory_for_update:
            geometry_target = geometry_target_from_pose(
                diagnosis["rvec"],
                diagnosis["tvec"],
                bbox=bbox,
                input_size=args.input_size,
                heatmap_size=heatmap.shape[-1],
                heatmap_sigma=args.heatmap_sigma,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                device=device,
            )
            losses = adapt_with_gate(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                memory_bank=memory_bank,
                current_image=image,
                geometry_target=geometry_target,
                gate_weight=gate_weight,
                args=args,
                device=device,
            )
            adapted = True
            loss_history.append({"step": step, "image_name": image_name, "gate_weight": gate_weight, **losses})

        gate_record = {
            "step": step,
            "image_name": image_name,
            "risk_label": float(risk_label.item()),
            "risk_probability": float(risk_prob.item()),
            "gate_weight": float(gate_weight),
            "gate_loss": gate_loss_value,
            "geo_features": geo_features.detach().cpu().flatten().tolist(),
            "feat_features": feat_features.detach().cpu().flatten().tolist(),
        }
        gate_history.append(gate_record)

        history.append(
            {
                "step": step,
                "image_name": image_name,
                "trigger_mode": "dual_branch",
                "gate_usage": args.gate_usage,
                "heuristic_triggered": bool(risk_label.item() >= 0.5),
                "adapted": adapted,
                "pushed_to_memory": pushed_to_memory,
                "memory_size": len(memory_bank),
                "quality": float(diagnosis["quality"]),
                "mean_confidence": float(diagnosis["mean_confidence"]),
                "num_ransac_inliers": int(diagnosis["num_ransac_inliers"]),
                "inlier_ratio": float(diagnosis["inlier_ratio"]),
                "mean_reprojection_error": float(diagnosis["mean_reprojection_error"]),
                "max_reprojection_error": safe_float(diagnosis.get("max_reprojection_error", 0.0), cap=args.feature_reprojection_cap),
                "used_fallback_epnp": bool(diagnosis["used_fallback_epnp"]),
                **gate_record,
                **losses,
            }
        )

    summary = {
        "target_split": args.target_split,
        "source_checkpoint": args.source_checkpoint,
        "num_samples": len(history),
        "num_heuristic_triggered": int(sum(1 for item in history if item["heuristic_triggered"])),
        "num_adapted": int(sum(1 for item in history if item["adapted"])),
        "num_pushed_to_memory": int(sum(1 for item in history if item["pushed_to_memory"])),
        "trigger_mode": "dual_branch",
        "gate_usage": args.gate_usage,
        "update_scope": args.update_scope,
        **memory_bank.summary(),
    }

    with (output_dir / "learnable_trigger_history.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "summary": summary,
                "history": history,
                "loss_history": loss_history,
                "gate_history": gate_history,
            },
            handle,
            indent=2,
        )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_name": args.model_name,
        "input_size": args.input_size,
        "heatmap_size": args.heatmap_size,
        "heatmap_sigma": args.heatmap_sigma,
        "mid_channels": args.mid_channels,
        "num_deconv_layers": args.num_deconv_layers,
        "adaptation": "learnable_trigger_single_model_tta_dual_branch",
        "trigger_mode": "dual_branch",
        "gate_usage": args.gate_usage,
        "update_scope": args.update_scope,
        "summary": summary,
        "gate_state_dict": gate.state_dict(),
    }
    torch.save(checkpoint, output_dir / "tta_final.pth")
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="speedplusv2")
    parser.add_argument("--source_checkpoint", type=str, required=True)
    parser.add_argument("--target_split", type=str, default="lightbox")
    parser.add_argument("--output_dir", type=str, default="output/dinov3_heatmap_dual_branch_tta")

    parser.add_argument("--model_name", type=str, default="vit_base_patch16_dinov3.lvd1689m")
    parser.add_argument("--input_size", type=int, default=384)
    parser.add_argument("--heatmap_size", type=int, default=96)
    parser.add_argument("--heatmap_sigma", type=float, default=3.0)
    parser.add_argument("--mid_channels", type=int, default=256)
    parser.add_argument("--num_deconv_layers", type=int, default=2)
    parser.add_argument("--num_keypoints", type=int, default=11)

    parser.add_argument("--update_scope", type=str, choices=["decoder", "decoder_last_block"], default="decoder_last_block")

    parser.add_argument("--trigger_mode", type=str, default="dual_branch")
    parser.add_argument("--gate_usage", type=str, choices=["hard", "soft_loss", "soft_lr"], default="soft_loss")
    parser.add_argument("--gate_threshold", type=float, default=0.6)
    parser.add_argument("--min_soft_gate_weight", type=float, default=0.10)
    parser.add_argument("--min_lr_gate_scale", type=float, default=0.10)
    parser.add_argument("--gate_hidden_dim", type=int, default=32)
    parser.add_argument("--gate_dropout", type=float, default=0.10)
    parser.add_argument("--gate_lr", type=float, default=5e-4)
    parser.add_argument("--gate_weight_decay", type=float, default=1e-4)
    parser.add_argument("--gate_warmup_steps", type=int, default=512)

    parser.add_argument("--prototype_batch_size", type=int, default=32)
    parser.add_argument("--prototype_max_samples", type=int, default=512)
    parser.add_argument("--feature_reprojection_cap", type=float, default=50.0)
    parser.add_argument("--feature_tvec_norm_cap", type=float, default=20.0)

    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adapt_steps", type=int, default=1)

    parser.add_argument("--memory_capacity", type=int, default=64)
    parser.add_argument("--memory_sample_size", type=int, default=8)
    parser.add_argument("--min_memory_for_update", type=int, default=8)
    parser.add_argument("--memory_min_quality", type=float, default=0.10)
    parser.add_argument("--memory_min_confidence", type=float, default=0.35)
    parser.add_argument("--memory_min_inliers", type=int, default=6)
    parser.add_argument("--memory_max_reproj_error", type=float, default=10.0)

    parser.add_argument("--lambda_self_training", type=float, default=0.5)
    parser.add_argument("--lambda_geo", type=float, default=0.2)
    parser.add_argument("--lambda_reg", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    parser.add_argument("--trigger_confidence", type=float, default=0.20)
    parser.add_argument("--trigger_min_inliers", type=int, default=6)
    parser.add_argument("--trigger_reprojection_error", type=float, default=6.0)
    parser.add_argument("--quality_reprojection_cap", type=float, default=50.0)

    parser.add_argument("--nms_kernel", type=int, default=3)
    parser.add_argument("--disable_nms", action="store_true")
    parser.add_argument("--disable_subpixel", action="store_true")
    parser.add_argument("--subpixel_radius", type=int, default=2)
    parser.add_argument("--min_confidence", type=float, default=0.05)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--min_points", type=int, default=6)
    parser.add_argument("--ransac_reproj_error", type=float, default=6.0)
    parser.add_argument("--ransac_iterations", type=int, default=100)
    parser.add_argument("--ransac_confidence", type=float, default=0.999)
    parser.add_argument("--disable_iterative_refine", action="store_true")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_learnable_trigger_tta(parse_args())