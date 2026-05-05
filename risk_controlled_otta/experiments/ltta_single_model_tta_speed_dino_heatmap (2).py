from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.crop_and_heatmap import SPEEDPLUS_3D_KEYPOINTS, load_camera
from risk_controlled_otta.data.dino_heatmap_dataset import SpeedPlusDinoHeatmapDataset
from risk_controlled_otta.eval.evaluate_dino_heatmap import (
    decode_heatmap_to_keypoints,
    map_crop_coords_to_image,
    solve_pose_robust,
)
from risk_controlled_otta.models.dino_pose_model import DinoHeatmapPoseModel


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_image_name(value) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0])
    return str(value)


def tensor_bbox_to_tuple(bbox: torch.Tensor) -> Tuple[float, float, float, float]:
    bbox_np = bbox.detach().cpu().numpy()
    if bbox_np.ndim == 2:
        bbox_np = bbox_np[0]
    return tuple(float(item) for item in bbox_np.tolist())


def finite_mean(values: List[float], default: float = 0.0) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(clean)) if clean else float(default)


def finite_median(values: List[float], default: float = 0.0) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.median(clean)) if clean else float(default)


def finite_percentile(values: List[float], q: float, default: float = 0.0) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.percentile(clean, q)) if clean else float(default)


# -----------------------------------------------------------------------------
# Model loading and SPEED dataset construction
# -----------------------------------------------------------------------------

def load_source_model(checkpoint_path: str, device: torch.device) -> DinoHeatmapPoseModel:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = DinoHeatmapPoseModel(
        model_name=checkpoint.get("model_name", "vit_base_patch16_dinov3.lvd1689m"),
        input_size=int(checkpoint.get("input_size", 384)),
        num_keypoints=int(checkpoint.get("num_keypoints", 11)),
        mid_channels=int(checkpoint.get("mid_channels", 256)),
        num_deconv_layers=int(checkpoint.get("num_deconv_layers", 2)),
        pretrained=False,
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "student_state_dict" in checkpoint:
        state_dict = checkpoint["student_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    return model.to(device)


def make_dataset(args) -> SpeedPlusDinoHeatmapDataset:
    return SpeedPlusDinoHeatmapDataset(
        data_root=args.data_root,
        split=args.target_split,
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        heatmap_sigma=args.heatmap_sigma,
        use_source_augmentation=False,
    )


# -----------------------------------------------------------------------------
# L-TTA trainable parameter selection
# -----------------------------------------------------------------------------

def named_module_exists(model: torch.nn.Module, module_name: str) -> bool:
    return any(name == module_name for name, _ in model.named_modules())


def get_module_by_name(model: torch.nn.Module, module_name: str) -> torch.nn.Module:
    current: torch.nn.Module = model
    for part in module_name.split("."):
        current = getattr(current, part)
    return current


def configure_ltta_parameters(model: DinoHeatmapPoseModel, update_scope: str) -> List[torch.nn.Parameter]:
    """Freeze the whole pose model and only enable the lightweight stem.

    For DINO/timm-style ViTs, the stem is usually encoder.patch_embed. The fallback
    modes are intentionally conservative so L-TTA remains lightweight.
    """
    for param in model.parameters():
        param.requires_grad_(False)

    selected: List[torch.nn.Parameter] = []

    if update_scope == "stem":
        candidate_names = [
            "encoder.patch_embed",
            "encoder.backbone.patch_embed",
            "backbone.patch_embed",
            "patch_embed",
        ]
        for name in candidate_names:
            if named_module_exists(model, name):
                module = get_module_by_name(model, name)
                for param in module.parameters():
                    param.requires_grad_(True)
                    selected.append(param)
                break

    elif update_scope == "stem_norm":
        # Stem plus early normalization, if present.
        candidate_keywords = ["patch_embed", "pos_drop", "norm_pre"]
        for name, module in model.named_modules():
            if any(keyword in name for keyword in candidate_keywords):
                for param in module.parameters(recurse=False):
                    param.requires_grad_(True)
                    selected.append(param)

    elif update_scope == "decoder":
        # Diagnostic fallback: not the canonical L-TTA setting, but useful if the
        # DINO wrapper has no accessible patch_embed module.
        for param in model.decoder.parameters():
            param.requires_grad_(True)
            selected.append(param)

    else:
        raise ValueError(f"Unsupported update_scope: {update_scope}")

    # Last fallback: if the wrapper hides patch_embed under an unexpected name,
    # enable the first parameter tensor only. This preserves the lightweight spirit.
    if not selected:
        first_name, first_param = next(model.named_parameters())
        first_param.requires_grad_(True)
        selected = [first_param]
        print(f"[WARN] No explicit stem module found. Falling back to first parameter: {first_name}")

    return selected


# -----------------------------------------------------------------------------
# Lightweight DWT helpers for L-TTA loss
# -----------------------------------------------------------------------------

def haar_dwt2(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-level Haar DWT implemented with tensor slicing.

    Returns LL, LH, HL, HH with gradients. If H/W are odd, the last row/column is
    dropped to keep the decomposition simple and dependency-free.
    """
    h = x.shape[-2] - (x.shape[-2] % 2)
    w = x.shape[-1] - (x.shape[-1] % 2)
    x = x[..., :h, :w]
    x00 = x[..., 0::2, 0::2]
    x01 = x[..., 0::2, 1::2]
    x10 = x[..., 1::2, 0::2]
    x11 = x[..., 1::2, 1::2]
    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 - x01 + x10 - x11) * 0.5
    hl = (x00 + x01 - x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return ll, lh, hl, hh


def normalize_like_image(x: torch.Tensor) -> torch.Tensor:
    """Map normalized tensors to a stable range for wavelet-energy statistics."""
    flat = x.flatten(1)
    mean = flat.mean(dim=1).view(-1, 1, 1, 1)
    std = flat.std(dim=1).clamp_min(1e-6).view(-1, 1, 1, 1)
    return (x - mean) / std


def heatmap_spatial_entropy(heatmap: torch.Tensor, temperature: float) -> torch.Tensor:
    b, k, h, w = heatmap.shape
    logits = heatmap.reshape(b, k, h * w) / max(float(temperature), 1e-6)
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-6))).sum(dim=-1)
    return entropy.mean()


def heatmap_confidence_loss(heatmap: torch.Tensor) -> torch.Tensor:
    peaks = heatmap.flatten(2).amax(dim=-1)
    return -peaks.mean()


def wavelet_consistency_loss(pred_heatmap: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """Encourage prediction sharpness to agree with low-level frequency content.

    This is a practical L-TTA-style loss for heatmap pose estimation: the update is
    stem-only, and the loss depends on wavelet statistics from the current target
    image plus the model's unlabeled heatmap output.
    """
    img = normalize_like_image(image.detach())
    _, lh, hl, hh = haar_dwt2(img)
    high_energy = torch.sqrt(lh.pow(2) + hl.pow(2) + hh.pow(2) + 1e-6).mean(dim=1, keepdim=True)
    high_energy = F.interpolate(high_energy, size=pred_heatmap.shape[-2:], mode="bilinear", align_corners=False)
    high_energy = high_energy / high_energy.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)

    heatmap_energy = pred_heatmap.mean(dim=1, keepdim=True)
    heatmap_energy = heatmap_energy - heatmap_energy.flatten(1).amin(dim=1).view(-1, 1, 1, 1)
    heatmap_energy = heatmap_energy / heatmap_energy.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
    return F.mse_loss(heatmap_energy, high_energy)


def ltta_objective(pred_heatmap: torch.Tensor, image: torch.Tensor, args) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss_entropy = heatmap_spatial_entropy(pred_heatmap, temperature=args.temperature)
    loss_conf = heatmap_confidence_loss(pred_heatmap)
    loss_dwt = wavelet_consistency_loss(pred_heatmap, image)
    total = args.lambda_entropy * loss_entropy + args.lambda_confidence * loss_conf + args.lambda_dwt * loss_dwt
    return total, {
        "loss_ltta_total": float(total.detach().item()),
        "loss_entropy": float(loss_entropy.detach().item()),
        "loss_confidence": float(loss_conf.detach().item()),
        "loss_dwt": float(loss_dwt.detach().item()),
    }


# -----------------------------------------------------------------------------
# Diagnosis only. This script intentionally does not compute GT pose metrics.
# Use evaluate_dino_heatmap.py after ltta_final.pth is saved.
# -----------------------------------------------------------------------------

def heatmap_confidence_stats(heatmap: torch.Tensor) -> Tuple[float, float]:
    peaks = heatmap.detach().reshape(heatmap.shape[0], heatmap.shape[1], -1).amax(dim=-1)
    return float(peaks.mean().item()), float(peaks.min().item())


@torch.no_grad()
def diagnose_prediction(
    heatmap: torch.Tensor,
    bbox: Tuple[float, float, float, float],
    input_size: int,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    args,
) -> Dict[str, object]:
    crop_coords_hm, confidences = decode_heatmap_to_keypoints(
        heatmap,
        apply_nms=not args.disable_nms,
        nms_kernel=args.nms_kernel,
        use_subpixel=not args.disable_subpixel,
        subpixel_radius=args.subpixel_radius,
    )

    hm_h, hm_w = heatmap.shape[-2:]
    crop_coords = crop_coords_hm[0].copy()
    crop_coords[:, 0] = crop_coords[:, 0] / float(hm_w) * float(input_size)
    crop_coords[:, 1] = crop_coords[:, 1] / float(hm_h) * float(input_size)
    image_coords = map_crop_coords_to_image(
        crop_coords,
        bbox,
        crop_size_w=float(input_size),
        crop_size_h=float(input_size),
    )

    pose_failed = False
    try:
        _, _, pose_debug = solve_pose_robust(
            image_points=image_coords,
            confidences=confidences[0],
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            min_confidence=args.min_confidence,
            top_k=args.top_k,
            min_points=args.min_points,
            ransac_reproj_error=args.ransac_reproj_error,
            ransac_iterations=args.ransac_iterations,
            confidence_prob=args.ransac_confidence,
            use_iterative_refine=not args.disable_iterative_refine,
        )
    except Exception as exc:
        pose_debug = {
            "selected_indices": [],
            "ransac_inliers": [],
            "used_fallback_epnp": True,
            "mean_reprojection_error": float("inf"),
            "max_reprojection_error": float("inf"),
            "tvec_norm": float("inf"),
            "solvepnp_error": str(exc),
        }
        pose_failed = True

    mean_conf, min_conf = heatmap_confidence_stats(heatmap)
    num_selected = int(pose_debug.get("num_selected_points", len(pose_debug.get("selected_indices", []))))
    num_inliers = int(len(pose_debug.get("ransac_inliers", [])))
    inlier_ratio = float(num_inliers / max(num_selected, 1))
    reproj_error = float(pose_debug.get("mean_reprojection_error", float("inf")))
    fallback = bool(pose_debug.get("used_fallback_epnp", False))
    reproj_quality = 1.0 / (1.0 + min(reproj_error, args.quality_reprojection_cap))
    quality = float(max(mean_conf, 0.0) * max(inlier_ratio, 0.0) * reproj_quality)

    return {
        "pose_failed": bool(pose_failed),
        "mean_confidence": float(mean_conf),
        "min_confidence": float(min_conf),
        "num_selected_points": int(num_selected),
        "num_ransac_inliers": int(num_inliers),
        "inlier_ratio": float(inlier_ratio),
        "mean_reprojection_error": float(reproj_error),
        "max_reprojection_error": float(pose_debug.get("max_reprojection_error", float("inf"))),
        "used_fallback_epnp": bool(fallback),
        "quality": float(quality),
        **pose_debug,
    }


# -----------------------------------------------------------------------------
# Main L-TTA loop
# -----------------------------------------------------------------------------

def summarize_history(history: List[Dict[str, object]], args) -> Dict[str, object]:
    losses = [float(item.get("loss_ltta_total", 0.0)) for item in history]
    qualities = [float(item.get("quality", 0.0)) for item in history]
    confs = [float(item.get("mean_confidence", 0.0)) for item in history]
    inliers = [float(item.get("num_ransac_inliers", 0.0)) for item in history]
    reproj = [float(item.get("mean_reprojection_error", float("inf"))) for item in history]
    fallback = [1.0 if bool(item.get("used_fallback_epnp", False)) else 0.0 for item in history]
    return {
        "method": "ltta_single_model_tta",
        "target_split": args.target_split,
        "source_checkpoint": args.source_checkpoint,
        "num_samples": len(history),
        "num_adapted": int(sum(1 for item in history if bool(item.get("adapted", False)))),
        "adapt_ratio": float(sum(1 for item in history if bool(item.get("adapted", False))) / max(len(history), 1)),
        "update_scope": args.update_scope,
        "adapt_steps": int(args.adapt_steps),
        "avg_loss_ltta_total": finite_mean(losses),
        "median_loss_ltta_total": finite_median(losses),
        "avg_quality": finite_mean(qualities),
        "avg_mean_confidence": finite_mean(confs),
        "avg_num_ransac_inliers": finite_mean(inliers),
        "avg_mean_reprojection_error": finite_mean(reproj),
        "median_mean_reprojection_error": finite_median(reproj),
        "p95_mean_reprojection_error": finite_percentile(reproj, 95),
        "fallback_epnp_ratio": finite_mean(fallback),
        "lr": float(args.lr),
        "lambda_entropy": float(args.lambda_entropy),
        "lambda_confidence": float(args.lambda_confidence),
        "lambda_dwt": float(args.lambda_dwt),
    }


def run_ltta_single_model_tta(args) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_dataset = make_dataset(args)
    target_loader = DataLoader(
        target_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    camera_matrix, dist_coeffs = load_camera(Path(args.data_root))
    model = load_source_model(args.source_checkpoint, device)
    trainable_params = configure_ltta_parameters(model, args.update_scope)
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=device.type == "cuda")

    history: List[Dict[str, object]] = []

    for step, batch in enumerate(tqdm(target_loader, desc="ltta_single_model_tta"), start=1):
        if args.max_samples is not None and step > args.max_samples:
            break

        image = batch["image"].to(device, non_blocking=True)
        bbox = tensor_bbox_to_tuple(batch["bbox"])
        image_name = tensor_image_name(batch["image_name"])

        adapted = False
        loss_record = {
            "loss_ltta_total": 0.0,
            "loss_entropy": 0.0,
            "loss_confidence": 0.0,
            "loss_dwt": 0.0,
        }

        model.train()
        for _ in range(args.adapt_steps):
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                pred_heatmap = model(image)
                loss, loss_record = ltta_objective(pred_heatmap, image, args)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            adapted = True

        model.eval()
        with torch.no_grad(), autocast(enabled=device.type == "cuda"):
            heatmap_after = model(image)

        diagnosis = diagnose_prediction(
            heatmap=heatmap_after,
            bbox=bbox,
            input_size=args.input_size,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            args=args,
        )

        history.append(
            {
                "step": int(step),
                "image_name": image_name,
                "adapted": bool(adapted),
                **loss_record,
                "quality": float(diagnosis["quality"]),
                "mean_confidence": float(diagnosis["mean_confidence"]),
                "min_confidence": float(diagnosis["min_confidence"]),
                "num_selected_points": int(diagnosis["num_selected_points"]),
                "num_ransac_inliers": int(diagnosis["num_ransac_inliers"]),
                "inlier_ratio": float(diagnosis["inlier_ratio"]),
                "mean_reprojection_error": float(diagnosis["mean_reprojection_error"]),
                "max_reprojection_error": float(diagnosis["max_reprojection_error"]),
                "used_fallback_epnp": bool(diagnosis["used_fallback_epnp"]),
                "pose_failed": bool(diagnosis["pose_failed"]),
            }
        )

    summary = summarize_history(history, args)
    with (output_dir / "ltta_history.json").open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "history": history}, handle, indent=2)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": args.model_name,
            "input_size": args.input_size,
            "heatmap_size": args.heatmap_size,
            "heatmap_sigma": args.heatmap_sigma,
            "mid_channels": args.mid_channels,
            "num_deconv_layers": args.num_deconv_layers,
            "num_keypoints": args.num_keypoints,
            "adaptation": "ltta_single_model_tta",
            "update_scope": args.update_scope,
            "summary": summary,
        },
        output_dir / "ltta_final.pth",
    )
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="L-TTA single-model adaptation for SPEED DINO heatmap pose model.")
    parser.add_argument("--data_root", type=str, default="speedplusv2")
    parser.add_argument("--source_checkpoint", type=str, required=True)
    parser.add_argument("--target_split", type=str, default="sunlamp", choices=["sunlamp", "sunlamp_test", "lightbox", "lightbox_test", "validation"])
    parser.add_argument("--output_dir", type=str, default="output/dinov3_heatmap_speed_ltta_single_model_tta")
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_dinov3.lvd1689m")
    parser.add_argument("--input_size", type=int, default=384)
    parser.add_argument("--heatmap_size", type=int, default=96)
    parser.add_argument("--heatmap_sigma", type=float, default=3.0)
    parser.add_argument("--num_keypoints", type=int, default=11)
    parser.add_argument("--mid_channels", type=int, default=256)
    parser.add_argument("--num_deconv_layers", type=int, default=2)

    parser.add_argument("--update_scope", type=str, choices=["stem", "stem_norm", "decoder"], default="stem")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adapt_steps", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lambda_entropy", type=float, default=1.0)
    parser.add_argument("--lambda_confidence", type=float, default=0.05)
    parser.add_argument("--lambda_dwt", type=float, default=0.1)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_ltta_single_model_tta(parse_args())

