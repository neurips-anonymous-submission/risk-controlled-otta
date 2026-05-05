from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data.crop_and_heatmap import SPEEDPLUS_3D_KEYPOINTS, crop_and_resize, load_camera, normalize_image
from risk_controlled_otta.eval.evaluate_dino_heatmap import (
    decode_heatmap_to_keypoints,
    map_crop_coords_to_image,
    solve_pose_robust,
)
from risk_controlled_otta.models.dino_pose_model import DinoHeatmapPoseModel


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def finite_mean(values: List[float], default: float = 0.0) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(clean)) if clean else float(default)


def finite_median(values: List[float], default: float = 0.0) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.median(clean)) if clean else float(default)


def finite_percentile(values: List[float], q: float, default: float = 0.0) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.percentile(clean, q)) if clean else float(default)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)


# -----------------------------------------------------------------------------
# Strict SPEED raw-image dataset: no GT crop, no GT pose, no GT heatmap in TTA.
# -----------------------------------------------------------------------------

def resolve_speed_split(data_root: Path, split: str) -> Tuple[Path, Path]:
    if split in {"sunlamp", "sunlamp_test"}:
        return data_root / "sunlamp" / "test.json", data_root / "sunlamp" / "images"
    if split in {"lightbox", "lightbox_test"}:
        return data_root / "lightbox" / "test.json", data_root / "lightbox" / "images"
    if split in {"shirt", "shirt_test"}:
        return data_root / "shirt" / "test.json", data_root / "shirt" / "images"
    if split == "validation":
        return data_root / "synthetic" / "validation.json", data_root / "synthetic" / "images"
    raise ValueError(f"Unsupported SPEED split: {split}")


class SpeedRawImageDataset(Dataset):
    """Raw SPEED images for strict test-time adaptation.

    The dataset only uses filenames from the annotation JSON. Ground-truth pose is
    intentionally not returned, and no crop is built from GT keypoints. This keeps
    the adaptation stage label-free.
    """

    def __init__(self, data_root: str, split: str, max_samples: Optional[int] = None) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.annotation_path, self.image_dir = resolve_speed_split(self.data_root, split)
        raw = load_json(self.annotation_path)
        self.annotations = raw if isinstance(raw, list) else raw["images"]
        if max_samples is not None and max_samples > 0:
            self.annotations = self.annotations[: int(max_samples)]

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        annotation = self.annotations[index]
        filename = annotation["filename"]
        image_path = self.image_dir / filename
        image = np.asarray(Image.open(image_path).convert("RGB"))
        return {
            "image_np": image,
            "image_name": filename,
            "image_path": str(image_path),
            "height": int(image.shape[0]),
            "width": int(image.shape[1]),
        }


def raw_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(batch) != 1:
        raise ValueError("Strict Hybrid-TTA-lite raw-image collate currently expects batch_size=1.")
    return batch[0]


# -----------------------------------------------------------------------------
# Model loading and parameter selection
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


def named_module_exists(model: torch.nn.Module, module_name: str) -> bool:
    return any(name == module_name for name, _ in model.named_modules())


def get_module_by_name(model: torch.nn.Module, module_name: str) -> torch.nn.Module:
    current: torch.nn.Module = model
    for part in module_name.split("."):
        current = getattr(current, part)
    return current


def unique_params(params: Iterable[torch.nn.Parameter]) -> List[torch.nn.Parameter]:
    seen: Set[int] = set()
    out: List[torch.nn.Parameter] = []
    for param in params:
        pid = id(param)
        if pid not in seen:
            seen.add(pid)
            out.append(param)
    return out


def select_update_parameters(model: DinoHeatmapPoseModel, update_scope: str) -> List[torch.nn.Parameter]:
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
                selected.extend(list(module.parameters()))
                print(f"[INFO] Hybrid-TTA-lite scope={update_scope}: {name}")
                break

    elif update_scope == "stem_norm":
        candidate_keywords = ["patch_embed", "norm_pre"]
        for name, module in model.named_modules():
            if any(keyword in name for keyword in candidate_keywords):
                selected.extend(list(module.parameters(recurse=False)))
        print(f"[INFO] Hybrid-TTA-lite scope={update_scope}")

    elif update_scope == "decoder":
        selected.extend(list(model.decoder.parameters()))
        print(f"[INFO] Hybrid-TTA-lite scope={update_scope}")

    elif update_scope == "stem_decoder":
        selected.extend(select_update_parameters(model, "stem"))
        selected.extend(list(model.decoder.parameters()))
        print(f"[INFO] Hybrid-TTA-lite scope={update_scope}")

    elif update_scope == "all":
        selected.extend(list(model.parameters()))
        print("[WARN] Hybrid-TTA-lite full scope=all. This is expensive and more prone to forgetting.")

    else:
        raise ValueError(f"Unsupported update_scope: {update_scope}")

    selected = unique_params(selected)
    if not selected:
        first_name, first_param = next(model.named_parameters())
        selected = [first_param]
        print(f"[WARN] No parameters found for scope={update_scope}. Falling back to first parameter: {first_name}")
    return selected


def set_active_trainable_params(model: torch.nn.Module, active_params: Sequence[torch.nn.Parameter]) -> None:
    active_ids = {id(param) for param in active_params}
    for param in model.parameters():
        param.requires_grad_(id(param) in active_ids)


def configure_hybrid_tta_parameters(
    model: DinoHeatmapPoseModel,
    efficient_scope: str,
    full_scope: str,
) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    for param in model.parameters():
        param.requires_grad_(False)

    efficient_params = select_update_parameters(model, efficient_scope)
    full_params = select_update_parameters(model, full_scope)

    print(f"[INFO] Efficient-tuning parameters: {sum(p.numel() for p in efficient_params)}")
    print(f"[INFO] Full-tuning parameters: {sum(p.numel() for p in full_params)}")
    return efficient_params, full_params


def make_ema_teacher(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    teacher = copy.deepcopy(model).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


@torch.no_grad()
def update_ema_teacher(teacher: torch.nn.Module, student: torch.nn.Module, momentum: float) -> None:
    momentum = float(momentum)
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)
    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
        if torch.is_floating_point(teacher_buffer):
            teacher_buffer.data.mul_(momentum).add_(student_buffer.data, alpha=1.0 - momentum)
        else:
            teacher_buffer.data.copy_(student_buffer.data)


# -----------------------------------------------------------------------------
# Image preprocessing without GT bbox
# -----------------------------------------------------------------------------

def resize_full_image_tensor(image_np: np.ndarray, input_size: int, device: torch.device) -> torch.Tensor:
    resized = cv2.resize(image_np, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    return normalize_image(resized).unsqueeze(0).to(device)


def full_crop_bbox(width: int, height: int) -> Tuple[float, float, float, float]:
    return 0.0, 0.0, float(width - 1), float(height - 1)


def expand_bbox_from_points(
    points: np.ndarray,
    confidences: np.ndarray,
    width: int,
    height: int,
    min_confidence: float,
    expand_ratio: float,
    min_box_size: float,
) -> Tuple[Tuple[float, float, float, float], Dict[str, Any]]:
    points = np.asarray(points, dtype=np.float64)
    confidences = np.asarray(confidences, dtype=np.float64)
    valid = np.where(confidences >= float(min_confidence))[0]
    used_fallback = False
    if len(valid) < 4:
        order = np.argsort(-confidences)
        valid = order[: max(4, min(8, len(order)))]
        used_fallback = True

    selected = points[valid]
    finite_mask = np.isfinite(selected).all(axis=1)
    selected = selected[finite_mask]
    if selected.shape[0] < 4:
        return full_crop_bbox(width, height), {
            "pseudo_bbox_fallback": True,
            "pseudo_bbox_reason": "too_few_finite_points",
            "pseudo_bbox_num_points": int(selected.shape[0]),
        }

    x1, y1 = selected.min(axis=0)
    x2, y2 = selected.max(axis=0)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = max((x2 - x1) * float(expand_ratio), float(min_box_size))
    bh = max((y2 - y1) * float(expand_ratio), float(min_box_size))

    x1 = max(0.0, cx - 0.5 * bw)
    y1 = max(0.0, cy - 0.5 * bh)
    x2 = min(float(width - 1), cx + 0.5 * bw)
    y2 = min(float(height - 1), cy + 0.5 * bh)

    if x2 <= x1 + 2.0 or y2 <= y1 + 2.0:
        return full_crop_bbox(width, height), {
            "pseudo_bbox_fallback": True,
            "pseudo_bbox_reason": "degenerate_bbox",
            "pseudo_bbox_num_points": int(selected.shape[0]),
        }

    return (float(x1), float(y1), float(x2), float(y2)), {
        "pseudo_bbox_fallback": bool(used_fallback),
        "pseudo_bbox_reason": "low_confidence_points" if used_fallback else "predicted_keypoints",
        "pseudo_bbox_num_points": int(selected.shape[0]),
    }


@torch.no_grad()
def predict_pseudo_bbox(
    model: DinoHeatmapPoseModel,
    image_np: np.ndarray,
    args,
    device: torch.device,
) -> Tuple[Tuple[float, float, float, float], Dict[str, Any]]:
    height, width = image_np.shape[:2]
    full_tensor = resize_full_image_tensor(image_np, args.input_size, device)
    model.eval()
    with autocast(enabled=device.type == "cuda"):
        heatmap_full = model(full_tensor)

    crop_coords_hm, confidences = decode_heatmap_to_keypoints(
        heatmap_full,
        apply_nms=not args.disable_nms,
        nms_kernel=args.nms_kernel,
        use_subpixel=not args.disable_subpixel,
        subpixel_radius=args.subpixel_radius,
    )
    hm_h, hm_w = heatmap_full.shape[-2:]
    coords = crop_coords_hm[0].copy()
    coords[:, 0] = coords[:, 0] / float(hm_w) * float(width)
    coords[:, 1] = coords[:, 1] / float(hm_h) * float(height)

    bbox, info = expand_bbox_from_points(
        points=coords,
        confidences=confidences[0],
        width=width,
        height=height,
        min_confidence=args.pseudo_bbox_min_confidence,
        expand_ratio=args.pseudo_bbox_expand_ratio,
        min_box_size=args.pseudo_bbox_min_size,
    )
    info.update(
        {
            "pseudo_bbox": [float(x) for x in bbox],
            "full_pass_mean_confidence": float(confidences[0].mean()),
            "full_pass_min_confidence": float(confidences[0].min()),
        }
    )
    return bbox, info


def make_predicted_crop_tensor(
    image_np: np.ndarray,
    bbox: Tuple[float, float, float, float],
    input_size: int,
    device: torch.device,
) -> torch.Tensor:
    crop_image, _ = crop_and_resize(
        image_np,
        np.zeros((SPEEDPLUS_3D_KEYPOINTS.shape[0], 2), dtype=np.float32),
        bbox,
        output_size=input_size,
    )
    return normalize_image(crop_image).unsqueeze(0).to(device)


# -----------------------------------------------------------------------------
# Hybrid-TTA-lite objective
# -----------------------------------------------------------------------------

def haar_dwt2(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    img = normalize_like_image(image.detach())
    _, lh, hl, hh = haar_dwt2(img)
    high_energy = torch.sqrt(lh.pow(2) + hl.pow(2) + hh.pow(2) + 1e-6).mean(dim=1, keepdim=True)
    high_energy = F.interpolate(high_energy, size=pred_heatmap.shape[-2:], mode="bilinear", align_corners=False)
    high_energy = high_energy / high_energy.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)

    heatmap_energy = pred_heatmap.mean(dim=1, keepdim=True)
    heatmap_energy = heatmap_energy - heatmap_energy.flatten(1).amin(dim=1).view(-1, 1, 1, 1)
    heatmap_energy = heatmap_energy / heatmap_energy.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
    return F.mse_loss(heatmap_energy, high_energy)


def random_block_mask_like_image(image: torch.Tensor, mask_ratio: float, patch_size: int) -> torch.Tensor:
    b, c, h, w = image.shape
    patch_size = max(int(patch_size), 1)
    gh = int(math.ceil(h / patch_size))
    gw = int(math.ceil(w / patch_size))
    drop = torch.rand((b, 1, gh, gw), device=image.device, dtype=image.dtype) < float(mask_ratio)
    drop = F.interpolate(drop.float(), size=(h, w), mode="nearest").to(dtype=image.dtype)
    return 1.0 - drop


def masked_heatmap_consistency_loss(
    student_model: torch.nn.Module,
    teacher_model: Optional[torch.nn.Module],
    image: torch.Tensor,
    clean_pred_heatmap: torch.Tensor,
    args,
) -> torch.Tensor:
    if args.lambda_mim <= 0:
        return clean_pred_heatmap.new_tensor(0.0)

    mask = random_block_mask_like_image(image, args.mim_mask_ratio, args.mim_patch_size)
    masked_image = image * mask

    with torch.no_grad():
        if teacher_model is not None:
            teacher_model.eval()
            teacher_heatmap = teacher_model(image).detach()
        else:
            teacher_heatmap = clean_pred_heatmap.detach()

    masked_pred = student_model(masked_image)
    return F.mse_loss(masked_pred, teacher_heatmap)


def hybrid_tta_objective(
    student_model: torch.nn.Module,
    teacher_model: Optional[torch.nn.Module],
    pred_heatmap: torch.Tensor,
    image: torch.Tensor,
    args,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss_entropy = heatmap_spatial_entropy(pred_heatmap, temperature=args.temperature)
    loss_conf = heatmap_confidence_loss(pred_heatmap)
    loss_dwt = wavelet_consistency_loss(pred_heatmap, image)
    loss_mim = masked_heatmap_consistency_loss(student_model, teacher_model, image, pred_heatmap, args)

    total = (
        args.lambda_entropy * loss_entropy
        + args.lambda_confidence * loss_conf
        + args.lambda_dwt * loss_dwt
        + args.lambda_mim * loss_mim
    )
    return total, {
        "loss_ltta_total": float(total.detach().item()),
        "loss_entropy": float(loss_entropy.detach().item()),
        "loss_confidence": float(loss_conf.detach().item()),
        "loss_dwt": float(loss_dwt.detach().item()),
        "loss_mim": float(loss_mim.detach().item()),
    }


# -----------------------------------------------------------------------------
# Label-free diagnosis and dynamic domain-shift detection
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
    image_coords = map_crop_coords_to_image(crop_coords, bbox, crop_size_w=float(input_size), crop_size_h=float(input_size))

    pose_failed = False
    try:
        rvec, tvec, pose_debug = solve_pose_robust(
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
        rvec, tvec = None, None
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

    entropy_value = float(heatmap_spatial_entropy(heatmap.detach(), temperature=args.temperature).item())

    return {
        "pose_failed": bool(pose_failed),
        "rvec": rvec,
        "tvec": tvec,
        "mean_confidence": float(mean_conf),
        "min_confidence": float(min_conf),
        "num_selected_points": int(num_selected),
        "num_ransac_inliers": int(num_inliers),
        "inlier_ratio": float(inlier_ratio),
        "mean_reprojection_error": float(reproj_error),
        "max_reprojection_error": float(pose_debug.get("max_reprojection_error", float("inf"))),
        "used_fallback_epnp": bool(fallback),
        "quality": float(quality),
        "entropy_for_ddsd": float(entropy_value),
        **pose_debug,
    }


class DynamicDomainShiftDetector:
    def __init__(self, window_size: int, warmup_steps: int, threshold: float, cooldown_steps: int) -> None:
        self.window_size = max(int(window_size), 2)
        self.warmup_steps = max(int(warmup_steps), 0)
        self.threshold = float(threshold)
        self.cooldown_steps = max(int(cooldown_steps), 0)
        self.history: Deque[Dict[str, float]] = deque(maxlen=self.window_size)
        self.cooldown_left = 0

    def _baseline(self, key: str, default: float) -> float:
        values = [item[key] for item in self.history if key in item and math.isfinite(float(item[key]))]
        return float(np.median(values)) if values else float(default)

    def decide(self, metrics: Dict[str, object], step: int, args) -> Dict[str, object]:
        quality = float(metrics.get("quality", 0.0))
        confidence = float(metrics.get("mean_confidence", 0.0))
        entropy = float(metrics.get("entropy_for_ddsd", 0.0))
        inlier_ratio = float(metrics.get("inlier_ratio", 0.0))
        reproj_error = float(metrics.get("mean_reprojection_error", args.quality_reprojection_cap))
        fallback = bool(metrics.get("used_fallback_epnp", False)) or bool(metrics.get("pose_failed", False))

        base_quality = self._baseline("quality", max(quality, 1e-6))
        base_conf = self._baseline("confidence", max(confidence, 1e-6))
        base_entropy = self._baseline("entropy", max(entropy, 1e-6))

        quality_drop = max(0.0, (base_quality - quality) / max(abs(base_quality), 1e-6))
        conf_drop = max(0.0, (base_conf - confidence) / max(abs(base_conf), 1e-6))
        entropy_rise = max(0.0, (entropy - base_entropy) / max(abs(base_entropy), 1e-6))
        reproj_penalty = min(max(reproj_error / max(args.quality_reprojection_cap, 1e-6), 0.0), 1.0)
        inlier_penalty = max(0.0, args.ddsd_min_inlier_ratio - inlier_ratio)
        fallback_penalty = 0.5 if fallback else 0.0

        score = (
            args.ddsd_quality_weight * quality_drop
            + args.ddsd_confidence_weight * conf_drop
            + args.ddsd_entropy_weight * entropy_rise
            + args.ddsd_reprojection_weight * reproj_penalty
            + args.ddsd_inlier_weight * inlier_penalty
            + fallback_penalty
        )

        has_baseline = len(self.history) >= self.warmup_steps
        force_efficient = step <= self.warmup_steps or not has_baseline
        shifted = (score >= self.threshold) and not force_efficient

        if self.cooldown_left > 0:
            shifted = True
            self.cooldown_left -= 1
        elif shifted and self.cooldown_steps > 0:
            self.cooldown_left = self.cooldown_steps

        mode = "full" if shifted else "efficient"
        return {
            "hybrid_mode": mode,
            "domain_shift_detected": bool(shifted),
            "ddsd_score": float(score),
            "ddsd_quality_drop": float(quality_drop),
            "ddsd_confidence_drop": float(conf_drop),
            "ddsd_entropy_rise": float(entropy_rise),
            "ddsd_reprojection_penalty": float(reproj_penalty),
            "ddsd_inlier_penalty": float(inlier_penalty),
            "ddsd_baseline_quality": float(base_quality),
            "ddsd_baseline_confidence": float(base_conf),
            "ddsd_baseline_entropy": float(base_entropy),
        }

    def update(self, metrics: Dict[str, object]) -> None:
        self.history.append(
            {
                "quality": float(metrics.get("quality", 0.0)),
                "confidence": float(metrics.get("mean_confidence", 0.0)),
                "entropy": float(metrics.get("entropy_for_ddsd", 0.0)),
            }
        )


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

def summarize_history(history: List[Dict[str, object]], args) -> Dict[str, object]:
    losses = [float(item.get("loss_ltta_total", 0.0)) for item in history]
    losses_mim = [float(item.get("loss_mim", 0.0)) for item in history]
    qualities = [float(item.get("quality", 0.0)) for item in history]
    confs = [float(item.get("mean_confidence", 0.0)) for item in history]
    inliers = [float(item.get("num_ransac_inliers", 0.0)) for item in history]
    reproj = [float(item.get("mean_reprojection_error", float("inf"))) for item in history]
    fallback = [1.0 if bool(item.get("used_fallback_epnp", False)) else 0.0 for item in history]
    bbox_fallback = [1.0 if bool(item.get("pseudo_bbox_fallback", False)) else 0.0 for item in history]
    full_steps = [1.0 if item.get("hybrid_mode") == "full" else 0.0 for item in history]
    efficient_steps = [1.0 if item.get("hybrid_mode") == "efficient" else 0.0 for item in history]
    shifted = [1.0 if bool(item.get("domain_shift_detected", False)) else 0.0 for item in history]
    ddsd_scores = [float(item.get("ddsd_score", 0.0)) for item in history]
    return {
        "method": "strict_hybrid_tta_lite_single_model_tta",
        "strict_tta": True,
        "strict_notes": "Adaptation uses raw target images and predicted pseudo bboxes only; no GT pose, GT keypoints, GT heatmaps, or GT-derived crop/bbox are used during adaptation.",
        "hybrid_notes": "Hybrid-TTA-lite for heatmap pose: label-free dynamic domain-shift detection switches between efficient tuning and full tuning; masked heatmap consistency provides the MIM-style auxiliary adaptation signal.",
        "target_split": args.target_split,
        "source_checkpoint": args.source_checkpoint,
        "num_samples": len(history),
        "num_adapted": int(sum(1 for item in history if bool(item.get("adapted", False)))),
        "adapt_ratio": float(sum(1 for item in history if bool(item.get("adapted", False))) / max(len(history), 1)),
        "trigger_ratio": finite_mean(shifted),
        "update_scope": args.update_scope,
        "efficient_update_scope": args.efficient_update_scope,
        "full_update_scope": args.full_update_scope,
        "adapt_steps": int(args.adapt_steps),
        "avg_loss_ltta_total": finite_mean(losses),
        "median_loss_ltta_total": finite_median(losses),
        "avg_loss_mim": finite_mean(losses_mim),
        "avg_quality": finite_mean(qualities),
        "avg_mean_confidence": finite_mean(confs),
        "avg_num_ransac_inliers": finite_mean(inliers),
        "avg_mean_reprojection_error": finite_mean(reproj),
        "median_mean_reprojection_error": finite_median(reproj),
        "p95_mean_reprojection_error": finite_percentile(reproj, 95),
        "fallback_epnp_ratio": finite_mean(fallback),
        "pseudo_bbox_fallback_ratio": finite_mean(bbox_fallback),
        "hybrid_full_ratio": finite_mean(full_steps),
        "hybrid_efficient_ratio": finite_mean(efficient_steps),
        "domain_shift_detected_ratio": finite_mean(shifted),
        "avg_ddsd_score": finite_mean(ddsd_scores),
        "pseudo_bbox_min_confidence": float(args.pseudo_bbox_min_confidence),
        "pseudo_bbox_expand_ratio": float(args.pseudo_bbox_expand_ratio),
        "pseudo_bbox_min_size": float(args.pseudo_bbox_min_size),
        "lr": float(args.lr),
        "full_lr": float(args.full_lr),
        "lambda_entropy": float(args.lambda_entropy),
        "lambda_confidence": float(args.lambda_confidence),
        "lambda_dwt": float(args.lambda_dwt),
        "lambda_mim": float(args.lambda_mim),
        "mim_mask_ratio": float(args.mim_mask_ratio),
        "mim_patch_size": int(args.mim_patch_size),
        "ddsd_window_size": int(args.ddsd_window_size),
        "ddsd_warmup_steps": int(args.ddsd_warmup_steps),
        "ddsd_threshold": float(args.ddsd_threshold),
    }


# -----------------------------------------------------------------------------
# Main strict Hybrid-TTA-lite loop
# -----------------------------------------------------------------------------

def run_strict_hybrid_tta_lite_single_model_tta(args) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = SpeedRawImageDataset(args.data_root, args.target_split, max_samples=args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=raw_collate,
    )

    camera_matrix, dist_coeffs = load_camera(Path(args.data_root))
    model = load_source_model(args.source_checkpoint, device)
    efficient_params, full_params = configure_hybrid_tta_parameters(
        model,
        efficient_scope=args.efficient_update_scope,
        full_scope=args.full_update_scope,
    )
    args.update_scope = args.efficient_update_scope

    efficient_optimizer = AdamW(efficient_params, lr=args.lr, weight_decay=args.weight_decay)
    full_optimizer = AdamW(full_params, lr=args.full_lr, weight_decay=args.full_weight_decay)
    scaler = GradScaler(enabled=device.type == "cuda")

    teacher_model = make_ema_teacher(model, device) if args.use_ema_teacher else None
    detector = DynamicDomainShiftDetector(
        window_size=args.ddsd_window_size,
        warmup_steps=args.ddsd_warmup_steps,
        threshold=args.ddsd_threshold,
        cooldown_steps=args.ddsd_cooldown_steps,
    )

    history: List[Dict[str, object]] = []

    for step, batch in enumerate(tqdm(loader, desc="strict_hybrid_tta_lite_single_model_tta"), start=1):
        image_np = batch["image_np"]
        image_name = str(batch["image_name"])
        width = int(batch["width"])
        height = int(batch["height"])

        pseudo_bbox, bbox_info = predict_pseudo_bbox(model, image_np, args, device)
        image = make_predicted_crop_tensor(image_np, pseudo_bbox, args.input_size, device)

        model.eval()
        with torch.no_grad(), autocast(enabled=device.type == "cuda"):
            heatmap_before = model(image)
        pre_diagnosis = diagnose_prediction(
            heatmap=heatmap_before,
            bbox=pseudo_bbox,
            input_size=args.input_size,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            args=args,
        )
        hybrid_decision = detector.decide(pre_diagnosis, step=step, args=args)

        if hybrid_decision["hybrid_mode"] == "full":
            active_params = full_params
            optimizer = full_optimizer
            active_grad_clip = args.full_grad_clip_norm
        else:
            active_params = efficient_params
            optimizer = efficient_optimizer
            active_grad_clip = args.grad_clip_norm

        adapted = False
        loss_record = {
            "loss_ltta_total": 0.0,
            "loss_entropy": 0.0,
            "loss_confidence": 0.0,
            "loss_dwt": 0.0,
            "loss_mim": 0.0,
        }

        set_active_trainable_params(model, active_params)
        model.train()
        for _ in range(args.adapt_steps):
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                pred_heatmap = model(image)
                loss, loss_record = hybrid_tta_objective(model, teacher_model, pred_heatmap, image, args)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if active_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(active_params, active_grad_clip)
            scaler.step(optimizer)
            scaler.update()
            adapted = True

        if teacher_model is not None:
            update_ema_teacher(teacher_model, model, momentum=args.teacher_momentum)

        model.eval()
        with torch.no_grad(), autocast(enabled=device.type == "cuda"):
            heatmap_after = model(image)

        diagnosis = diagnose_prediction(
            heatmap=heatmap_after,
            bbox=pseudo_bbox,
            input_size=args.input_size,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            args=args,
        )
        detector.update(diagnosis)

        history.append(
            {
                "step": int(step),
                "image_name": image_name,
                "image_width": width,
                "image_height": height,
                "adapted": bool(adapted),
                **bbox_info,
                **hybrid_decision,
                "pre_quality": float(pre_diagnosis["quality"]),
                "pre_mean_confidence": float(pre_diagnosis["mean_confidence"]),
                "pre_entropy_for_ddsd": float(pre_diagnosis["entropy_for_ddsd"]),
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

    save_json({"summary": summary, "history": history}, output_dir / "strict_hybrid_tta_lite_history.json")

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
            "adaptation": "strict_hybrid_tta_lite_single_model_tta",
            "update_scope": args.update_scope,
            "efficient_update_scope": args.efficient_update_scope,
            "full_update_scope": args.full_update_scope,
            "strict_tta": True,
            "summary": summary,
        },
        output_dir / "strict_hybrid_tta_lite_final.pth",
    )
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Strict Hybrid-TTA-lite for SPEED DINO heatmap pose model without GT-derived target crops."
    )
    parser.add_argument("--data_root", type=str, default="speedplusv2")
    parser.add_argument("--source_checkpoint", type=str, required=True)
    parser.add_argument(
        "--target_split",
        type=str,
        default="sunlamp",
        choices=["sunlamp", "sunlamp_test", "lightbox", "lightbox_test", "shirt", "shirt_test", "validation"],
    )
    parser.add_argument("--output_dir", type=str, default="output/dinov3_heatmap_speed_strict_hybrid_tta_lite_single_model_tta")
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_dinov3.lvd1689m")
    parser.add_argument("--input_size", type=int, default=384)
    parser.add_argument("--heatmap_size", type=int, default=96)
    parser.add_argument("--heatmap_sigma", type=float, default=3.0)
    parser.add_argument("--num_keypoints", type=int, default=11)
    parser.add_argument("--mid_channels", type=int, default=256)
    parser.add_argument("--num_deconv_layers", type=int, default=2)

    parser.add_argument("--update_scope", type=str, choices=["stem", "stem_norm", "decoder", "stem_decoder", "all"], default="stem")
    parser.add_argument("--efficient_update_scope", type=str, choices=["stem", "stem_norm", "decoder", "stem_decoder", "all"], default=None)
    parser.add_argument("--full_update_scope", type=str, choices=["stem", "stem_norm", "decoder", "stem_decoder", "all"], default="stem_decoder")

    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--full_lr", type=float, default=2e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--full_weight_decay", type=float, default=0.0)
    parser.add_argument("--adapt_steps", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lambda_entropy", type=float, default=1.0)
    parser.add_argument("--lambda_confidence", type=float, default=0.05)
    parser.add_argument("--lambda_dwt", type=float, default=0.1)
    parser.add_argument("--lambda_mim", type=float, default=0.25)
    parser.add_argument("--mim_mask_ratio", type=float, default=0.35)
    parser.add_argument("--mim_patch_size", type=int, default=32)
    parser.add_argument("--use_ema_teacher", action="store_true")
    parser.add_argument("--teacher_momentum", type=float, default=0.999)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--full_grad_clip_norm", type=float, default=0.5)

    parser.add_argument("--pseudo_bbox_min_confidence", type=float, default=0.05)
    parser.add_argument("--pseudo_bbox_expand_ratio", type=float, default=1.50)
    parser.add_argument("--pseudo_bbox_min_size", type=float, default=96.0)

    parser.add_argument("--ddsd_window_size", type=int, default=32)
    parser.add_argument("--ddsd_warmup_steps", type=int, default=8)
    parser.add_argument("--ddsd_threshold", type=float, default=0.75)
    parser.add_argument("--ddsd_cooldown_steps", type=int, default=0)
    parser.add_argument("--ddsd_quality_weight", type=float, default=1.00)
    parser.add_argument("--ddsd_confidence_weight", type=float, default=0.50)
    parser.add_argument("--ddsd_entropy_weight", type=float, default=0.25)
    parser.add_argument("--ddsd_reprojection_weight", type=float, default=0.50)
    parser.add_argument("--ddsd_inlier_weight", type=float, default=0.50)
    parser.add_argument("--ddsd_min_inlier_ratio", type=float, default=0.50)

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
    args = parser.parse_args()

    if args.efficient_update_scope is None:
        args.efficient_update_scope = args.update_scope
    else:
        args.update_scope = args.efficient_update_scope

    if args.lambda_mim <= 0:
        print("[WARN] lambda_mim <= 0 disables the MIM-style auxiliary branch.")
    if args.full_update_scope == "all":
        print("[WARN] full_update_scope=all may be unstable for online TTA. Prefer stem_decoder first.")
    return args


if __name__ == "__main__":
    run_strict_hybrid_tta_lite_single_model_tta(parse_args())


