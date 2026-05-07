from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data.crop_and_heatmap import (
    compute_expanded_bbox,
    crop_and_resize,
    normalize_image,
    project_keypoints,
    visible_keypoints_mask,
)
from dinov2_heatmap_otta.models.dino_pose_model import DinoHeatmapPoseModel


SHIRT_KEYPOINTS_3D = np.array(
    [
        [-0.37, -0.385, 0.3215],
        [-0.37, 0.385, 0.3215],
        [0.37, 0.385, 0.3215],
        [0.37, -0.385, 0.3215],
        [-0.37, -0.264, 0.0],
        [-0.37, 0.304, 0.0],
        [0.37, 0.304, 0.0],
        [0.37, -0.264, 0.0],
        [-0.5427, 0.4877, 0.2535],
        [0.5427, 0.4877, 0.2591],
        [0.305, -0.579, 0.2515],
    ],
    dtype=np.float32,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_camera_from_shirt(data_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    camera_path = data_root / "camera.json"
    cam = load_json(camera_path)

    if "cameraMatrix" in cam:
        camera_matrix = np.asarray(cam["cameraMatrix"], dtype=np.float32)
    elif "camera_matrix" in cam:
        camera_matrix = np.asarray(cam["camera_matrix"], dtype=np.float32)
    elif "K" in cam:
        camera_matrix = np.asarray(cam["K"], dtype=np.float32)
    elif all(k in cam for k in ["fx", "fy", "ccx", "ccy"]):
        camera_matrix = np.array(
            [[cam["fx"], 0.0, cam["ccx"]], [0.0, cam["fy"], cam["ccy"]], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
    elif all(k in cam for k in ["fx", "fy", "cx", "cy"]):
        camera_matrix = np.array(
            [[cam["fx"], 0.0, cam["cx"]], [0.0, cam["fy"], cam["cy"]], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
    else:
        raise KeyError(f"Unsupported camera.json format: {camera_path}")

    if "distCoeffs" in cam:
        dist_coeffs = np.asarray(cam["distCoeffs"], dtype=np.float32)
    elif "dist_coeffs" in cam:
        dist_coeffs = np.asarray(cam["dist_coeffs"], dtype=np.float32)
    elif "distortion_coefficients" in cam:
        dist_coeffs = np.asarray(cam["distortion_coefficients"], dtype=np.float32)
    elif "distortion" in cam:
        dist_coeffs = np.asarray(cam["distortion"], dtype=np.float32)
    else:
        dist_coeffs = np.zeros(5, dtype=np.float32)

    return camera_matrix.astype(np.float64), dist_coeffs.reshape(-1).astype(np.float64)


def gaussian2d(shape: Tuple[int, int], sigma: float, center: Tuple[float, float]) -> np.ndarray:
    h, w = shape
    x = np.arange(0, w, dtype=np.float32)
    y = np.arange(0, h, dtype=np.float32)[:, None]
    cx, cy = center
    return np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma * sigma))


def build_heatmaps(keypoints: np.ndarray, visible: np.ndarray, heatmap_size: int, sigma: float) -> np.ndarray:
    num_keypoints = keypoints.shape[0]
    heatmaps = np.zeros((num_keypoints, heatmap_size, heatmap_size), dtype=np.float32)
    for idx in range(num_keypoints):
        if not bool(visible[idx]):
            continue
        x, y = keypoints[idx]
        if x < 0 or y < 0 or x >= heatmap_size or y >= heatmap_size:
            continue
        heatmaps[idx] = gaussian2d((heatmap_size, heatmap_size), sigma=sigma, center=(float(x), float(y)))
    return heatmaps


def parse_shirt_annotation_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "filename": entry["filename"],
        "quaternion": np.asarray(entry["q_vbs2tango_true"], dtype=np.float64),
        "translation": np.asarray(entry["r_Vo2To_vbs_true"], dtype=np.float64),
    }


def _roes_from_arg(roe: str) -> List[str]:
    return ["roe1", "roe2"] if roe == "all" else [roe]


def load_shirt_records(data_root: Path, roe: str, domain: str, val_ratio: float, seed: int, split: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for roe_name in _roes_from_arg(roe):
        roe_dir = data_root / roe_name
        ann_path = roe_dir / f"{roe_name}.json"
        image_dir = roe_dir / domain / "images"
        raw = load_json(ann_path)
        if not isinstance(raw, list):
            raise ValueError(f"Expected list annotation file: {ann_path}")
        for item in raw:
            parsed = parse_shirt_annotation_entry(item)
            parsed["roe"] = roe_name
            parsed["domain"] = domain
            parsed["image_path"] = image_dir / parsed["filename"]
            records.append(parsed)

    rng = np.random.RandomState(seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * val_ratio))
    val_idx = set(indices[:n_val].tolist())
    train_idx = set(indices[n_val:].tolist())

    if split == "train":
        return [records[i] for i in range(len(records)) if i in train_idx]
    if split == "val":
        return [records[i] for i in range(len(records)) if i in val_idx]
    if split == "all":
        return records
    raise ValueError(f"Unsupported split: {split}")


class ShirtTTADataset(Dataset):
    def __init__(
        self,
        data_root: str,
        roe: str,
        domain: str,
        split: str,
        val_ratio: float,
        seed: int,
        input_size: int,
        expand_ratio: float,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.roe = roe
        self.domain = domain
        self.split = split
        self.input_size = int(input_size)
        self.expand_ratio = float(expand_ratio)
        self.camera_matrix, self.dist_coeffs = load_camera_from_shirt(self.data_root)
        self.records = load_shirt_records(self.data_root, roe, domain, val_ratio, seed, split)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        image = np.array(Image.open(record["image_path"]).convert("RGB"))
        keypoints_2d = project_keypoints(
            SHIRT_KEYPOINTS_3D,
            record["quaternion"].astype(np.float32),
            record["translation"].astype(np.float32),
            self.camera_matrix.astype(np.float32),
            self.dist_coeffs.astype(np.float32),
        )
        visible = visible_keypoints_mask(keypoints_2d, (image.shape[1], image.shape[0]))
        bbox = compute_expanded_bbox(keypoints_2d, visible, (image.shape[1], image.shape[0]), expand_ratio=self.expand_ratio)
        crop_image, _ = crop_and_resize(image, keypoints_2d, bbox, output_size=self.input_size)
        return {
            "image": normalize_image(crop_image),
            "image_name": record["filename"],
            "roe": record["roe"],
            "domain": record["domain"],
            "bbox": torch.tensor(bbox, dtype=torch.float32),
        }


def tensor_bbox_to_tuple(bbox: torch.Tensor) -> Tuple[float, float, float, float]:
    if bbox.ndim == 2:
        bbox = bbox[0]
    return tuple(float(x) for x in bbox.detach().cpu().tolist())  # type: ignore[return-value]


def tensor_string(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        value = value[0]
    return str(value)


def safe_float(value: Any, default: float = 0.0, cap: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default
    if not np.isfinite(result):
        result = default if cap is None else cap
    if cap is not None:
        result = min(result, cap)
    return result


def load_source_model(checkpoint_path: str, device: torch.device) -> DinoHeatmapPoseModel:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = DinoHeatmapPoseModel(
        model_name=checkpoint.get("model_name", "vit_base_patch16_dinov3.lvd1689m"),
        input_size=int(checkpoint.get("input_size", 384)),
        num_keypoints=int(checkpoint.get("num_keypoints", 11)),
        mid_channels=int(checkpoint.get("mid_channels", 256)),
        num_deconv_layers=int(checkpoint.get("num_deconv_layers", 2)),
        pretrained=False,
        pretrained_path=None,
    ).to(device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "student_state_dict" in checkpoint:
        state_dict = checkpoint["student_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def configure_trainable_parameters(model: nn.Module, update_scope: str) -> List[nn.Parameter]:
    for param in model.parameters():
        param.requires_grad_(False)

    keywords_by_scope = {
        "all": [""],
        "decoder": ["decoder", "deconv", "heatmap", "head", "final", "up"],
        "decoder_last_block": ["final", "head", "output", "heatmap"],
    }
    keywords = keywords_by_scope[update_scope]
    selected: List[nn.Parameter] = []
    for name, param in model.named_parameters():
        if any(keyword == "" or keyword in name.lower() for keyword in keywords):
            param.requires_grad_(True)
            selected.append(param)

    if not selected and update_scope == "decoder_last_block":
        return configure_trainable_parameters(model, "decoder")
    if not selected:
        raise RuntimeError(f"No trainable parameters found for update_scope={update_scope}")
    return selected


def _refine_subpixel_from_heatmap_single(heatmap_2d: np.ndarray, peak_x: float, peak_y: float, patch_radius: int = 2) -> Tuple[float, float]:
    h, w = heatmap_2d.shape
    x0, y0 = int(round(peak_x)), int(round(peak_y))
    x_min, x_max = max(0, x0 - patch_radius), min(w - 1, x0 + patch_radius)
    y_min, y_max = max(0, y0 - patch_radius), min(h - 1, y0 + patch_radius)
    patch = np.maximum(heatmap_2d[y_min:y_max + 1, x_min:x_max + 1].astype(np.float64), 0.0)
    mass = patch.sum()
    if mass <= 1e-12:
        return float(peak_x), float(peak_y)
    ys, xs = np.mgrid[y_min:y_max + 1, x_min:x_max + 1]
    return float((xs * patch).sum() / mass), float((ys * patch).sum() / mass)


def decode_heatmap_to_keypoints(
    heatmap: torch.Tensor,
    apply_nms: bool = True,
    nms_kernel: int = 3,
    use_subpixel: bool = True,
    subpixel_radius: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    if apply_nms:
        pooled = F.max_pool2d(heatmap, kernel_size=nms_kernel, stride=1, padding=nms_kernel // 2)
        peak_heatmap = torch.where(heatmap == pooled, heatmap, torch.zeros_like(heatmap))
    else:
        peak_heatmap = heatmap
    batch_size, num_keypoints, hm_h, hm_w = peak_heatmap.shape
    flat = peak_heatmap.view(batch_size, num_keypoints, -1)
    confidences, indices = flat.max(dim=-1)
    ys = (indices // hm_w).float()
    xs = (indices % hm_w).float()
    coords = torch.stack([xs, ys], dim=-1)
    coords_np = coords.detach().cpu().numpy().astype(np.float64)
    conf_np = confidences.detach().cpu().numpy().astype(np.float64)
    if use_subpixel:
        heatmap_np = heatmap.detach().cpu().numpy()
        for b in range(batch_size):
            for k in range(num_keypoints):
                coords_np[b, k, 0], coords_np[b, k, 1] = _refine_subpixel_from_heatmap_single(
                    heatmap_np[b, k], coords_np[b, k, 0], coords_np[b, k, 1], patch_radius=subpixel_radius
                )
    return coords_np, conf_np


def map_crop_coords_to_image(crop_coords: np.ndarray, bbox: Tuple[float, float, float, float], crop_size: float) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    bbox_w, bbox_h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    image_coords = crop_coords.copy().astype(np.float64)
    image_coords[:, 0] = image_coords[:, 0] / crop_size * bbox_w + x1
    image_coords[:, 1] = image_coords[:, 1] / crop_size * bbox_h + y1
    return image_coords


def map_image_coords_to_heatmap(image_coords: np.ndarray, bbox: Tuple[float, float, float, float], input_size: int, heatmap_size: int) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    bbox_w, bbox_h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    crop = image_coords.copy().astype(np.float64)
    crop[:, 0] = (crop[:, 0] - x1) / bbox_w * input_size
    crop[:, 1] = (crop[:, 1] - y1) / bbox_h * input_size
    crop[:, 0] = crop[:, 0] / input_size * heatmap_size
    crop[:, 1] = crop[:, 1] / input_size * heatmap_size
    return crop


def select_correspondences_by_confidence(
    image_points: np.ndarray,
    confidences: np.ndarray,
    min_confidence: float,
    top_k: int,
    min_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    object_points = SHIRT_KEYPOINTS_3D.astype(np.float64)
    order = np.argsort(-confidences.astype(np.float64))
    valid = [idx for idx in order if confidences[idx] >= min_confidence]
    if len(valid) < min_points:
        valid = list(order[: max(min_points, min(top_k, len(order)))])
    else:
        valid = valid[: min(top_k, len(valid))]
    selected = np.asarray(valid, dtype=np.int64)
    return object_points[selected], image_points.astype(np.float64)[selected], selected


def solve_pose_robust(
    image_points: np.ndarray,
    confidences: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    min_confidence: float,
    top_k: int,
    min_points: int,
    ransac_reproj_error: float,
    ransac_iterations: int,
    confidence_prob: float,
    use_iterative_refine: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    obj_pts, img_pts, selected = select_correspondences_by_confidence(image_points, confidences, min_confidence, top_k, min_points)
    debug: Dict[str, object] = {
        "selected_indices": selected.tolist(),
        "selected_confidences": confidences[selected].tolist() if len(selected) else [],
        "num_selected_points": int(len(selected)),
        "ransac_inliers": [],
        "used_fallback_epnp": False,
        "mean_selected_confidence": float(confidences[selected].mean()) if len(selected) else 0.0,
    }
    if len(selected) < 4:
        raise RuntimeError(f"Not enough correspondences after filtering: {len(selected)}")

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=obj_pts,
        imagePoints=img_pts,
        cameraMatrix=camera_matrix.astype(np.float64),
        distCoeffs=dist_coeffs.astype(np.float64),
        useExtrinsicGuess=False,
        iterationsCount=ransac_iterations,
        reprojectionError=ransac_reproj_error,
        confidence=confidence_prob,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success:
        debug["used_fallback_epnp"] = True
        success, rvec, tvec = cv2.solvePnP(
            objectPoints=obj_pts,
            imagePoints=img_pts,
            cameraMatrix=camera_matrix.astype(np.float64),
            distCoeffs=dist_coeffs.astype(np.float64),
            flags=cv2.SOLVEPNP_EPNP,
        )
        inliers = None
    if not success:
        raise RuntimeError("Robust PnP failed.")

    if inliers is not None:
        debug["ransac_inliers"] = selected[inliers.reshape(-1)].tolist()
    if use_iterative_refine and inliers is not None and len(inliers) >= 4:
        cv2.solvePnP(
            objectPoints=obj_pts[inliers.reshape(-1)],
            imagePoints=img_pts[inliers.reshape(-1)],
            cameraMatrix=camera_matrix.astype(np.float64),
            distCoeffs=dist_coeffs.astype(np.float64),
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

    projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix.astype(np.float64), dist_coeffs.astype(np.float64))
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - img_pts, axis=1)
    debug["mean_reprojection_error"] = float(errors.mean())
    debug["max_reprojection_error"] = float(errors.max())
    debug["tvec_norm"] = float(np.linalg.norm(tvec.reshape(-1)))
    return rvec, tvec, debug


def diagnose_prediction(
    heatmap: torch.Tensor,
    bbox: Tuple[float, float, float, float],
    input_size: int,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    args,
) -> Dict[str, object]:
    coords_hm, confidences = decode_heatmap_to_keypoints(
        heatmap,
        apply_nms=not args.disable_nms,
        nms_kernel=args.nms_kernel,
        use_subpixel=not args.disable_subpixel,
        subpixel_radius=args.subpixel_radius,
    )
    crop_coords = coords_hm[0].copy()
    crop_coords[:, 0] = crop_coords[:, 0] * input_size / args.heatmap_size
    crop_coords[:, 1] = crop_coords[:, 1] * input_size / args.heatmap_size
    image_coords = map_crop_coords_to_image(crop_coords, bbox, crop_size=float(input_size))
    mean_conf = float(np.mean(confidences[0]))
    trigger_reasons: List[str] = []

    try:
        rvec, tvec, debug = solve_pose_robust(
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
        num_inliers = len(debug.get("ransac_inliers", []))
        if num_inliers == 0 and not bool(debug.get("used_fallback_epnp", False)):
            num_inliers = int(debug.get("num_selected_points", 0))
        mean_reproj = float(debug.get("mean_reprojection_error", math.inf))
        inlier_ratio = float(num_inliers / max(int(debug.get("num_selected_points", 0)), 1))
        quality = mean_conf * max(inlier_ratio, 0.05) * math.exp(-min(mean_reproj, args.quality_reprojection_cap) / args.quality_reprojection_cap)
        
        if mean_conf < args.trigger_confidence:
            trigger_reasons.append("low_confidence")
        if num_inliers < args.trigger_min_inliers:
            trigger_reasons.append("few_inliers")
        if mean_reproj > args.trigger_reprojection_error:
            trigger_reasons.append("high_reprojection_error")
        if bool(debug.get("used_fallback_epnp", False)):
            trigger_reasons.append("fallback_epnp")
            
        return {
            "rvec": rvec,
            "tvec": tvec,
            "quality": float(quality),
            "mean_confidence": mean_conf,
            "num_ransac_inliers": int(num_inliers),
            "inlier_ratio": inlier_ratio,
            "triggered": len(trigger_reasons) > 0,
            "trigger_reasons": trigger_reasons,
            **debug,
        }
    except Exception as exc:
        trigger_reasons.append("pnp_failed")
        return {
            "rvec": None,
            "tvec": None,
            "quality": 0.0,
            "mean_confidence": mean_conf,
            "num_ransac_inliers": 0,
            "inlier_ratio": 0.0,
            "mean_reprojection_error": float("inf"),
            "max_reprojection_error": float("inf"),
            "tvec_norm": float("inf"),
            "used_fallback_epnp": True,
            "num_selected_points": 0,
            "ransac_inliers": [],
            "triggered": True,
            "trigger_reasons": trigger_reasons,
            "error": str(exc),
        }


def geometry_target_from_pose(
    rvec: np.ndarray | None,
    tvec: np.ndarray | None,
    bbox: Tuple[float, float, float, float],
    input_size: int,
    heatmap_size: int,
    heatmap_sigma: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    device: torch.device,
) -> torch.Tensor | None:
    if rvec is None or tvec is None:
        return None
    projected, _ = cv2.projectPoints(
        SHIRT_KEYPOINTS_3D.astype(np.float64),
        np.asarray(rvec, dtype=np.float64),
        np.asarray(tvec, dtype=np.float64),
        camera_matrix.astype(np.float64),
        dist_coeffs.astype(np.float64),
    )
    image_coords = projected.reshape(-1, 2)
    heatmap_coords = map_image_coords_to_heatmap(image_coords, bbox, input_size=input_size, heatmap_size=heatmap_size)
    visible = (
        (heatmap_coords[:, 0] >= 0)
        & (heatmap_coords[:, 1] >= 0)
        & (heatmap_coords[:, 0] < heatmap_size)
        & (heatmap_coords[:, 1] < heatmap_size)
    )
    target = build_heatmaps(heatmap_coords.astype(np.float32), visible.astype(np.float32), heatmap_size, heatmap_sigma)
    return torch.tensor(target, dtype=torch.float32, device=device).unsqueeze(0)


def weighted_heatmap_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    loss = (pred - target) ** 2
    if weights is None:
        return loss.mean()
    while weights.ndim < loss.ndim:
        weights = weights.view(*weights.shape, *([1] * (loss.ndim - weights.ndim)))
    return (loss * weights).mean()


def confidence_weighted_regularization(pred: torch.Tensor, pseudo: torch.Tensor, tau: float = 0.7) -> torch.Tensor:
    confidence = pseudo.detach().flatten(2).amax(dim=-1)
    weight = (confidence >= tau).float().view(confidence.shape[0], confidence.shape[1], 1, 1)
    if float(weight.sum().item()) <= 0.0:
        return pred.new_tensor(0.0)
    return ((pred - pseudo.detach()) ** 2 * weight).sum() / weight.sum().clamp_min(1.0)


@dataclass
class QualityMemoryEntry:
    image: torch.Tensor
    pseudo_heatmap: torch.Tensor
    quality: float
    image_name: str


class QualityMemoryBank:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.entries: List[QualityMemoryEntry] = []

    def __len__(self) -> int:
        return len(self.entries)

    def push(self, image: torch.Tensor, pseudo_heatmap: torch.Tensor, quality: float, image_name: str) -> None:
        entry = QualityMemoryEntry(
            image=image.detach().cpu().clone(),
            pseudo_heatmap=pseudo_heatmap.detach().cpu().clone(),
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
            raise RuntimeError("Quality memory is empty.")
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


@torch.no_grad()
def maybe_push_current_sample(memory_bank: QualityMemoryBank, image: torch.Tensor, heatmap: torch.Tensor, image_name: str, quality: float, args) -> bool:
    if quality < args.memory_min_quality:
        return False
    memory_bank.push(image.squeeze(0), heatmap.squeeze(0), quality=quality, image_name=image_name)
    return True


def adapt_single_trigger(
    model: DinoHeatmapPoseModel,
    optimizer: AdamW,
    scaler: GradScaler,
    memory_bank: QualityMemoryBank,
    current_image: torch.Tensor,
    current_pseudo: torch.Tensor,
    geometry_target: torch.Tensor | None,
    args,
    device: torch.device,
) -> Dict[str, float]:
    if len(memory_bank) > 0:
        mem_images, mem_pseudo, mem_weights, mem_indices = memory_bank.sample(args.memory_sample_size, device)
    else:
        mem_images = current_image
        mem_pseudo = current_pseudo.detach()
        mem_weights = torch.ones(mem_images.shape[0], dtype=torch.float32, device=device)
        mem_indices = np.asarray([], dtype=np.int64)

    total_loss_value = loss_st_value = loss_geo_value = loss_reg_value = 0.0
    executed_step = False
    model.train()
    for _ in range(args.adapt_steps):
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=device.type == "cuda"):
            mem_pred = model(mem_images)
            loss_st = weighted_heatmap_mse(mem_pred, mem_pseudo.detach(), weights=mem_weights)
            loss_geo = mem_pred.new_tensor(0.0)
            if geometry_target is not None and args.lambda_geo > 0.0:
                current_pred = model(current_image)
                loss_geo = F.mse_loss(current_pred, geometry_target.detach())
            loss_reg = mem_pred.new_tensor(0.0)
            if args.lambda_reg > 0.0:
                loss_reg = confidence_weighted_regularization(mem_pred, mem_pseudo, tau=args.tau)
            total_loss = args.lambda_self_training * loss_st + args.lambda_geo * loss_geo + args.lambda_reg * loss_reg
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        if args.grad_clip_norm > 0:
            params = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        executed_step = True
        total_loss_value = float(total_loss.item())
        loss_st_value = float(loss_st.item())
        loss_geo_value = float(loss_geo.item())
        loss_reg_value = float(loss_reg.item())

    if len(mem_indices) > 0:
        with torch.no_grad():
            updated_mem_pred = model(mem_images)
            updated_quality = updated_mem_pred.detach().flatten(2).amax(dim=-1).mean(dim=1)
            memory_bank.update_if_better(mem_indices, updated_mem_pred, updated_quality)
    return {"executed_step": executed_step, "total_loss": total_loss_value, "loss_self_training": loss_st_value, "loss_geometry": loss_geo_value, "loss_regularization": loss_reg_value}


def run_triggered_single_model_tta(args) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_dataset = ShirtTTADataset(
        data_root=args.data_root,
        roe=args.roe,
        domain=args.domain,
        split=args.target_split,
        val_ratio=args.val_ratio,
        seed=args.seed,
        input_size=args.input_size,
        expand_ratio=args.expand_ratio,
    )
    target_loader = DataLoader(target_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    camera_matrix, dist_coeffs = load_camera_from_shirt(Path(args.data_root))
    model = load_source_model(args.source_checkpoint, device)
    optimizer = AdamW(configure_trainable_parameters(model, args.update_scope), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    memory_bank = QualityMemoryBank(capacity=args.memory_capacity)

    history: List[Dict[str, object]] = []
    loss_history: List[Dict[str, object]] = []

    for step, batch in enumerate(tqdm(target_loader, desc="shirt_triggered_single_model_tta"), start=1):
        if args.max_samples is not None and step > args.max_samples:
            break
        image = batch["image"].to(device, non_blocking=True)
        bbox = tensor_bbox_to_tuple(batch["bbox"])
        image_name = tensor_string(batch["image_name"])
        roe = tensor_string(batch["roe"])
        domain = tensor_string(batch["domain"])
        
        model.eval()
        with torch.no_grad(), autocast(device_type="cuda", enabled=device.type == "cuda"):
            heatmap = model(image)
            
        diagnosis = diagnose_prediction(heatmap=heatmap, bbox=bbox, input_size=args.input_size, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, args=args)
        pushed_to_memory = maybe_push_current_sample(memory_bank, image, heatmap, image_name, float(diagnosis["quality"]), args)
        
        adapted = False
        losses = {"total_loss": 0.0, "loss_self_training": 0.0, "loss_geometry": 0.0, "loss_regularization": 0.0}
        
        if bool(diagnosis["triggered"]) and len(memory_bank) >= args.min_memory_for_update:
            geometry_target = geometry_target_from_pose(
                diagnosis.get("rvec"), diagnosis.get("tvec"), bbox=bbox, input_size=args.input_size, heatmap_size=args.heatmap_size,
                heatmap_sigma=args.heatmap_sigma, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, device=device,
            )
            losses = adapt_single_trigger(model, optimizer, scaler, memory_bank, image, heatmap.detach(), geometry_target, args, device)
            adapted = bool(losses.pop("executed_step", False))
            loss_history.append({"step": step, "image_name": image_name, **losses})
            
        history.append({
            "step": step,
            "image_name": image_name,
            "roe": roe,
            "domain": domain,
            "triggered": bool(diagnosis["triggered"]),
            "trigger_reasons": diagnosis.get("trigger_reasons", []),
            "optimizer_step_executed": adapted,
            "adapted": adapted,
            "pushed_to_memory": pushed_to_memory,
            "memory_size": len(memory_bank),
            "quality": float(diagnosis["quality"]),
            "mean_confidence": float(diagnosis["mean_confidence"]),
            "min_confidence": float(diagnosis.get("min_confidence", 0.0)),
            "num_selected_points": int(diagnosis.get("num_selected_points", 0)),
            "num_ransac_inliers": int(diagnosis["num_ransac_inliers"]),
            "inlier_ratio": float(diagnosis["inlier_ratio"]),
            "mean_reprojection_error": safe_float(diagnosis.get("mean_reprojection_error", 0.0), cap=args.feature_reprojection_cap),
            "max_reprojection_error": safe_float(diagnosis.get("max_reprojection_error", 0.0), cap=args.feature_reprojection_cap),
            "tvec_norm": safe_float(diagnosis.get("tvec_norm", 0.0), cap=args.feature_tvec_norm_cap),
            "used_fallback_epnp": bool(diagnosis["used_fallback_epnp"]),
            **losses,
        })

    summary = {
        "dataset": "SHIRT",
        "target_split": args.target_split,
        "domain": args.domain,
        "roe": args.roe,
        "source_checkpoint": args.source_checkpoint,
        "num_samples": len(history),
        "num_triggered": int(sum(1 for item in history if item["triggered"])),
        "num_adapted": int(sum(1 for item in history if item["adapted"])),
        "trigger_ratio": float(sum(1 for item in history if item["triggered"]) / max(len(history), 1)),
        "adapt_ratio": float(sum(1 for item in history if item["adapted"]) / max(len(history), 1)),
        "num_pushed_to_memory": int(sum(1 for item in history if item["pushed_to_memory"])),
        "update_scope": args.update_scope,
        **memory_bank.summary(),
    }
    
    with (output_dir / "shirt_trigger_history.json").open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "history": history, "loss_history": loss_history}, handle, indent=2)
        
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_name": args.model_name,
        "input_size": args.input_size,
        "heatmap_size": args.heatmap_size,
        "heatmap_sigma": args.heatmap_sigma,
        "mid_channels": args.mid_channels,
        "num_deconv_layers": args.num_deconv_layers,
        "num_keypoints": args.num_keypoints,
        "dataset": "SHIRT",
        "adaptation": "triggered_single_model_tta_shirt",
        "update_scope": args.update_scope,
        "summary": summary,
    }
    torch.save(checkpoint, output_dir / "shirt_tta_final.pth")
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="SHIRT_Dataset")
    parser.add_argument("--source_checkpoint", type=str, required=True)
    parser.add_argument("--roe", type=str, choices=["roe1", "roe2", "all"], default="all")
    parser.add_argument("--domain", type=str, default="synthetic")
    parser.add_argument("--target_split", type=str, choices=["train", "val", "all"], default="val")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="output/dinov3_heatmap_shirt_triggered_single_model_tta")
    
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_dinov3.lvd1689m")
    parser.add_argument("--input_size", type=int, default=384)
    parser.add_argument("--heatmap_size", type=int, default=96)
    parser.add_argument("--heatmap_sigma", type=float, default=3.0)
    parser.add_argument("--num_keypoints", type=int, default=11)
    parser.add_argument("--mid_channels", type=int, default=256)
    parser.add_argument("--num_deconv_layers", type=int, default=2)
    parser.add_argument("--expand_ratio", type=float, default=1.25)
    parser.add_argument("--update_scope", type=str, choices=["decoder", "decoder_last_block", "all"], default="decoder")
    
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adapt_steps", type=int, default=1)
    parser.add_argument("--memory_capacity", type=int, default=32)
    parser.add_argument("--memory_sample_size", type=int, default=8)
    parser.add_argument("--min_memory_for_update", type=int, default=4)
    parser.add_argument("--memory_min_quality", type=float, default=0.01)
    
    parser.add_argument("--lambda_self_training", type=float, default=1.0)
    parser.add_argument("--lambda_geo", type=float, default=0.1)
    parser.add_argument("--lambda_reg", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    
    parser.add_argument("--trigger_confidence", type=float, default=0.15)
    parser.add_argument("--trigger_min_inliers", type=int, default=5)
    parser.add_argument("--trigger_reprojection_error", type=float, default=8.0)
    parser.add_argument("--quality_reprojection_cap", type=float, default=50.0)
    parser.add_argument("--feature_reprojection_cap", type=float, default=50.0)
    parser.add_argument("--feature_tvec_norm_cap", type=float, default=20.0)
    
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
    run_triggered_single_model_tta(parse_args())