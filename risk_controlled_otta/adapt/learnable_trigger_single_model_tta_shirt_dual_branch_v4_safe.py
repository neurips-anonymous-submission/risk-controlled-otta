from __future__ import annotations

import argparse
import copy
import json
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
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dinov2_heatmap_otta.models.dino_pose_model import DinoHeatmapPoseModel
from data.crop_and_heatmap import (
    compute_expanded_bbox,
    crop_and_resize,
    normalize_image,
    project_keypoints,
    visible_keypoints_mask,
)


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


def gaussian2d(shape: Tuple[int, int], sigma: float, center: Tuple[float, float]) -> np.ndarray:
    h, w = shape
    x = np.arange(0, w, dtype=np.float32)
    y = np.arange(0, h, dtype=np.float32)[:, None]
    cx, cy = center
    return np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma * sigma))


def build_heatmaps(
    keypoints: np.ndarray,
    visible: np.ndarray,
    heatmap_size: int,
    sigma: float,
) -> np.ndarray:
    num_keypoints = keypoints.shape[0]
    heatmaps = np.zeros((num_keypoints, heatmap_size, heatmap_size), dtype=np.float32)

    for idx in range(num_keypoints):
        if not bool(visible[idx]):
            continue
        x, y = keypoints[idx]
        if x < 0 or y < 0 or x >= heatmap_size or y >= heatmap_size:
            continue
        heatmaps[idx] = gaussian2d((heatmap_size, heatmap_size), sigma=sigma, center=(x, y))

    return heatmaps


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
            [
                [cam["fx"], 0.0, cam["ccx"]],
                [0.0, cam["fy"], cam["ccy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    elif all(k in cam for k in ["fx", "fy", "cx", "cy"]):
        camera_matrix = np.array(
            [
                [cam["fx"], 0.0, cam["cx"]],
                [0.0, cam["fy"], cam["cy"]],
                [0.0, 0.0, 1.0],
            ],
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

    return camera_matrix, dist_coeffs.reshape(-1)


def parse_shirt_annotation_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "filename": entry["filename"],
        "quaternion": np.asarray(entry["q_vbs2tango_true"], dtype=np.float32),
        "translation": np.asarray(entry["r_Vo2To_vbs_true"], dtype=np.float32),
    }


def load_shirt_records(
    data_root: Path,
    roes: List[str],
    domain: str,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for roe in roes:
        roe_dir = data_root / roe
        ann_path = roe_dir / f"{roe}.json"
        image_dir = roe_dir / domain / "images"

        raw = load_json(ann_path)
        if not isinstance(raw, list):
            raise ValueError(f"Expected list annotation file: {ann_path}")

        for item in raw:
            parsed = parse_shirt_annotation_entry(item)
            parsed["roe"] = roe
            parsed["image_path"] = image_dir / parsed["filename"]
            parsed["domain"] = domain
            records.append(parsed)

    return records


class ShirtDinoHeatmapDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        roes: List[str],
        domain: str = "synthetic",
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 42,
        input_size: int = 384,
        heatmap_size: int = 96,
        heatmap_sigma: float = 3.0,
        expand_ratio: float = 1.25,
        use_source_augmentation: bool = False,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.roes = roes
        self.domain = domain
        self.split = split
        self.input_size = int(input_size)
        self.heatmap_size = int(heatmap_size)
        self.heatmap_sigma = float(heatmap_sigma)
        self.expand_ratio = float(expand_ratio)
        self.use_source_augmentation = bool(use_source_augmentation)

        self.camera_matrix, self.dist_coeffs = load_camera_from_shirt(self.data_root)
        self.object_keypoints_3d = SHIRT_KEYPOINTS_3D.astype(np.float32)

        all_records = load_shirt_records(self.data_root, roes, domain=domain)

        rng = np.random.RandomState(seed)
        indices = np.arange(len(all_records))
        rng.shuffle(indices)

        n_val = max(1, int(len(indices) * val_ratio))
        val_idx = set(indices[:n_val].tolist())
        train_idx = set(indices[n_val:].tolist())

        if split == "train":
            self.records = [all_records[i] for i in range(len(all_records)) if i in train_idx]
        elif split == "val":
            self.records = [all_records[i] for i in range(len(all_records)) if i in val_idx]
        elif split == "all":
            self.records = all_records
        else:
            raise ValueError(f"Unsupported split: {split}")

    def __len__(self) -> int:
        return len(self.records)

    def _apply_source_aug(self, image: np.ndarray) -> np.ndarray:
        if not self.use_source_augmentation:
            return image

        out = image.copy()
        if random.random() < 0.5:
            scale = 0.9 + 0.2 * random.random()
            out = np.clip(out.astype(np.float32) * scale, 0, 255).astype(np.uint8)

        if random.random() < 0.3:
            noise = np.random.normal(0, 4.0, size=out.shape).astype(np.float32)
            out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return out

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]

        image = np.array(Image.open(record["image_path"]).convert("RGB"))
        quaternion = record["quaternion"]
        translation = record["translation"]

        keypoints_2d = project_keypoints(
            self.object_keypoints_3d,
            quaternion,
            translation,
            self.camera_matrix,
            self.dist_coeffs,
        )

        visible = visible_keypoints_mask(keypoints_2d, (image.shape[1], image.shape[0]))
        bbox = compute_expanded_bbox(
            keypoints_2d,
            visible,
            (image.shape[1], image.shape[0]),
            expand_ratio=self.expand_ratio,
        )

        crop_image, crop_keypoints = crop_and_resize(
            image,
            keypoints_2d,
            bbox,
            output_size=self.input_size,
        )
        crop_image = self._apply_source_aug(crop_image)

        heatmap_keypoints = crop_keypoints.copy().astype(np.float32)
        heatmap_keypoints[:, 0] = heatmap_keypoints[:, 0] / self.input_size * self.heatmap_size
        heatmap_keypoints[:, 1] = heatmap_keypoints[:, 1] / self.input_size * self.heatmap_size

        heatmaps = build_heatmaps(
            heatmap_keypoints,
            visible.astype(np.float32),
            heatmap_size=self.heatmap_size,
            sigma=self.heatmap_sigma,
        )

        return {
            "image": normalize_image(crop_image),
            "heatmap": torch.tensor(heatmaps, dtype=torch.float32),
            "image_name": record["filename"],
            "bbox": torch.tensor(bbox, dtype=torch.float32),
            "quaternion": torch.tensor(quaternion, dtype=torch.float32),
            "translation": torch.tensor(translation, dtype=torch.float32),
        }


def make_dataset(args, split: str) -> ShirtDinoHeatmapDataset:
    roes = ["roe1", "roe2"] if args.roe == "all" else [args.roe]
    return ShirtDinoHeatmapDataset(
        data_root=args.data_root,
        roes=roes,
        domain=args.domain,
        split=split,
        val_ratio=args.val_ratio,
        seed=args.seed,
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        heatmap_sigma=args.heatmap_sigma,
        expand_ratio=args.expand_ratio,
        use_source_augmentation=False,
    )


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

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def configure_trainable_parameters(model: DinoHeatmapPoseModel, update_scope: str) -> List[nn.Parameter]:
    for _, param in model.named_parameters():
        param.requires_grad = False

    trainable_params: List[nn.Parameter] = []
    for name, param in model.named_parameters():
        train = False
        if update_scope == "decoder":
            train = "decoder" in name
        elif update_scope == "decoder_last_block":
            train = ("decoder" in name) or ("encoder.blocks.11" in name) or ("encoder.norm" in name)
        else:
            raise ValueError(f"Unsupported update_scope: {update_scope}")

        if train:
            param.requires_grad = True
            trainable_params.append(param)

    return trainable_params


def clone_teacher_model(model: DinoHeatmapPoseModel) -> DinoHeatmapPoseModel:
    teacher = copy.deepcopy(model)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


@torch.no_grad()
def update_ema(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    for student_param, teacher_param in zip(student.parameters(), teacher.parameters()):
        teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)
    for student_buffer, teacher_buffer in zip(student.buffers(), teacher.buffers()):
        teacher_buffer.copy_(student_buffer)


def tensor_bbox_to_tuple(bbox_tensor: torch.Tensor) -> Tuple[float, float, float, float]:
    if bbox_tensor.ndim == 2:
        bbox_tensor = bbox_tensor[0]
    arr = bbox_tensor.detach().cpu().numpy().astype(np.float64).tolist()
    return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])


def tensor_image_name(image_name_value) -> str:
    if isinstance(image_name_value, str):
        return image_name_value
    if isinstance(image_name_value, (list, tuple)):
        return str(image_name_value[0])
    return str(image_name_value)


def weighted_heatmap_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    per_sample = F.mse_loss(pred, target, reduction="none").flatten(1).mean(dim=1)
    if weights is None:
        return per_sample.mean()
    weights = weights / weights.mean().clamp_min(1e-6)
    return (per_sample * weights).mean()


def confidence_weighted_regularization(pred: torch.Tensor, ref: torch.Tensor, tau: float = 0.7) -> torch.Tensor:
    conf = ref.flatten(2).amax(dim=-1).mean(dim=1)
    conf = torch.clamp(conf / max(tau, 1e-6), 0.0, 1.0)
    per_sample = (pred - ref.detach()).abs().flatten(1).mean(dim=1)
    return (per_sample * conf).mean()


def _refine_subpixel_from_heatmap_single(
    heatmap_2d: np.ndarray,
    peak_x: float,
    peak_y: float,
    patch_radius: int = 2,
    eps: float = 1e-12,
) -> Tuple[float, float]:
    h, w = heatmap_2d.shape
    x0 = int(round(peak_x))
    y0 = int(round(peak_y))

    x_min = max(0, x0 - patch_radius)
    x_max = min(w - 1, x0 + patch_radius)
    y_min = max(0, y0 - patch_radius)
    y_max = min(h - 1, y0 + patch_radius)

    patch = heatmap_2d[y_min:y_max + 1, x_min:x_max + 1].astype(np.float64)
    patch = np.maximum(patch, 0.0)
    mass = patch.sum()
    if mass <= eps:
        return float(peak_x), float(peak_y)

    ys, xs = np.mgrid[y_min:y_max + 1, x_min:x_max + 1]
    refined_x = float((xs * patch).sum() / mass)
    refined_y = float((ys * patch).sum() / mass)
    return refined_x, refined_y


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

    batch_size, num_keypoints, _, hm_w = peak_heatmap.shape
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
                rx, ry = _refine_subpixel_from_heatmap_single(
                    heatmap_np[b, k],
                    peak_x=coords_np[b, k, 0],
                    peak_y=coords_np[b, k, 1],
                    patch_radius=subpixel_radius,
                )
                coords_np[b, k, 0] = rx
                coords_np[b, k, 1] = ry

    return coords_np, conf_np


def map_crop_coords_to_image(
    crop_coords: np.ndarray,
    bbox: Tuple[float, float, float, float],
    crop_size_w: float,
    crop_size_h: float,
) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    bbox_w = max(x2 - x1, 1.0)
    bbox_h = max(y2 - y1, 1.0)
    image_coords = crop_coords.copy().astype(np.float64)
    image_coords[:, 0] = image_coords[:, 0] / crop_size_w * bbox_w + x1
    image_coords[:, 1] = image_coords[:, 1] / crop_size_h * bbox_h + y1
    return image_coords


def select_correspondences_by_confidence(
    image_points: np.ndarray,
    confidences: np.ndarray,
    min_confidence: float = 0.05,
    top_k: int = 8,
    min_points: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    object_points = SHIRT_KEYPOINTS_3D.astype(np.float64)
    image_points = image_points.astype(np.float64)
    confidences = confidences.astype(np.float64)

    order = np.argsort(-confidences)
    valid = [idx for idx in order if confidences[idx] >= min_confidence]

    if len(valid) < min_points:
        valid = list(order[:max(min_points, min(top_k, len(order)))])
    else:
        valid = valid[:min(top_k, len(valid))]

    selected_indices = np.array(valid, dtype=np.int64)
    return object_points[selected_indices], image_points[selected_indices], selected_indices


def solve_pose_robust(
    image_points: np.ndarray,
    confidences: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    min_confidence: float = 0.05,
    top_k: int = 8,
    min_points: int = 6,
    ransac_reproj_error: float = 6.0,
    ransac_iterations: int = 100,
    confidence_prob: float = 0.999,
    use_iterative_refine: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    obj_pts_sel, img_pts_sel, selected_indices = select_correspondences_by_confidence(
        image_points=image_points,
        confidences=confidences,
        min_confidence=min_confidence,
        top_k=top_k,
        min_points=min_points,
    )

    debug = {
        "selected_indices": selected_indices.tolist(),
        "selected_confidences": confidences[selected_indices].tolist(),
        "num_selected_points": int(len(selected_indices)),
        "num_ransac_inliers": 0,
        "inlier_ratio": 0.0,
        "ransac_inliers": [],
        "used_fallback_epnp": False,
        "mean_selected_confidence": float(confidences[selected_indices].mean()) if len(selected_indices) else 0.0,
    }

    if len(selected_indices) < 4:
        raise RuntimeError(f"Not enough correspondences after filtering: {len(selected_indices)}")

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=obj_pts_sel,
        imagePoints=img_pts_sel,
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
            objectPoints=obj_pts_sel,
            imagePoints=img_pts_sel,
            cameraMatrix=camera_matrix.astype(np.float64),
            distCoeffs=dist_coeffs.astype(np.float64),
            flags=cv2.SOLVEPNP_EPNP,
        )
        inliers = None

    if not success:
        raise RuntimeError("Robust PnP failed.")

    if inliers is not None:
        inlier_indices = selected_indices[inliers.reshape(-1)]
        debug["ransac_inliers"] = inlier_indices.tolist()
        debug["num_ransac_inliers"] = int(len(inlier_indices))
        debug["inlier_ratio"] = float(len(inlier_indices) / max(len(selected_indices), 1))

    if use_iterative_refine and inliers is not None and len(inliers) >= 4:
        inlier_obj = obj_pts_sel[inliers.reshape(-1)]
        inlier_img = img_pts_sel[inliers.reshape(-1)]
        ok, rvec_refine, tvec_refine = cv2.solvePnP(
            objectPoints=inlier_obj,
            imagePoints=inlier_img,
            cameraMatrix=camera_matrix.astype(np.float64),
            distCoeffs=dist_coeffs.astype(np.float64),
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if ok:
            rvec, tvec = rvec_refine, tvec_refine

    projected, _ = cv2.projectPoints(
        objectPoints=obj_pts_sel,
        rvec=rvec,
        tvec=tvec,
        cameraMatrix=camera_matrix.astype(np.float64),
        distCoeffs=dist_coeffs.astype(np.float64),
    )
    projected = projected.reshape(-1, 2)
    reprojection_errors = np.linalg.norm(projected - img_pts_sel, axis=1)
    debug["mean_reprojection_error"] = float(reprojection_errors.mean())
    debug["max_reprojection_error"] = float(reprojection_errors.max())
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
    crop_coords_hm, confidences = decode_heatmap_to_keypoints(
        heatmap,
        apply_nms=not args.disable_nms,
        nms_kernel=args.nms_kernel,
        use_subpixel=not args.disable_subpixel,
        subpixel_radius=args.subpixel_radius,
    )

    crop_coords = crop_coords_hm[0].copy()
    crop_coords[:, 0] = crop_coords[:, 0] * input_size / args.heatmap_size
    crop_coords[:, 1] = crop_coords[:, 1] * input_size / args.heatmap_size
    image_coords = map_crop_coords_to_image(
        crop_coords,
        bbox,
        crop_size_w=float(input_size),
        crop_size_h=float(input_size),
    )

    mean_confidence = float(confidences[0].mean())
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
    except Exception as exc:
        return {
            "quality": 0.0,
            "mean_confidence": mean_confidence,
            "num_ransac_inliers": 0,
            "inlier_ratio": 0.0,
            "mean_reprojection_error": float("inf"),
            "max_reprojection_error": float("inf"),
            "tvec_norm": 0.0,
            "used_fallback_epnp": True,
            "trigger_reasons": ["pnp_failed"],
            "rvec": None,
            "tvec": None,
            "pnp_error": str(exc),
        }

    num_inliers = int(debug.get("num_ransac_inliers", 0))
    inlier_ratio = float(debug.get("inlier_ratio", 0.0))
    mean_reproj = float(debug.get("mean_reprojection_error", 1e9))
    tvec_norm = float(debug.get("tvec_norm", 0.0))
    used_fallback = bool(debug.get("used_fallback_epnp", False))

    if mean_confidence < args.trigger_confidence:
        trigger_reasons.append("low_confidence")
    if num_inliers < args.trigger_min_inliers:
        trigger_reasons.append("low_inliers")
    if mean_reproj > args.trigger_reprojection_error:
        trigger_reasons.append("high_reprojection_error")
    if tvec_norm < args.trigger_min_tvec_norm or tvec_norm > args.trigger_max_tvec_norm:
        trigger_reasons.append("abnormal_tvec_norm")
    if used_fallback:
        trigger_reasons.append("fallback_epnp")

    conf_term = np.clip(mean_confidence, 0.0, 1.0)
    inlier_term = np.clip(inlier_ratio, 0.0, 1.0)
    reproj_term = 1.0 - np.clip(mean_reproj / max(args.quality_reprojection_cap, 1e-6), 0.0, 1.0)
    quality = float(max(0.0, 0.45 * conf_term + 0.35 * inlier_term + 0.20 * reproj_term))

    return {
        "quality": quality,
        "mean_confidence": mean_confidence,
        "num_ransac_inliers": num_inliers,
        "inlier_ratio": inlier_ratio,
        "mean_reprojection_error": mean_reproj,
        "max_reprojection_error": float(debug.get("max_reprojection_error", mean_reproj)),
        "tvec_norm": tvec_norm,
        "used_fallback_epnp": used_fallback,
        "trigger_reasons": trigger_reasons,
        "rvec": rvec,
        "tvec": tvec,
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
        objectPoints=SHIRT_KEYPOINTS_3D.astype(np.float64),
        rvec=rvec.astype(np.float64),
        tvec=tvec.astype(np.float64),
        cameraMatrix=camera_matrix.astype(np.float64),
        distCoeffs=dist_coeffs.astype(np.float64),
    )
    image_points = projected.reshape(-1, 2).astype(np.float32)

    x1, y1, x2, y2 = bbox
    bbox_w = max(x2 - x1, 1.0)
    bbox_h = max(y2 - y1, 1.0)

    crop_coords = image_points.copy()
    crop_coords[:, 0] = (crop_coords[:, 0] - x1) / bbox_w * input_size
    crop_coords[:, 1] = (crop_coords[:, 1] - y1) / bbox_h * input_size

    visible = (
        (crop_coords[:, 0] >= 0.0)
        & (crop_coords[:, 0] < input_size)
        & (crop_coords[:, 1] >= 0.0)
        & (crop_coords[:, 1] < input_size)
    ).astype(np.float32)

    heatmap_coords = crop_coords.copy().astype(np.float32)
    heatmap_coords[:, 0] = heatmap_coords[:, 0] / input_size * heatmap_size
    heatmap_coords[:, 1] = heatmap_coords[:, 1] / input_size * heatmap_size

    heatmaps = build_heatmaps(
        heatmap_coords,
        visible,
        heatmap_size=heatmap_size,
        sigma=heatmap_sigma,
    )
    return torch.tensor(heatmaps, dtype=torch.float32, device=device).unsqueeze(0)


def filter_heatmap_by_peak_and_concentration(
    heatmap: torch.Tensor,
    peak_thresh: float,
    concentration_thresh: float,
) -> torch.Tensor:
    if heatmap.ndim != 3:
        raise ValueError(f"Expected [K,H,W] heatmap, got shape {tuple(heatmap.shape)}")
    flat = heatmap.flatten(1)
    peak = flat.amax(dim=1)
    total = flat.sum(dim=1).clamp_min(1e-8)
    concentration = peak / total
    keep = ((peak >= peak_thresh) & (concentration >= concentration_thresh)).to(dtype=heatmap.dtype)
    return heatmap * keep.view(-1, 1, 1)


def sharpen_heatmap(heatmap: torch.Tensor, temperature: float = 0.9, eps: float = 1e-8) -> torch.Tensor:
    if temperature <= 0:
        return heatmap
    flat = heatmap.flatten(1)
    flat = flat / flat.sum(dim=1, keepdim=True).clamp_min(eps)
    logits = torch.log(flat.clamp_min(eps)) / temperature
    sharp = torch.softmax(logits, dim=1)
    return sharp.view_as(heatmap)


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

    def replace_entry(self, index: int, pseudo_heatmap: torch.Tensor, quality: float) -> None:
        self.entries[index].pseudo_heatmap = pseudo_heatmap.detach().cpu().clone()
        self.entries[index].quality = float(quality)

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
    inlier_ratio = safe_float(diagnosis.get("inlier_ratio", 0.0))
    tvec_norm = safe_float(diagnosis.get("tvec_norm", 0.0), cap=args.feature_tvec_norm_cap)
    fallback = bool(diagnosis.get("used_fallback_epnp", False))
    pnp_failed = "pnp_failed" in diagnosis.get("trigger_reasons", [])

    label = (
        mean_conf < args.trigger_confidence
        or num_inliers < args.trigger_min_inliers
        or mean_reproj > args.trigger_reprojection_error
        or inlier_ratio < args.trigger_min_inlier_ratio
        or tvec_norm < args.trigger_min_tvec_norm
        or tvec_norm > args.trigger_max_tvec_norm
        or fallback
        or pnp_failed
    )
    return torch.tensor([1.0 if label else 0.0], dtype=torch.float32, device=device)


@torch.no_grad()
def build_source_prototype(model: DinoHeatmapPoseModel, args, device: torch.device) -> torch.Tensor | None:
    source_dataset = make_dataset(args, args.source_split)
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


def gate_probability(gate: nn.Module, geo_features: torch.Tensor, feat_features: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(gate(geo_features, feat_features))


def choose_gate_decision(step: int, heuristic_label: torch.Tensor, risk_prob: torch.Tensor, args) -> Tuple[bool, float]:
    if step <= args.gate_warmup_steps:
        safe_weight = 1.0 - float(heuristic_label.item())
        return bool(safe_weight >= 0.5), safe_weight

    risk = float(risk_prob.detach().item())
    safe_prob = 1.0 - risk
    if args.gate_usage == "hard":
        return safe_prob >= args.gate_threshold, 1.0 if safe_prob >= args.gate_threshold else 0.0
    if args.gate_usage in {"soft_loss", "soft_lr"}:
        return safe_prob >= args.min_soft_gate_weight, max(safe_prob, 0.0)
    raise ValueError(f"Unsupported gate_usage: {args.gate_usage}")


def scale_optimizer_lr(optimizer: AdamW, scale: float):
    old_lrs = [group["lr"] for group in optimizer.param_groups]
    for group, old_lr in zip(optimizer.param_groups, old_lrs):
        group["lr"] = old_lr * scale
    return old_lrs


def restore_optimizer_lr(optimizer: AdamW, old_lrs) -> None:
    for group, old_lr in zip(optimizer.param_groups, old_lrs):
        group["lr"] = old_lr


def compute_geo_boost(diagnosis: Dict[str, object], args) -> float:
    inliers = float(diagnosis.get("num_ransac_inliers", 0))
    inlier_ratio = float(diagnosis.get("inlier_ratio", 0.0))
    mean_reproj = safe_float(diagnosis.get("mean_reprojection_error", 999.0), cap=args.feature_reprojection_cap)
    tvec_norm = safe_float(diagnosis.get("tvec_norm", 0.0), cap=args.feature_tvec_norm_cap)

    boost = 1.0
    if inliers >= args.geo_boost_min_inliers:
        boost += 0.10
    if inlier_ratio >= args.geo_boost_min_inlier_ratio:
        boost += 0.10
    if mean_reproj <= args.geo_boost_max_reproj:
        boost += 0.10
    if args.trigger_min_tvec_norm <= tvec_norm <= args.trigger_max_tvec_norm:
        boost += 0.10
    return boost


def compute_geometry_reliability(diagnosis: Dict[str, object], args) -> float:
    conf = np.clip(float(diagnosis.get("mean_confidence", 0.0)), 0.0, 1.0)
    inlier_ratio = np.clip(float(diagnosis.get("inlier_ratio", 0.0)), 0.0, 1.0)
    reproj = float(diagnosis.get("mean_reprojection_error", args.feature_reprojection_cap))
    reproj_term = 1.0 - np.clip(reproj / max(args.geometry_max_reproj_error, 1e-6), 0.0, 1.0)
    rel = 0.35 * conf + 0.45 * inlier_ratio + 0.20 * reproj_term
    return float(max(0.0, min(rel, 1.0)))


def should_use_geometry_target(diagnosis: Dict[str, object], args) -> bool:
    if diagnosis.get("rvec") is None or diagnosis.get("tvec") is None:
        return False
    if float(diagnosis.get("mean_confidence", 0.0)) < args.geometry_min_confidence:
        return False
    if int(diagnosis.get("num_ransac_inliers", 0)) < args.geometry_min_inliers:
        return False
    if float(diagnosis.get("inlier_ratio", 0.0)) < args.geometry_min_inlier_ratio:
        return False
    if float(diagnosis.get("mean_reprojection_error", float("inf"))) > args.geometry_max_reproj_error:
        return False
    tvec_norm = safe_float(diagnosis.get("tvec_norm", 0.0), cap=args.feature_tvec_norm_cap)
    if tvec_norm < args.geometry_min_tvec_norm or tvec_norm > args.geometry_max_tvec_norm:
        return False
    return True


def build_teacher_memory_target(teacher_heatmap: torch.Tensor, args) -> torch.Tensor:
    filtered = filter_heatmap_by_peak_and_concentration(
        teacher_heatmap,
        peak_thresh=args.memory_peak_conf_thresh,
        concentration_thresh=args.memory_concentration_thresh,
    )
    if args.teacher_sharpen_temperature < 0.999:
        filtered = sharpen_heatmap(filtered, temperature=args.teacher_sharpen_temperature)
    return filtered


def adapt_with_gate(
    student_model: DinoHeatmapPoseModel,
    teacher_model: DinoHeatmapPoseModel,
    optimizer: AdamW,
    scaler: GradScaler,
    memory_bank: FeatureQualityMemoryBank,
    current_image: torch.Tensor,
    geometry_target: torch.Tensor | None,
    gate_weight: float,
    diagnosis: Dict[str, object],
    args,
    device: torch.device,
) -> Dict[str, float]:
    mem_images, mem_pseudo, mem_weights, _ = memory_bank.sample(args.memory_sample_size, device)

    total_loss_value = 0.0
    loss_st_value = 0.0
    loss_geo_value = 0.0
    loss_reg_value = 0.0

    geo_weight_eff = 0.0
    if geometry_target is not None:
        geo_weight_eff = args.lambda_geo * compute_geo_boost(diagnosis, args) * compute_geometry_reliability(diagnosis, args)

    student_model.train()
    teacher_model.eval()
    for _ in range(args.adapt_steps):
        optimizer.zero_grad(set_to_none=True)
        old_lrs = None
        if args.gate_usage == "soft_lr":
            old_lrs = scale_optimizer_lr(optimizer, max(gate_weight, args.min_lr_gate_scale))

        with autocast(enabled=device.type == "cuda"):
            mem_pred = student_model(mem_images)
            loss_st = weighted_heatmap_mse(mem_pred, mem_pseudo.detach(), weights=mem_weights)

            loss_geo = mem_pred.new_tensor(0.0)
            if geometry_target is not None and geo_weight_eff > 0.0:
                current_pred = student_model(current_image)
                loss_geo = F.mse_loss(current_pred, geometry_target.detach())

            loss_reg = mem_pred.new_tensor(0.0)
            if args.lambda_reg > 0.0:
                with torch.no_grad():
                    teacher_mem_pred = teacher_model(mem_images)
                loss_reg = confidence_weighted_regularization(mem_pred, teacher_mem_pred, tau=args.tau)

            total_loss = (
                args.lambda_self_training * loss_st
                + geo_weight_eff * loss_geo
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

        update_ema(student_model, teacher_model, momentum=args.teacher_momentum)

        total_loss_value = float(total_loss.item())
        loss_st_value = float(loss_st.item())
        loss_geo_value = float(loss_geo.item())
        loss_reg_value = float(loss_reg.item())

    return {
        "total_loss": total_loss_value,
        "loss_self_training": loss_st_value,
        "loss_geometry": loss_geo_value,
        "loss_regularization": loss_reg_value,
        "effective_lambda_geo": float(geo_weight_eff),
    }


def refresh_memory_entries(
    teacher_model: DinoHeatmapPoseModel,
    memory_bank: FeatureQualityMemoryBank,
    sampled_indices: np.ndarray,
    args,
    device: torch.device,
) -> None:
    if len(sampled_indices) == 0:
        return

    unique_indices = sorted(set(int(i) for i in sampled_indices.tolist()))
    for index in unique_indices:
        entry = memory_bank.entries[index]
        image = entry.image.unsqueeze(0).to(device)
        with torch.no_grad(), autocast(enabled=device.type == "cuda"):
            teacher_heatmap, _ = teacher_model(image, return_features=True)

        pseudo = build_teacher_memory_target(teacher_heatmap.squeeze(0), args)
        flat = pseudo.flatten(1)
        peak = flat.amax(dim=1)
        total = flat.sum(dim=1).clamp_min(1e-8)
        concentration = peak / total
        pseudo_quality = float((0.60 * peak.mean() + 0.40 * concentration.mean()).item())

        if pseudo_quality > entry.quality + args.memory_update_margin:
            memory_bank.replace_entry(index, pseudo, pseudo_quality)


def run_learnable_trigger_tta(args) -> None:
    set_seed(args.seed)

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

    camera_matrix, dist_coeffs = load_camera_from_shirt(Path(args.data_root))
    student_model = load_source_model(args.source_checkpoint, device)
    teacher_model = clone_teacher_model(student_model)
    source_prototype = build_source_prototype(teacher_model, args, device)

    trainable_params = configure_trainable_parameters(student_model, args.update_scope)
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=device.type == "cuda")

    gate = DualBranchRiskGate(hidden_dim=args.gate_hidden_dim, dropout=args.gate_dropout).to(device)
    gate_optimizer = AdamW(gate.parameters(), lr=args.gate_lr, weight_decay=args.gate_weight_decay)

    memory_bank = FeatureQualityMemoryBank(capacity=args.memory_capacity)

    history: List[Dict[str, object]] = []
    loss_history: List[Dict[str, object]] = []
    gate_history: List[Dict[str, object]] = []

    for step, batch in enumerate(tqdm(target_loader, desc="learnable_trigger_shirt_dual_branch_v4_rot_safe"), start=1):
        if args.max_samples is not None and step > args.max_samples:
            break

        image = batch["image"].to(device, non_blocking=True)
        bbox = tensor_bbox_to_tuple(batch["bbox"])
        image_name = tensor_image_name(batch["image_name"])

        teacher_model.eval()
        with torch.no_grad(), autocast(enabled=device.type == "cuda"):
            teacher_heatmap, teacher_cls_token = teacher_model(image, return_features=True)

        diagnosis = diagnose_prediction(
            heatmap=teacher_heatmap,
            bbox=bbox,
            input_size=args.input_size,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            args=args,
        )

        geo_features = build_geo_features(diagnosis, args, device)
        feat_features = memory_bank.feature_distances(teacher_cls_token.squeeze(0), source_prototype).to(device).unsqueeze(0)
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
        tvec_norm = safe_float(diagnosis.get("tvec_norm", 0.0), cap=args.feature_tvec_norm_cap)

        if (
            float(diagnosis["quality"]) >= args.memory_min_quality
            and float(diagnosis["mean_confidence"]) >= args.memory_min_confidence
            and int(diagnosis["num_ransac_inliers"]) >= args.memory_min_inliers
            and float(diagnosis["inlier_ratio"]) >= args.memory_min_inlier_ratio
            and float(diagnosis["mean_reprojection_error"]) <= args.memory_max_reproj_error
            and args.memory_min_tvec_norm <= tvec_norm <= args.memory_max_tvec_norm
        ):
            teacher_pseudo = build_teacher_memory_target(teacher_heatmap.squeeze(0), args)
            memory_bank.push(
                image=image.squeeze(0),
                pseudo_heatmap=teacher_pseudo,
                feature=teacher_cls_token.squeeze(0),
                quality=float(diagnosis["quality"]),
                image_name=image_name,
            )
            pushed_to_memory = True

        adapted = False
        sampled_indices = np.array([], dtype=np.int64)
        losses = {
            "total_loss": 0.0,
            "loss_self_training": 0.0,
            "loss_geometry": 0.0,
            "loss_regularization": 0.0,
            "effective_lambda_geo": 0.0,
        }

        if should_adapt and len(memory_bank) >= args.min_memory_for_update:
            geometry_target = None
            if should_use_geometry_target(diagnosis, args):
                geometry_target = geometry_target_from_pose(
                    diagnosis["rvec"],
                    diagnosis["tvec"],
                    bbox=bbox,
                    input_size=args.input_size,
                    heatmap_size=args.heatmap_size,
                    heatmap_sigma=args.heatmap_sigma,
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                    device=device,
                )

            _, _, _, sampled_indices = memory_bank.sample(args.memory_sample_size, device)
            losses = adapt_with_gate(
                student_model=student_model,
                teacher_model=teacher_model,
                optimizer=optimizer,
                scaler=scaler,
                memory_bank=memory_bank,
                current_image=image,
                geometry_target=geometry_target,
                gate_weight=gate_weight,
                diagnosis=diagnosis,
                args=args,
                device=device,
            )
            refresh_memory_entries(
                teacher_model=teacher_model,
                memory_bank=memory_bank,
                sampled_indices=sampled_indices,
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
                "trigger_mode": "dual_branch_rot_safe",
                "gate_usage": args.gate_usage,
                "heuristic_triggered": bool(risk_label.item() < 0.5),
                "adapted": adapted,
                "pushed_to_memory": pushed_to_memory,
                "memory_size": len(memory_bank),
                "quality": float(diagnosis["quality"]),
                "mean_confidence": float(diagnosis["mean_confidence"]),
                "num_ransac_inliers": int(diagnosis["num_ransac_inliers"]),
                "inlier_ratio": float(diagnosis["inlier_ratio"]),
                "mean_reprojection_error": float(diagnosis["mean_reprojection_error"]),
                "tvec_norm": tvec_norm,
                **gate_record,
                **losses,
            }
        )

    summary = {
        "dataset": "SHIRT",
        "roe": args.roe,
        "domain": args.domain,
        "target_split": args.target_split,
        "source_checkpoint": args.source_checkpoint,
        "num_samples": len(history),
        "num_safe_to_adapt": int(sum(1 for item in history if item["heuristic_triggered"])),
        "num_adapted": int(sum(1 for item in history if item["adapted"])),
        "num_pushed_to_memory": int(sum(1 for item in history if item["pushed_to_memory"])),
        "trigger_mode": "dual_branch_rot_safe",
        "gate_usage": args.gate_usage,
        "update_scope": args.update_scope,
        "teacher_momentum": args.teacher_momentum,
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
        "model_state_dict": teacher_model.state_dict() if args.save_teacher else student_model.state_dict(),
        "student_model_state_dict": student_model.state_dict(),
        "teacher_model_state_dict": teacher_model.state_dict(),
        "model_name": args.model_name,
        "input_size": args.input_size,
        "heatmap_size": args.heatmap_size,
        "heatmap_sigma": args.heatmap_sigma,
        "mid_channels": args.mid_channels,
        "num_deconv_layers": args.num_deconv_layers,
        "num_keypoints": args.num_keypoints,
        "dataset": "SHIRT",
        "roe": args.roe,
        "domain": args.domain,
        "adaptation": "learnable_trigger_single_model_tta_shirt_dual_branch_v4_rot_safe",
        "trigger_mode": "dual_branch_rot_safe",
        "gate_usage": args.gate_usage,
        "update_scope": args.update_scope,
        "summary": summary,
        "gate_state_dict": gate.state_dict(),
    }
    torch.save(checkpoint, output_dir / "tta_final.pth")
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="SHIRT_Dataset")
    parser.add_argument("--source_checkpoint", type=str, required=True)
    parser.add_argument("--roe", type=str, choices=["roe1", "roe2", "all"], default="all")
    parser.add_argument("--domain", type=str, default="synthetic")
    parser.add_argument("--source_split", type=str, choices=["train", "val", "all"], default="train")
    parser.add_argument("--target_split", type=str, choices=["train", "val", "all"], default="val")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="output/dinov3_heatmap_shirt_dual_branch_tta_v4_rot_safe")

    parser.add_argument("--model_name", type=str, default="vit_base_patch16_dinov3.lvd1689m")
    parser.add_argument("--input_size", type=int, default=384)
    parser.add_argument("--heatmap_size", type=int, default=96)
    parser.add_argument("--heatmap_sigma", type=float, default=3.0)
    parser.add_argument("--mid_channels", type=int, default=256)
    parser.add_argument("--num_deconv_layers", type=int, default=2)
    parser.add_argument("--num_keypoints", type=int, default=11)
    parser.add_argument("--expand_ratio", type=float, default=1.25)

    parser.add_argument("--update_scope", type=str, choices=["decoder", "decoder_last_block"], default="decoder")

    parser.add_argument("--gate_usage", type=str, choices=["hard", "soft_loss", "soft_lr"], default="hard")
    parser.add_argument("--gate_threshold", type=float, default=0.65)
    parser.add_argument("--min_soft_gate_weight", type=float, default=0.10)
    parser.add_argument("--min_lr_gate_scale", type=float, default=0.15)
    parser.add_argument("--gate_hidden_dim", type=int, default=32)
    parser.add_argument("--gate_dropout", type=float, default=0.10)
    parser.add_argument("--gate_lr", type=float, default=5e-4)
    parser.add_argument("--gate_weight_decay", type=float, default=1e-4)
    parser.add_argument("--gate_warmup_steps", type=int, default=32)

    parser.add_argument("--prototype_batch_size", type=int, default=32)
    parser.add_argument("--prototype_max_samples", type=int, default=512)
    parser.add_argument("--feature_reprojection_cap", type=float, default=50.0)
    parser.add_argument("--feature_tvec_norm_cap", type=float, default=20.0)

    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adapt_steps", type=int, default=1)
    parser.add_argument("--teacher_momentum", type=float, default=0.999)
    parser.add_argument("--save_teacher", action="store_true")

    parser.add_argument("--memory_capacity", type=int, default=64)
    parser.add_argument("--memory_sample_size", type=int, default=8)
    parser.add_argument("--min_memory_for_update", type=int, default=8)
    parser.add_argument("--memory_min_quality", type=float, default=0.32)
    parser.add_argument("--memory_min_confidence", type=float, default=0.60)
    parser.add_argument("--memory_min_inliers", type=int, default=8)
    parser.add_argument("--memory_min_inlier_ratio", type=float, default=0.75)
    parser.add_argument("--memory_max_reproj_error", type=float, default=4.0)
    parser.add_argument("--memory_min_tvec_norm", type=float, default=4.5)
    parser.add_argument("--memory_max_tvec_norm", type=float, default=8.5)
    parser.add_argument("--memory_peak_conf_thresh", type=float, default=0.45)
    parser.add_argument("--memory_concentration_thresh", type=float, default=0.20)
    parser.add_argument("--memory_update_margin", type=float, default=0.02)
    parser.add_argument("--teacher_sharpen_temperature", type=float, default=0.90)

    parser.add_argument("--lambda_self_training", type=float, default=0.20)
    parser.add_argument("--lambda_geo", type=float, default=0.08)
    parser.add_argument("--lambda_reg", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    parser.add_argument("--geo_boost_min_inliers", type=int, default=8)
    parser.add_argument("--geo_boost_min_inlier_ratio", type=float, default=0.80)
    parser.add_argument("--geo_boost_max_reproj", type=float, default=4.0)

    parser.add_argument("--trigger_confidence", type=float, default=0.20)
    parser.add_argument("--trigger_min_inliers", type=int, default=6)
    parser.add_argument("--trigger_min_inlier_ratio", type=float, default=0.55)
    parser.add_argument("--trigger_reprojection_error", type=float, default=6.0)
    parser.add_argument("--trigger_min_tvec_norm", type=float, default=4.5)
    parser.add_argument("--trigger_max_tvec_norm", type=float, default=8.5)

    parser.add_argument("--geometry_min_confidence", type=float, default=0.60)
    parser.add_argument("--geometry_min_inliers", type=int, default=8)
    parser.add_argument("--geometry_min_inlier_ratio", type=float, default=0.80)
    parser.add_argument("--geometry_max_reproj_error", type=float, default=3.5)
    parser.add_argument("--geometry_min_tvec_norm", type=float, default=4.5)
    parser.add_argument("--geometry_max_tvec_norm", type=float, default=8.5)

    parser.add_argument("--quality_reprojection_cap", type=float, default=50.0)

    parser.add_argument("--nms_kernel", type=int, default=3)
    parser.add_argument("--disable_nms", action="store_true")
    parser.add_argument("--disable_subpixel", action="store_true")
    parser.add_argument("--subpixel_radius", type=int, default=2)
    parser.add_argument("--min_confidence", type=float, default=0.08)
    parser.add_argument("--top_k", type=int, default=7)
    parser.add_argument("--min_points", type=int, default=6)
    parser.add_argument("--ransac_reproj_error", type=float, default=4.5)
    parser.add_argument("--ransac_iterations", type=int, default=100)
    parser.add_argument("--ransac_confidence", type=float, default=0.999)
    parser.add_argument("--disable_iterative_refine", action="store_true")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_learnable_trigger_tta(parse_args())
