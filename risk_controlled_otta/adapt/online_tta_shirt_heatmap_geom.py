from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
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
from losses.otta_losses import (
    class_awareness_consistency_loss,
    masked_heatmap_consistency_loss,
    self_training_loss,
    total_target_loss,
)
from memory.dynamic_bank import (
    DynamicMemoryBank,
    heatmap_confidence_score,
    sample_from_memory_bank,
    update_memory_pseudolabels,
)

from data.crop_and_heatmap import (
    compute_expanded_bbox,
    crop_and_resize,
    normalize_image,
    project_keypoints,
    visible_keypoints_mask,
)


SHIRT_KEYPOINTS_3D = np.array(
    [
        [-0.37,   -0.385,   0.3215],
        [-0.37,    0.385,   0.3215],
        [ 0.37,    0.385,   0.3215],
        [ 0.37,   -0.385,   0.3215],
        [-0.37,   -0.264,   0.0   ],
        [-0.37,    0.304,   0.0   ],
        [ 0.37,    0.304,   0.0   ],
        [ 0.37,   -0.264,   0.0   ],
        [-0.5427,  0.4877,  0.2535],
        [ 0.5427,  0.4877,  0.2591],
        [ 0.305,  -0.579,   0.2515],
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
            "roe": record["roe"],
            "bbox": torch.tensor(bbox, dtype=torch.float32),
            "camera_matrix": torch.tensor(self.camera_matrix, dtype=torch.float32),
            "dist_coeffs": torch.tensor(self.dist_coeffs, dtype=torch.float32),
            "quaternion": torch.tensor(quaternion, dtype=torch.float32),
            "translation": torch.tensor(translation, dtype=torch.float32),
        }


class TargetTTADataset(Dataset):
    def __init__(
        self,
        data_root: str,
        roes: List[str],
        domain: str,
        split: str,
        input_size: int,
        val_ratio: float = 0.1,
        seed: int = 42,
        expand_ratio: float = 1.25,
    ) -> None:
        self.data_root = Path(data_root)
        self.roes = roes
        self.domain = domain
        self.split = split
        self.input_size = int(input_size)
        self.expand_ratio = float(expand_ratio)

        self.records = load_shirt_records(self.data_root, roes, domain=domain)
        rng = np.random.RandomState(seed)
        indices = np.arange(len(self.records))
        rng.shuffle(indices)

        n_val = max(1, int(len(indices) * val_ratio))
        val_idx = set(indices[:n_val].tolist())
        train_idx = set(indices[n_val:].tolist())

        if split == "train":
            self.records = [self.records[i] for i in range(len(self.records)) if i in train_idx]
        elif split == "val":
            self.records = [self.records[i] for i in range(len(self.records)) if i in val_idx]
        elif split == "all":
            self.records = self.records
        else:
            raise ValueError(f"Unsupported split: {split}")

        self.camera_matrix, self.dist_coeffs = load_camera_from_shirt(self.data_root)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        ann = self.records[index]
        image = np.array(Image.open(ann["image_path"]).convert("RGB"))

        gt_quaternion = np.asarray(ann["quaternion"], dtype=np.float32)
        gt_translation = np.asarray(ann["translation"], dtype=np.float32)

        gt_keypoints_2d = project_keypoints(
            SHIRT_KEYPOINTS_3D,
            gt_quaternion,
            gt_translation,
            self.camera_matrix,
            self.dist_coeffs,
        )

        visible = visible_keypoints_mask(gt_keypoints_2d, (image.shape[1], image.shape[0]))
        bbox = compute_expanded_bbox(
            gt_keypoints_2d,
            visible,
            (image.shape[1], image.shape[0]),
            expand_ratio=self.expand_ratio,
        )
        crop_image, _ = crop_and_resize(image, gt_keypoints_2d, bbox, output_size=self.input_size)
        image_tensor = normalize_image(crop_image)

        return {
            "image": image_tensor,
            "bbox": torch.tensor(bbox, dtype=torch.float32),
            "camera_matrix": torch.tensor(self.camera_matrix, dtype=torch.float32),
            "dist_coeffs": torch.tensor(self.dist_coeffs, dtype=torch.float32),
            "filename": ann["filename"],
            "roe": ann["roe"],
        }


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
        debug["ransac_inliers"] = selected_indices[inliers.reshape(-1)].tolist()

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


def random_discontinuous_mask(images: torch.Tensor, mask_ratio: float = 0.8, patch_size: int = 16) -> torch.Tensor:
    batch_size, _, height, width = images.shape
    grid_h = height // patch_size
    grid_w = width // patch_size
    num_patches = grid_h * grid_w
    num_mask = max(1, int(num_patches * mask_ratio))

    patch_mask = torch.ones(batch_size, grid_h, grid_w, device=images.device)
    for batch_idx in range(batch_size):
        chosen = torch.randperm(num_patches, device=images.device)[:num_mask]
        patch_mask[batch_idx].view(-1)[chosen] = 0.0

    mask = patch_mask.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2).unsqueeze(1)
    return images * mask


@torch.no_grad()
def update_ema_teacher(student_model, teacher_model, alpha: float = 0.999) -> None:
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1.0 - alpha)


def set_batchnorm_eval(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            m.eval()


def configure_adaptation_modules(model: nn.Module, adapt_modules: str = "decoder") -> list[nn.Parameter]:
    trainable_params: list[nn.Parameter] = []

    for _, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        should_train = False
        if adapt_modules == "decoder":
            should_train = ("decoder" in name)
        elif adapt_modules == "decoder_norm":
            should_train = ("decoder" in name) or ("norm" in name)
        elif adapt_modules == "all":
            should_train = True
        else:
            raise ValueError(f"Unsupported adapt_modules: {adapt_modules}")

        if should_train:
            param.requires_grad = True
            trainable_params.append(param)

    return trainable_params


@torch.no_grad()
def build_source_prototype(model, dataloader, device) -> torch.Tensor:
    model.eval()
    features = []
    for batch in tqdm(dataloader, desc="build_prototype", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        with autocast(enabled=device.type == "cuda"):
            _, cls_token = model(images, return_features=True)
        features.append(cls_token.detach().cpu())
    prototype = torch.cat(features, dim=0).mean(dim=0)
    return torch.cat([prototype, prototype], dim=0)


@torch.no_grad()
def initialize_memory_bank(dataloader, memory_bank, max_samples: int = 16) -> None:
    dataset = getattr(dataloader, "dataset", None)
    if dataset is None:
        raise ValueError("Memory-bank initialization requires access to the source dataset.")

    num_samples = min(max_samples, len(dataset))
    sampled_indices = np.random.choice(len(dataset), size=num_samples, replace=False)
    for sample_index in tqdm(sampled_indices, desc="init_memory", leave=False):
        sample = dataset[int(sample_index)]
        memory_bank.push(sample["image"], sample["heatmap"])


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
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device)


def make_source_dataset(args, split: str, use_source_augmentation: bool = False) -> ShirtDinoHeatmapDataset:
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
        use_source_augmentation=use_source_augmentation,
    )


def teacher_heatmap_pass(model, images, device):
    with torch.no_grad():
        with autocast(enabled=device.type == "cuda"):
            return model(images)


def geometric_gate_sample(
    teacher_heatmap: torch.Tensor,
    bbox_tensor: torch.Tensor,
    camera_matrix_tensor: torch.Tensor,
    dist_coeffs_tensor: torch.Tensor,
    input_size: int,
    heatmap_size: int,
    pnp_min_confidence: float,
    top_k: int,
    min_points: int,
    ransac_reproj_error: float,
    ransac_iterations: int,
    ransac_confidence: float,
    gate_min_confidence_mean: float,
    gate_min_inliers: int,
    gate_max_reproj_error: float,
) -> Tuple[bool, Dict[str, float]]:
    crop_coords_hm, confidences = decode_heatmap_to_keypoints(
        teacher_heatmap,
        apply_nms=True,
        nms_kernel=3,
        use_subpixel=True,
        subpixel_radius=2,
    )

    crop_coords = crop_coords_hm[0].copy()
    crop_coords[:, 0] = crop_coords[:, 0] * input_size / heatmap_size
    crop_coords[:, 1] = crop_coords[:, 1] * input_size / heatmap_size

    bbox = tuple(float(x) for x in bbox_tensor[0].detach().cpu().numpy().tolist())
    camera_matrix = camera_matrix_tensor[0].detach().cpu().numpy()
    dist_coeffs = dist_coeffs_tensor[0].detach().cpu().numpy()

    image_coords = map_crop_coords_to_image(
        crop_coords,
        bbox,
        crop_size_w=float(input_size),
        crop_size_h=float(input_size),
    )

    try:
        _, _, debug = solve_pose_robust(
            image_points=image_coords,
            confidences=confidences[0],
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            min_confidence=pnp_min_confidence,
            top_k=top_k,
            min_points=min_points,
            ransac_reproj_error=ransac_reproj_error,
            ransac_iterations=ransac_iterations,
            confidence_prob=ransac_confidence,
            use_iterative_refine=True,
        )
        mean_conf = float(debug.get("mean_selected_confidence", 0.0))
        num_inliers = int(len(debug.get("ransac_inliers", [])))
        mean_reproj = float(debug.get("mean_reprojection_error", 1e9))

        passed = (
            mean_conf >= gate_min_confidence_mean
            and num_inliers >= gate_min_inliers
            and mean_reproj <= gate_max_reproj_error
        )
        metrics = {
            "mean_selected_confidence": mean_conf,
            "num_inliers": num_inliers,
            "mean_reprojection_error": mean_reproj,
            "passed": float(passed),
        }
        return passed, metrics
    except Exception:
        return False, {
            "mean_selected_confidence": 0.0,
            "num_inliers": 0,
            "mean_reprojection_error": 1e9,
            "passed": 0.0,
        }


def save_loss_curves(
    loss_history: dict[str, list[float]],
    output_dir: Path,
    lambda_st: float,
    lambda_ca: float,
    plot_stride: int = 25,
) -> None:
    with (output_dir / "loss_history.json").open("w", encoding="utf-8") as handle:
        json.dump(loss_history, handle, indent=2)

    if len(loss_history["step"]) == 0:
        return

    steps = np.asarray(loss_history["step"], dtype=np.int32)
    total_loss = np.asarray(loss_history["total_loss"], dtype=np.float32)
    raw_st = np.asarray(loss_history["loss_st"], dtype=np.float32)
    raw_mh = np.asarray(loss_history["loss_mh"], dtype=np.float32)
    raw_ca = np.asarray(loss_history["loss_ca"], dtype=np.float32)
    weighted_st = raw_st * lambda_st
    weighted_ca = raw_ca * lambda_ca

    stride = max(1, int(plot_stride))
    sampled = np.arange(0, len(steps), stride, dtype=np.int32)
    if len(steps) > 0 and sampled[-1] != len(steps) - 1:
        sampled = np.append(sampled, len(steps) - 1)

    plt.figure(figsize=(8, 5))
    plt.plot(steps[sampled], total_loss[sampled], label="total loss", linewidth=1.8)
    plt.plot(steps[sampled], weighted_st[sampled], label=f"{lambda_st} x L_st", linewidth=1.5)
    plt.plot(steps[sampled], raw_mh[sampled], label="L_mh", linewidth=1.5)
    plt.plot(steps[sampled], weighted_ca[sampled], label=f"{lambda_ca} x L_ca", linewidth=1.5)
    plt.xlabel("Training Step")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "otta_losses_weighted.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(steps[sampled], total_loss[sampled], label="total loss", linewidth=1.8)
    plt.plot(steps[sampled], raw_st[sampled], label="L_st", linewidth=1.5)
    plt.plot(steps[sampled], raw_mh[sampled], label="L_mh", linewidth=1.5)
    plt.plot(steps[sampled], raw_ca[sampled], label="L_ca", linewidth=1.5)
    plt.xlabel("Training Step")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "otta_losses_raw.png", dpi=200)
    plt.close()


def online_adapt(args) -> None:
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_dataset = make_source_dataset(args, args.source_split, use_source_augmentation=False)
    roes = ["roe1", "roe2"] if args.roe == "all" else [args.roe]
    target_dataset = TargetTTADataset(
        data_root=args.data_root,
        roes=roes,
        domain=args.domain,
        split=args.target_split,
        input_size=args.input_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        expand_ratio=args.expand_ratio,
    )

    source_loader = DataLoader(
        source_dataset,
        batch_size=args.prototype_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    student_model = load_source_model(args.source_checkpoint, device)
    teacher_model = copy.deepcopy(student_model)

    source_prototype = build_source_prototype(student_model, source_loader, device).to(device)

    memory_bank = DynamicMemoryBank(capacity=args.memory_capacity)
    initialize_memory_bank(source_loader, memory_bank, max_samples=args.memory_capacity)

    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    trainable_params = configure_adaptation_modules(student_model, adapt_modules=args.adapt_modules)
    student_model.train()
    set_batchnorm_eval(student_model)

    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=device.type == "cuda")

    lambda_st = args.lambda_st
    lambda_ca = args.lambda_ca

    loss_history = {
        "step": [],
        "total_loss": [],
        "loss_st": [],
        "loss_mh": [],
        "loss_ca": [],
        "num_reliable": [],
        "geom_passed": [],
        "geom_conf": [],
        "geom_inliers": [],
        "geom_reproj": [],
    }

    skipped_geom_gate = 0
    skipped_bank_conf = 0
    skipped_unreliable_batch = 0

    for step, batch in enumerate(tqdm(target_loader, desc="online_tta_shirt_geom"), start=1):
        x_online = batch["image"].to(device, non_blocking=True)
        bbox = batch["bbox"]
        camera_matrix = batch["camera_matrix"]
        dist_coeffs = batch["dist_coeffs"]

        teacher_online_heatmap = teacher_heatmap_pass(teacher_model, x_online, device)

        passed_geom, geom_metrics = geometric_gate_sample(
            teacher_heatmap=teacher_online_heatmap,
            bbox_tensor=bbox,
            camera_matrix_tensor=camera_matrix,
            dist_coeffs_tensor=dist_coeffs,
            input_size=args.input_size,
            heatmap_size=args.heatmap_size,
            pnp_min_confidence=args.pnp_min_confidence,
            top_k=args.top_k,
            min_points=args.min_points,
            ransac_reproj_error=args.ransac_reproj_error,
            ransac_iterations=args.ransac_iterations,
            ransac_confidence=args.ransac_confidence,
            gate_min_confidence_mean=args.gate_min_confidence_mean,
            gate_min_inliers=args.gate_min_inliers,
            gate_max_reproj_error=args.gate_max_reproj_error,
        )

        if not passed_geom:
            skipped_geom_gate += 1
            continue

        with torch.no_grad():
            online_scores = heatmap_confidence_score(teacher_online_heatmap)
            best_index = int(online_scores.argmax().item())
            best_conf = float(online_scores[best_index].item())

            if best_conf >= args.bank_push_conf_thresh:
                memory_bank.push(x_online[best_index], teacher_online_heatmap[best_index])
            else:
                skipped_bank_conf += 1
                continue

        if len(memory_bank) < max(args.memory_sample_size, args.min_reliable_samples):
            continue

        x_memory, old_pseudo, mem_indices = sample_from_memory_bank(
            memory_bank, args.memory_sample_size, device
        )

        with torch.no_grad():
            pseudo_scores = heatmap_confidence_score(old_pseudo)
            reliable_mask = pseudo_scores >= args.memory_conf_thresh

        num_reliable = int(reliable_mask.sum().item())
        if num_reliable < args.min_reliable_samples:
            skipped_unreliable_batch += 1
            continue

        x_memory = x_memory[reliable_mask]
        old_pseudo = old_pseudo[reliable_mask]
        mem_indices = [mem_indices[i] for i in range(len(mem_indices)) if bool(reliable_mask[i].item())]

        masked_memory = random_discontinuous_mask(
            x_memory,
            mask_ratio=args.mask_ratio,
            patch_size=args.patch_size,
        )

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=device.type == "cuda"):
            h_student, cls_student = student_model(x_memory, return_features=True)
            with torch.no_grad():
                h_teacher, cls_teacher = teacher_model(masked_memory, return_features=True)

            loss_st = self_training_loss(h_student, old_pseudo)
            loss_mh = masked_heatmap_consistency_loss(h_student, h_teacher, tau=args.tau)
            loss_ca = class_awareness_consistency_loss(cls_student, cls_teacher, source_prototype)

            total_loss = total_target_loss(
                loss_st,
                loss_mh,
                loss_ca,
                lambda_st=lambda_st,
                lambda_ca=lambda_ca,
            )

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            refreshed_teacher_heatmap = teacher_model(x_memory)
            update_memory_pseudolabels(memory_bank, mem_indices, refreshed_teacher_heatmap, old_pseudo)
            update_ema_teacher(student_model, teacher_model, alpha=args.ema_alpha)

        loss_history["step"].append(step)
        loss_history["total_loss"].append(float(total_loss.item()))
        loss_history["loss_st"].append(float(loss_st.item()))
        loss_history["loss_mh"].append(float(loss_mh.item()))
        loss_history["loss_ca"].append(float(loss_ca.item()))
        loss_history["num_reliable"].append(num_reliable)
        loss_history["geom_passed"].append(float(geom_metrics["passed"]))
        loss_history["geom_conf"].append(float(geom_metrics["mean_selected_confidence"]))
        loss_history["geom_inliers"].append(int(geom_metrics["num_inliers"]))
        loss_history["geom_reproj"].append(float(geom_metrics["mean_reprojection_error"]))

    torch.save(
        {
            "model_state_dict": teacher_model.state_dict(),
            "teacher_state_dict": teacher_model.state_dict(),
            "student_state_dict": student_model.state_dict(),
            "model_name": args.model_name,
            "input_size": args.input_size,
            "heatmap_size": args.heatmap_size,
            "heatmap_sigma": args.heatmap_sigma,
            "mid_channels": args.mid_channels,
            "num_deconv_layers": args.num_deconv_layers,
            "num_keypoints": 11,
            "dataset": "SHIRT",
            "roe": args.roe,
            "domain": args.domain,
            "source_split": args.source_split,
            "target_split": args.target_split,
            "adapt_modules": args.adapt_modules,
            "lr": args.lr,
            "lambda_st": args.lambda_st,
            "lambda_ca": args.lambda_ca,
            "ema_alpha": args.ema_alpha,
            "bank_push_conf_thresh": args.bank_push_conf_thresh,
            "memory_conf_thresh": args.memory_conf_thresh,
            "min_reliable_samples": args.min_reliable_samples,
            "gate_min_confidence_mean": args.gate_min_confidence_mean,
            "gate_min_inliers": args.gate_min_inliers,
            "gate_max_reproj_error": args.gate_max_reproj_error,
            "skipped_geom_gate": skipped_geom_gate,
            "skipped_bank_conf": skipped_bank_conf,
            "skipped_unreliable_batch": skipped_unreliable_batch,
        },
        output_dir / "tta_final.pth",
    )

    save_loss_curves(
        loss_history,
        output_dir,
        lambda_st=lambda_st,
        lambda_ca=lambda_ca,
        plot_stride=args.plot_stride,
    )

    summary = {
        "dataset": "SHIRT",
        "roe": args.roe,
        "domain": args.domain,
        "source_split": args.source_split,
        "target_split": args.target_split,
        "adapt_modules": args.adapt_modules,
        "lr": args.lr,
        "lambda_st": args.lambda_st,
        "lambda_ca": args.lambda_ca,
        "ema_alpha": args.ema_alpha,
        "bank_push_conf_thresh": args.bank_push_conf_thresh,
        "memory_conf_thresh": args.memory_conf_thresh,
        "min_reliable_samples": args.min_reliable_samples,
        "gate_min_confidence_mean": args.gate_min_confidence_mean,
        "gate_min_inliers": args.gate_min_inliers,
        "gate_max_reproj_error": args.gate_max_reproj_error,
        "skipped_geom_gate": skipped_geom_gate,
        "skipped_bank_conf": skipped_bank_conf,
        "skipped_unreliable_batch": skipped_unreliable_batch,
        "num_optimization_steps": len(loss_history["step"]),
    }
    with (output_dir / "adapt_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


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
    parser.add_argument("--output_dir", type=str, default="output/dinov3_heatmap_shirt_tta_geom")

    parser.add_argument("--model_name", type=str, default="vit_base_patch16_dinov3.lvd1689m")
    parser.add_argument("--input_size", type=int, default=384)
    parser.add_argument("--heatmap_size", type=int, default=96)
    parser.add_argument("--heatmap_sigma", type=float, default=3.0)
    parser.add_argument("--expand_ratio", type=float, default=1.25)
    parser.add_argument("--mid_channels", type=int, default=256)
    parser.add_argument("--num_deconv_layers", type=int, default=2)

    parser.add_argument("--adapt_modules", type=str, default="decoder", choices=["decoder", "decoder_norm", "all"])
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--memory_capacity", type=int, default=16)
    parser.add_argument("--memory_sample_size", type=int, default=16)
    parser.add_argument("--bank_push_conf_thresh", type=float, default=0.20)
    parser.add_argument("--memory_conf_thresh", type=float, default=0.20)
    parser.add_argument("--min_reliable_samples", type=int, default=4)

    parser.add_argument("--mask_ratio", type=float, default=0.8)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--ema_alpha", type=float, default=0.999)

    parser.add_argument("--lambda_st", type=float, default=1.0)
    parser.add_argument("--lambda_ca", type=float, default=0.01)

    parser.add_argument("--pnp_min_confidence", type=float, default=0.05)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--min_points", type=int, default=6)
    parser.add_argument("--ransac_reproj_error", type=float, default=6.0)
    parser.add_argument("--ransac_iterations", type=int, default=100)
    parser.add_argument("--ransac_confidence", type=float, default=0.999)

    parser.add_argument("--gate_min_confidence_mean", type=float, default=0.75)
    parser.add_argument("--gate_min_inliers", type=int, default=6)
    parser.add_argument("--gate_max_reproj_error", type=float, default=12.0)

    parser.add_argument("--prototype_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--plot_stride", type=int, default=25)
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    online_adapt(parse_args())