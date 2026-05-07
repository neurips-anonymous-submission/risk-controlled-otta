from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPO_ROOT / "output"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(OUTPUT_ROOT) not in sys.path:
    sys.path.insert(0, str(OUTPUT_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dinov2_heatmap_otta.data.dino_heatmap_dataset import SpeedPlusDinoHeatmapDataset
from dinov2_heatmap_otta.models.dinov2_pose_model import DinoHeatmapPoseModel
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
    SPEEDPLUS_3D_KEYPOINTS,
    compute_expanded_bbox,
    crop_and_resize,
    load_camera,
    normalize_image,
    project_keypoints,
    visible_keypoints_mask,
)


# =========================
# Checkpoint / size helpers
# =========================
def infer_input_size_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    patch_size: int = 14,
) -> Optional[int]:
    pos_embed = state_dict.get("encoder.embeddings.position_embeddings", None)
    if pos_embed is None:
        return None

    if pos_embed.ndim != 3:
        raise ValueError(f"Unexpected position_embeddings shape: {tuple(pos_embed.shape)}")

    num_tokens = int(pos_embed.shape[1])
    num_patch_tokens = num_tokens - 1
    if num_patch_tokens <= 0:
        raise ValueError(f"Invalid position_embeddings shape: {tuple(pos_embed.shape)}")

    grid_size = int(round(num_patch_tokens ** 0.5))
    if grid_size * grid_size != num_patch_tokens:
        raise ValueError(
            f"Cannot infer square grid from position_embeddings shape: {tuple(pos_embed.shape)}"
        )

    return int(grid_size * patch_size)


def resolve_model_sizes_from_checkpoint(
    checkpoint: Dict,
    state_dict: Dict[str, torch.Tensor],
    cli_input_size: Optional[int],
    cli_heatmap_size: Optional[int],
) -> Tuple[int, int]:
    inferred_input_size = infer_input_size_from_state_dict(state_dict, patch_size=14)
    meta_input_size = checkpoint.get("input_size", None)
    meta_heatmap_size = checkpoint.get("heatmap_size", None)

    if inferred_input_size is not None and meta_input_size is not None:
        if int(inferred_input_size) != int(meta_input_size):
            print(
                f"[tta][warn] checkpoint metadata input_size={meta_input_size}, "
                f"but inferred from position_embeddings={inferred_input_size}. "
                f"Using inferred value."
            )

    resolved_input_size = inferred_input_size
    if resolved_input_size is None:
        resolved_input_size = int(meta_input_size) if meta_input_size is not None else 384

    if cli_input_size is not None and int(cli_input_size) != int(resolved_input_size):
        raise ValueError(
            f"CLI --input_size={cli_input_size} does not match resolved checkpoint "
            f"input_size={resolved_input_size}."
        )

    resolved_heatmap_size = int(meta_heatmap_size) if meta_heatmap_size is not None else 96
    if cli_heatmap_size is not None:
        if meta_heatmap_size is not None and int(cli_heatmap_size) != int(meta_heatmap_size):
            raise ValueError(
                f"CLI --heatmap_size={cli_heatmap_size} does not match checkpoint "
                f"heatmap_size={meta_heatmap_size}."
            )
        resolved_heatmap_size = int(cli_heatmap_size)

    return int(resolved_input_size), int(resolved_heatmap_size)


def load_source_model(
    checkpoint_path: str,
    device: torch.device,
    cli_input_size: Optional[int] = None,
    cli_heatmap_size: Optional[int] = None,
) -> Tuple[DinoHeatmapPoseModel, int, int, Dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "student_state_dict" in checkpoint:
        state_dict = checkpoint["student_state_dict"]
    else:
        state_dict = checkpoint

    resolved_input_size, resolved_heatmap_size = resolve_model_sizes_from_checkpoint(
        checkpoint=checkpoint,
        state_dict=state_dict,
        cli_input_size=cli_input_size,
        cli_heatmap_size=cli_heatmap_size,
    )

    print(
        f"[tta] resolved input_size={resolved_input_size}, "
        f"heatmap_size={resolved_heatmap_size}"
    )

    model = DinoHeatmapPoseModel(
        input_size=resolved_input_size,
        output_heatmap_size=resolved_heatmap_size,
        num_keypoints=11,
        mid_channels=int(checkpoint.get("mid_channels", 256)),
        num_deconv_layers=int(checkpoint.get("num_deconv_layers", 2)),
        pretrained=False,
    )
    model.load_state_dict(state_dict, strict=True)
    return model.to(device), resolved_input_size, resolved_heatmap_size, checkpoint


# =========================
# Geometry helpers
# =========================
def load_annotations(data_root: Path, split: str) -> Tuple[List[Dict], Path]:
    if split in {"sunlamp", "sunlamp_test"}:
        annotation_path = data_root / "sunlamp" / "test.json"
        image_dir = data_root / "sunlamp" / "images"
    elif split in {"lightbox", "lightbox_test"}:
        annotation_path = data_root / "lightbox" / "test.json"
        image_dir = data_root / "lightbox" / "images"
    elif split == "validation":
        annotation_path = data_root / "synthetic" / "validation.json"
        image_dir = data_root / "synthetic" / "images"
    else:
        raise ValueError(f"Unsupported split: {split}")

    with annotation_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    annotations = data if isinstance(data, list) else data["images"]
    return annotations, image_dir


class TargetTTADataset(Dataset):
    def __init__(self, data_root: str, split: str, input_size: int):
        self.data_root = Path(data_root)
        self.split = split
        self.input_size = int(input_size)
        self.annotations, self.image_dir = load_annotations(self.data_root, split)
        self.camera_matrix, self.dist_coeffs = load_camera(self.data_root)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> Dict[str, object]:
        ann = self.annotations[index]
        image_path = self.image_dir / ann["filename"]
        image = np.array(Image.open(image_path).convert("RGB"))

        gt_quaternion = np.asarray(ann["q_vbs2tango_true"], dtype=np.float32)
        gt_translation = np.asarray(ann["r_Vo2To_vbs_true"], dtype=np.float32)

        gt_keypoints_2d = project_keypoints(
            SPEEDPLUS_3D_KEYPOINTS,
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
            expand_ratio=1.25,
        )
        crop_image, _ = crop_and_resize(
            image,
            gt_keypoints_2d,
            bbox,
            output_size=self.input_size,
        )
        image_tensor = normalize_image(crop_image)

        return {
            "image": image_tensor,
            "bbox": torch.tensor(bbox, dtype=torch.float32),
            "camera_matrix": torch.tensor(self.camera_matrix, dtype=torch.float32),
            "dist_coeffs": torch.tensor(self.dist_coeffs, dtype=torch.float32),
            "filename": ann["filename"],
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
        pooled = F.max_pool2d(
            heatmap,
            kernel_size=nms_kernel,
            stride=1,
            padding=nms_kernel // 2,
        )
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
    object_points = SPEEDPLUS_3D_KEYPOINTS.astype(np.float64)
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


# =========================
# OTTA helpers
# =========================
def random_discontinuous_mask(
    images: torch.Tensor,
    mask_ratio: float = 0.8,
    patch_size: int = 14,
) -> torch.Tensor:
    batch_size, _, height, width = images.shape
    grid_h = height // patch_size
    grid_w = width // patch_size
    num_patches = grid_h * grid_w
    num_mask = max(1, int(num_patches * mask_ratio))

    patch_mask = torch.ones(batch_size, grid_h, grid_w, device=images.device)
    for batch_idx in range(batch_size):
        chosen = torch.randperm(num_patches, device=images.device)[:num_mask]
        patch_mask[batch_idx].view(-1)[chosen] = 0.0

    mask = (
        patch_mask.repeat_interleave(patch_size, dim=1)
        .repeat_interleave(patch_size, dim=2)
        .unsqueeze(1)
    )
    return images * mask


@torch.no_grad()
def update_ema_teacher(student_model: nn.Module, teacher_model: nn.Module, alpha: float = 0.999) -> None:
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1.0 - alpha)


def set_batchnorm_eval(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            m.eval()


def configure_adaptation_modules(model: nn.Module, adapt_modules: str = "decoder") -> List[nn.Parameter]:
    trainable_params: List[nn.Parameter] = []

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
def build_source_prototype(model: nn.Module, dataloader: DataLoader, device: torch.device) -> torch.Tensor:
    model.eval()
    features = []
    for batch in tqdm(dataloader, desc="build_prototype", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        with autocast(device_type="cuda", enabled=device.type == "cuda"):
            _, cls_token = model(images, return_features=True)
        features.append(cls_token.detach().cpu())
    prototype = torch.cat(features, dim=0).mean(dim=0)
    return torch.cat([prototype, prototype], dim=0)


@torch.no_grad()
def initialize_memory_bank(
    dataloader: DataLoader,
    memory_bank: DynamicMemoryBank,
    max_samples: int = 16,
) -> None:
    dataset = getattr(dataloader, "dataset", None)
    if dataset is None:
        raise ValueError("Memory-bank initialization requires access to the source dataset.")

    num_samples = min(max_samples, len(dataset))
    sampled_indices = np.random.choice(len(dataset), size=num_samples, replace=False)
    for sample_index in tqdm(sampled_indices, desc="init_memory", leave=False):
        sample = dataset[int(sample_index)]
        memory_bank.push(sample["image"], sample["heatmap"])


def make_source_dataset(
    args,
    split: str,
    resolved_input_size: int,
    resolved_heatmap_size: int,
    use_source_augmentation: bool = False,
) -> SpeedPlusDinoHeatmapDataset:
    return SpeedPlusDinoHeatmapDataset(
        data_root=args.data_root,
        split=split,
        input_size=resolved_input_size,
        heatmap_size=resolved_heatmap_size,
        heatmap_sigma=args.heatmap_sigma,
        use_source_augmentation=use_source_augmentation,
    )


def teacher_heatmap_pass(model: nn.Module, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        with autocast(device_type="cuda", enabled=device.type == "cuda"):
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
    loss_history: Dict[str, List[float]],
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


def online_adapt(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    student_model, resolved_input_size, resolved_heatmap_size, checkpoint = load_source_model(
        checkpoint_path=args.source_checkpoint,
        device=device,
        cli_input_size=args.input_size,
        cli_heatmap_size=args.heatmap_size,
    )
    teacher_model = copy.deepcopy(student_model)

    source_dataset = make_source_dataset(
        args,
        "train",
        resolved_input_size=resolved_input_size,
        resolved_heatmap_size=resolved_heatmap_size,
        use_source_augmentation=False,
    )
    target_dataset = TargetTTADataset(
        args.data_root,
        args.target_split,
        resolved_input_size,
    )

    source_loader = DataLoader(
        source_dataset,
        batch_size=args.prototype_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

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
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

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

    for step, batch in enumerate(tqdm(target_loader, desc="online_tta_dinov2_geom"), start=1):
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
            input_size=resolved_input_size,
            heatmap_size=resolved_heatmap_size,
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
            memory_bank,
            args.memory_sample_size,
            device,
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

        with autocast(device_type="cuda", enabled=device.type == "cuda"):
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
            "input_size": resolved_input_size,
            "heatmap_size": resolved_heatmap_size,
            "heatmap_sigma": args.heatmap_sigma,
            "mid_channels": args.mid_channels,
            "num_deconv_layers": args.num_deconv_layers,
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
            "source_checkpoint": args.source_checkpoint,
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
        "lr": args.lr,
        "adapt_modules": args.adapt_modules,
        "lambda_st": args.lambda_st,
        "lambda_ca": args.lambda_ca,
        "ema_alpha": args.ema_alpha,
        "bank_push_conf_thresh": args.bank_push_conf_thresh,
        "memory_conf_thresh": args.memory_conf_thresh,
        "min_reliable_samples": args.min_reliable_samples,
        "gate_min_confidence_mean": args.gate_min_confidence_mean,
        "gate_min_inliers": args.gate_min_inliers,
        "gate_max_reproj_error": args.gate_max_reproj_error,
        "resolved_input_size": resolved_input_size,
        "resolved_heatmap_size": resolved_heatmap_size,
        "skipped_geom_gate": skipped_geom_gate,
        "skipped_bank_conf": skipped_bank_conf,
        "skipped_unreliable_batch": skipped_unreliable_batch,
        "num_optimization_steps": len(loss_history["step"]),
    }
    with (output_dir / "adapt_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="speedplusv2")
    parser.add_argument("--source_checkpoint", type=str, required=True)
    parser.add_argument("--target_split", type=str, default="lightbox")
    parser.add_argument("--output_dir", type=str, default="output/dinov2_heatmap_tta_geom")

    parser.add_argument("--input_size", type=int, default=None)
    parser.add_argument("--heatmap_size", type=int, default=None)
    parser.add_argument("--heatmap_sigma", type=float, default=3.0)
    parser.add_argument("--mid_channels", type=int, default=256)
    parser.add_argument("--num_deconv_layers", type=int, default=2)

    parser.add_argument(
        "--adapt_modules",
        type=str,
        default="decoder",
        choices=["decoder", "decoder_norm", "all"],
    )
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--memory_capacity", type=int, default=16)
    parser.add_argument("--memory_sample_size", type=int, default=16)
    parser.add_argument("--bank_push_conf_thresh", type=float, default=0.20)
    parser.add_argument("--memory_conf_thresh", type=float, default=0.20)
    parser.add_argument("--min_reliable_samples", type=int, default=4)

    parser.add_argument("--mask_ratio", type=float, default=0.8)
    parser.add_argument("--patch_size", type=int, default=14)
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