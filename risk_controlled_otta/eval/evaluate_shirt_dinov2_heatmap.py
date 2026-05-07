from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPO_ROOT / "output"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(OUTPUT_ROOT) not in sys.path:
    sys.path.insert(0, str(OUTPUT_ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from data.crop_and_heatmap import (
    compute_expanded_bbox,
    crop_and_resize,
    normalize_image,
    project_keypoints,
    visible_keypoints_mask,
)
from dinov2_heatmap_otta.models.dinov2_pose_model import DinoHeatmapPoseModel

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

    return camera_matrix.astype(np.float64), dist_coeffs.reshape(-1).astype(np.float64)


def parse_shirt_annotation_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "filename": entry["filename"],
        "quaternion": np.asarray(entry["q_vbs2tango_true"], dtype=np.float64),
        "translation": np.asarray(entry["r_Vo2To_vbs_true"], dtype=np.float64),
    }


def load_annotations(
    data_root: Path,
    roe: str,
    domain: str,
    val_ratio: float = 0.1,
    seed: int = 42,
    split: str = "val",
) -> List[Dict[str, Any]]:
    if roe == "all":
        roes = ["roe1", "roe2"]
    else:
        roes = [roe]

    records: List[Dict[str, Any]] = []
    for roe_name in roes:
        roe_dir = data_root / roe_name
        ann_path = roe_dir / f"{roe_name}.json"
        image_dir = roe_dir / domain / "images"

        raw = load_json(ann_path)
        if not isinstance(raw, list):
            raise ValueError(f"Expected list annotation file: {ann_path}")

        for item in raw:
            parsed = parse_shirt_annotation_entry(item)
            parsed["roe"] = roe_name
            parsed["image_path"] = image_dir / parsed["filename"]
            parsed["domain"] = domain
            records.append(parsed)

    rng = np.random.RandomState(seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)

    n_val = max(1, int(len(indices) * val_ratio))
    val_idx = set(indices[:n_val].tolist())
    train_idx = set(indices[n_val:].tolist())

    if split == "val":
        selected = [records[i] for i in range(len(records)) if i in val_idx]
    elif split == "train":
        selected = [records[i] for i in range(len(records)) if i in train_idx]
    elif split == "all":
        selected = records
    else:
        raise ValueError(f"Unsupported split: {split}, expected one of ['train', 'val', 'all']")

    return selected


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


def resolve_model_sizes(
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
                f"[eval][warn] checkpoint metadata input_size={meta_input_size}, "
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


def load_checkpoint_model(
    model_path: str,
    device: torch.device,
    cli_input_size: Optional[int] = None,
    cli_heatmap_size: Optional[int] = None,
) -> Tuple[DinoHeatmapPoseModel, int, int]:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "student_state_dict" in checkpoint:
        state_dict = checkpoint["student_state_dict"]
    else:
        state_dict = checkpoint

    resolved_input_size, resolved_heatmap_size = resolve_model_sizes(
        checkpoint=checkpoint,
        state_dict=state_dict,
        cli_input_size=cli_input_size,
        cli_heatmap_size=cli_heatmap_size,
    )

    model = DinoHeatmapPoseModel(
        input_size=resolved_input_size,
        output_heatmap_size=resolved_heatmap_size,
        num_keypoints=int(checkpoint.get("num_keypoints", 11)),
        mid_channels=int(checkpoint.get("mid_channels", 256)),
        num_deconv_layers=int(checkpoint.get("num_deconv_layers", 2)),
        pretrained=False,
        pretrained_path=None,
    ).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, resolved_input_size, resolved_heatmap_size


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
        cv2.solvePnP(
            objectPoints=inlier_obj,
            imagePoints=inlier_img,
            cameraMatrix=camera_matrix.astype(np.float64),
            distCoeffs=dist_coeffs.astype(np.float64),
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

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


def rotation_vector_to_quaternion(rotation_vector: np.ndarray) -> np.ndarray:
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    trace = np.trace(rotation_matrix)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * s
        y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * s
        z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * s
    elif rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2])
        w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
        x = 0.25 * s
        y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
        z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
    elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2])
        w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
        x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
        y = 0.25 * s
        z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1])
        w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
        x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
        y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def compute_metrics(
    pred_quaternion: np.ndarray,
    pred_translation: np.ndarray,
    gt_quaternion: np.ndarray,
    gt_translation: np.ndarray,
) -> Dict[str, float]:
    pred_quaternion = pred_quaternion / (np.linalg.norm(pred_quaternion) + 1e-12)
    gt_quaternion = gt_quaternion / (np.linalg.norm(gt_quaternion) + 1e-12)

    et = float(np.linalg.norm(pred_translation - gt_translation))
    dot = float(np.clip(np.abs(np.dot(pred_quaternion, gt_quaternion)), -1.0, 1.0))
    eq = float(2.0 * np.arccos(dot))
    eq_deg = float(np.rad2deg(eq))
    gt_t_norm = float(np.linalg.norm(gt_translation))
    e_t_bar = float(et / max(gt_t_norm, 1e-12))
    ep = float(eq + e_t_bar)

    theta_q = 0.169 * np.pi / 180.0
    theta_t = 2.173e-3
    if eq < theta_q and e_t_bar < theta_t:
        e_star_t = 0.0
        e_star_t_bar = 0.0
        e_star_q = 0.0
        e_star_p = 0.0
    else:
        e_star_t = et
        e_star_t_bar = e_t_bar
        e_star_q = eq
        e_star_p = ep

    return {
        "et": et,
        "eq": eq,
        "eq_deg": eq_deg,
        "e_t_bar": e_t_bar,
        "ep": ep,
        "e_star_t": e_star_t,
        "e_star_t_bar": e_star_t_bar,
        "e_star_q": e_star_q,
        "e_star_p": e_star_p,
    }


def draw_keypoints(
    image: np.ndarray,
    pred_keypoints: np.ndarray,
    confidences: np.ndarray,
    gt_keypoints: Optional[np.ndarray] = None,
) -> np.ndarray:
    vis = image.copy()
    if gt_keypoints is not None:
        for index, point in enumerate(gt_keypoints):
            x, y = int(round(point[0])), int(round(point[1]))
            cv2.circle(vis, (x, y), 5, (255, 0, 0), 1)
            cv2.putText(vis, f"g{index}", (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    for index, (point, conf) in enumerate(zip(pred_keypoints, confidences)):
        x, y = int(round(point[0])), int(round(point[1]))
        color = (0, 255, 0) if conf > 0.05 else (0, 0, 255)
        cv2.circle(vis, (x, y), 5, color, -1)
        cv2.putText(vis, f"p{index}", (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return vis


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, resolved_input_size, resolved_heatmap_size = load_checkpoint_model(
        args.model_path,
        device,
        cli_input_size=args.input_size,
        cli_heatmap_size=args.heatmap_size,
    )

    camera_matrix, dist_coeffs = load_camera_from_shirt(data_root)
    annotations = load_annotations(
        data_root=data_root,
        roe=args.roe,
        domain=args.domain,
        val_ratio=args.val_ratio,
        seed=args.seed,
        split=args.split,
    )

    results = []
    failures = 0

    for index, annotation in enumerate(tqdm(annotations, desc=f"eval_shirt_{args.split}_{args.roe}")):
        image_path = annotation["image_path"]
        image = np.array(Image.open(image_path).convert("RGB"))

        gt_quaternion = np.asarray(annotation["quaternion"], dtype=np.float64)
        gt_translation = np.asarray(annotation["translation"], dtype=np.float64)

        gt_keypoints_2d = project_keypoints(
            SHIRT_KEYPOINTS_3D,
            gt_quaternion.astype(np.float32),
            gt_translation.astype(np.float32),
            camera_matrix.astype(np.float32),
            dist_coeffs.astype(np.float32),
        )

        visible = visible_keypoints_mask(gt_keypoints_2d, (image.shape[1], image.shape[0]))
        bbox = compute_expanded_bbox(
            gt_keypoints_2d,
            visible,
            (image.shape[1], image.shape[0]),
            expand_ratio=args.expand_ratio,
        )
        crop_image, gt_crop_coords = crop_and_resize(
            image,
            gt_keypoints_2d,
            bbox,
            output_size=resolved_input_size,
        )

        image_tensor = normalize_image(crop_image).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_heatmap = model(image_tensor)

        crop_coords_hm, confidences = decode_heatmap_to_keypoints(
            pred_heatmap,
            apply_nms=not args.disable_nms,
            nms_kernel=args.nms_kernel,
            use_subpixel=not args.disable_subpixel,
            subpixel_radius=args.subpixel_radius,
        )

        crop_coords = crop_coords_hm[0].copy()
        crop_coords[:, 0] = crop_coords[:, 0] * resolved_input_size / resolved_heatmap_size
        crop_coords[:, 1] = crop_coords[:, 1] * resolved_input_size / resolved_heatmap_size
        image_coords = map_crop_coords_to_image(
            crop_coords,
            bbox,
            crop_size_w=resolved_input_size,
            crop_size_h=resolved_input_size,
        )

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

            pred_quaternion = rotation_vector_to_quaternion(rvec)
            pred_translation = tvec.reshape(-1)
            metrics = compute_metrics(
                pred_quaternion,
                pred_translation,
                gt_quaternion,
                gt_translation,
            )

            result = {
                "image_name": annotation["filename"],
                "roe": annotation["roe"],
                "domain": annotation["domain"],
                "quaternion_pred": pred_quaternion.tolist(),
                "translation_pred": pred_translation.tolist(),
                "confidences": confidences[0].tolist(),
                "success": True,
                **debug,
                **metrics,
            }
        except Exception as exc:
            failures += 1
            result = {
                "image_name": annotation["filename"],
                "roe": annotation["roe"],
                "domain": annotation["domain"],
                "confidences": confidences[0].tolist(),
                "success": False,
                "error": str(exc),
                "num_selected_points": 0,
                "ransac_inliers": [],
                "used_fallback_epnp": True,
                "mean_selected_confidence": float(np.mean(confidences[0])),
                "mean_reprojection_error": float("inf"),
                "max_reprojection_error": float("inf"),
                "tvec_norm": float("inf"),
                "et": float("inf"),
                "eq": float("inf"),
                "eq_deg": float("inf"),
                "e_t_bar": float("inf"),
                "ep": float("inf"),
                "e_star_t": float("inf"),
                "e_star_t_bar": float("inf"),
                "e_star_q": float("inf"),
                "e_star_p": float("inf"),
            }

        results.append(result)

        if index < args.num_vis:
            vis = draw_keypoints(image, image_coords, confidences[0], gt_keypoints=gt_keypoints_2d)
            vis_path = output_dir / f"vis_{annotation['roe']}_{annotation['filename']}"
            cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

            crop_vis = draw_keypoints(crop_image, crop_coords, confidences[0], gt_keypoints=gt_crop_coords)
            crop_vis_path = output_dir / f"crop_vis_{annotation['roe']}_{annotation['filename']}"
            cv2.imwrite(str(crop_vis_path), cv2.cvtColor(crop_vis, cv2.COLOR_RGB2BGR))

    valid_results = [item for item in results if item["success"]]
    if len(valid_results) == 0:
        raise RuntimeError("All evaluation samples failed during pose solving.")

    num_collapsed = int(sum(1 for item in valid_results if item["e_star_p"] > args.collapse_threshold))

    summary = {
        "dataset": "SHIRT",
        "split": args.split,
        "roe": args.roe,
        "domain": args.domain,
        "model_path": args.model_path,
        "resolved_input_size": resolved_input_size,
        "resolved_heatmap_size": resolved_heatmap_size,
        "num_samples": len(results),
        "num_success": len(valid_results),
        "num_failures": failures,
        "success_ratio": float(len(valid_results) / max(len(results), 1)),
        "collapse_threshold": float(args.collapse_threshold),
        "num_collapsed": num_collapsed,
        "collapse_rate": float(num_collapsed / max(len(valid_results), 1)),
        "avg_et": float(np.mean([item["et"] for item in valid_results])),
        "median_et": float(np.median([item["et"] for item in valid_results])),
        "p95_et": float(np.percentile([item["et"] for item in valid_results], 95)),
        "max_et": float(np.max([item["et"] for item in valid_results])),
        "avg_eq": float(np.mean([item["eq"] for item in valid_results])),
        "avg_eq_deg": float(np.mean([item["eq_deg"] for item in valid_results])),
        "median_eq_deg": float(np.median([item["eq_deg"] for item in valid_results])),
        "p95_eq_deg": float(np.percentile([item["eq_deg"] for item in valid_results], 95)),
        "avg_e_t_bar": float(np.mean([item["e_t_bar"] for item in valid_results])),
        "avg_ep": float(np.mean([item["ep"] for item in valid_results])),
        "avg_e_star_t": float(np.mean([item["e_star_t"] for item in valid_results])),
        "avg_e_star_t_bar": float(np.mean([item["e_star_t_bar"] for item in valid_results])),
        "avg_e_star_q": float(np.mean([item["e_star_q"] for item in valid_results])),
        "avg_e_star_q_deg": float(np.mean([item["e_star_q"] for item in valid_results]) * 180.0 / np.pi),
        "avg_e_star_p": float(np.mean([item["e_star_p"] for item in valid_results])),
        "avg_num_selected_points": float(np.mean([item["num_selected_points"] for item in valid_results])),
        "avg_num_ransac_inliers": float(np.mean([len(item["ransac_inliers"]) for item in valid_results])),
        "fallback_epnp_ratio": float(np.mean([1.0 if item["used_fallback_epnp"] else 0.0 for item in valid_results])),
        "avg_mean_selected_confidence": float(np.mean([item["mean_selected_confidence"] for item in valid_results])),
        "avg_mean_reprojection_error": float(np.mean([item["mean_reprojection_error"] for item in valid_results])),
        "median_mean_reprojection_error": float(np.median([item["mean_reprojection_error"] for item in valid_results])),
        "p95_mean_reprojection_error": float(np.percentile([item["mean_reprojection_error"] for item in valid_results], 95)),
        "avg_tvec_norm": float(np.mean([item["tvec_norm"] for item in valid_results])),
        "p95_tvec_norm": float(np.percentile([item["tvec_norm"] for item in valid_results], 95)),
        "expand_ratio": args.expand_ratio,
        "min_confidence": args.min_confidence,
        "top_k": args.top_k,
        "min_points": args.min_points,
        "ransac_reproj_error": args.ransac_reproj_error,
        "subpixel_radius": args.subpixel_radius,
    }

    with (output_dir / f"{args.split}_results.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with (output_dir / f"{args.split}_per_image_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="SHIRT_Dataset")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--roe", type=str, choices=["roe1", "roe2", "all"], default="all")
    parser.add_argument("--domain", type=str, default="synthetic")
    parser.add_argument("--split", type=str, choices=["train", "val", "all"], default="val")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="evaluation_results_shirt_dinov2_heatmap")

    parser.add_argument("--input_size", type=int, default=None)
    parser.add_argument("--heatmap_size", type=int, default=None)
    parser.add_argument("--expand_ratio", type=float, default=1.25)

    parser.add_argument("--num_vis", type=int, default=20)
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
    parser.add_argument("--collapse_threshold", type=float, default=0.1)

    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())