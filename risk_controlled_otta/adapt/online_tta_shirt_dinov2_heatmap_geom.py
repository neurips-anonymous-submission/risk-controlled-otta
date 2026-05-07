from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import deque
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

# =========================
# Model & Size Inference
# =========================
def infer_input_size_from_state_dict(state_dict: Dict[str, torch.Tensor], patch_size: int = 14) -> int | None:
    pos_embed = state_dict.get("encoder.embeddings.position_embeddings", None)
    if pos_embed is None:
        return None
    num_patch_tokens = int(pos_embed.shape[1]) - 1
    grid_size = int(round(num_patch_tokens ** 0.5))
    return int(grid_size * patch_size)

def resolve_model_sizes(checkpoint: Dict, state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    inferred_input_size = infer_input_size_from_state_dict(state_dict, patch_size=14)
    meta_input_size = checkpoint.get("input_size", 384)
    meta_heatmap_size = checkpoint.get("heatmap_size", 96)
    
    resolved_input_size = inferred_input_size if inferred_input_size is not None else meta_input_size
    return int(resolved_input_size), int(meta_heatmap_size)

def load_models(checkpoint_path: str, device: torch.device) -> Tuple[DinoHeatmapPoseModel, DinoHeatmapPoseModel, int, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("student_state_dict", checkpoint))
    
    input_size, heatmap_size = resolve_model_sizes(checkpoint, state_dict)
    print(f"[online_tta] Resolved input_size={input_size}, heatmap_size={heatmap_size}")

    student = DinoHeatmapPoseModel(
        input_size=input_size,
        output_heatmap_size=heatmap_size,
        num_keypoints=11,
        mid_channels=int(checkpoint.get("mid_channels", 256)),
        num_deconv_layers=int(checkpoint.get("num_deconv_layers", 2)),
        pretrained=False,
    ).to(device)
    student.load_state_dict(state_dict, strict=True)
    
    teacher = copy.deepcopy(student)
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()
    
    return student, teacher, input_size, heatmap_size

def update_ema_variables(model: nn.Module, ema_model: nn.Module, alpha: float) -> None:
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1.0 - alpha)

# =========================
# Dataset Helpers
# =========================
def load_camera_from_shirt(data_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    cam = json.loads((data_root / "camera.json").read_text(encoding="utf-8"))
    camera_matrix = np.asarray(cam.get("cameraMatrix", cam.get("K")), dtype=np.float64)
    dist_coeffs = np.asarray(cam.get("distCoeffs", np.zeros(5)), dtype=np.float64)
    return camera_matrix, dist_coeffs.reshape(-1)

class ShirtTtaDataset(Dataset):
    def __init__(self, data_root: str, roes: List[str], domain: str, input_size: int, expand_ratio: float = 1.25):
        self.data_root = Path(data_root)
        self.input_size = input_size
        self.expand_ratio = expand_ratio
        self.camera_matrix, self.dist_coeffs = load_camera_from_shirt(self.data_root)
        
        self.records = []
        for roe in roes:
            ann_path = self.data_root / roe / f"{roe}.json"
            image_dir = self.data_root / roe / domain / "images"
            raw = json.loads(ann_path.read_text(encoding="utf-8"))
            for item in raw:
                self.records.append({
                    "image_path": image_dir / item["filename"],
                    "quaternion": np.asarray(item["q_vbs2tango_true"], dtype=np.float64),
                    "translation": np.asarray(item["r_Vo2To_vbs_true"], dtype=np.float64),
                })

    def __len__(self) -> int: return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        image = np.array(Image.open(record["image_path"]).convert("RGB"))
        
        kp_2d = project_keypoints(
            SHIRT_KEYPOINTS_3D, record["quaternion"].astype(np.float32),
            record["translation"].astype(np.float32),
            self.camera_matrix.astype(np.float32), self.dist_coeffs.astype(np.float32)
        )
        visible = visible_keypoints_mask(kp_2d, (image.shape[1], image.shape[0]))
        bbox = compute_expanded_bbox(kp_2d, visible, (image.shape[1], image.shape[0]), self.expand_ratio)
        crop_img, _ = crop_and_resize(image, kp_2d, bbox, output_size=self.input_size)
        
        return {
            "image": normalize_image(crop_img),
            "bbox": torch.tensor(bbox, dtype=torch.float32)
        }

# =========================
# Geometric Gating & PnP
# =========================
def decode_heatmap_to_keypoints(heatmap: np.ndarray, input_size: int, heatmap_size: int) -> Tuple[np.ndarray, np.ndarray]:
    b, k, h, w = heatmap.shape
    flat = heatmap.reshape(b, k, -1)
    conf = flat.max(axis=-1)
    idx = flat.argmax(axis=-1)
    xs, ys = (idx % w).astype(np.float64), (idx // w).astype(np.float64)
    
    # Simple subpixel refinement
    for batch_idx in range(b):
        for k_idx in range(k):
            x0, y0 = int(xs[batch_idx, k_idx]), int(ys[batch_idx, k_idx])
            patch = heatmap[batch_idx, k_idx, max(0, y0-2):min(h, y0+3), max(0, x0-2):min(w, x0+3)]
            mass = patch.sum()
            if mass > 1e-6:
                grid_y, grid_x = np.mgrid[max(0, y0-2):min(h, y0+3), max(0, x0-2):min(w, x0+3)]
                xs[batch_idx, k_idx] = (grid_x * patch).sum() / mass
                ys[batch_idx, k_idx] = (grid_y * patch).sum() / mass

    coords = np.stack([xs, ys], axis=-1)
    coords = coords * (input_size / heatmap_size)
    return coords, conf

def evaluate_geometric_gate(
    heatmap: torch.Tensor, bbox: torch.Tensor,
    input_size: int, heatmap_size: int,
    camera_matrix: np.ndarray, dist_coeffs: np.ndarray, args
) -> bool:
    hm_np = heatmap.detach().cpu().numpy()
    crop_coords, confidences = decode_heatmap_to_keypoints(hm_np, input_size, heatmap_size)
    
    crop_coords = crop_coords[0]
    confs = confidences[0]
    bbox_np = bbox[0].cpu().numpy()
    
    # Map to original image
    x1, y1, x2, y2 = bbox_np
    img_coords = crop_coords.copy()
    img_coords[:, 0] = img_coords[:, 0] / input_size * max(x2 - x1, 1.0) + x1
    img_coords[:, 1] = img_coords[:, 1] / input_size * max(y2 - y1, 1.0) + y1
    
    valid_idx = np.where(confs >= args.pnp_min_confidence)[0]
    if len(valid_idx) < args.min_points:
        return False
    
    order = np.argsort(-confs[valid_idx])
    sel_idx = valid_idx[order[:args.top_k]]
    
    obj_pts = SHIRT_KEYPOINTS_3D[sel_idx].astype(np.float64)
    img_pts = img_coords[sel_idx].astype(np.float64)
    
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=obj_pts, imagePoints=img_pts,
        cameraMatrix=camera_matrix, distCoeffs=dist_coeffs,
        iterationsCount=args.ransac_iterations, reprojectionError=args.ransac_reproj_error,
        confidence=args.ransac_confidence, flags=cv2.SOLVEPNP_EPNP
    )
    
    if not success or inliers is None:
        return False
        
    projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
    reproj_errors = np.linalg.norm(projected.reshape(-1, 2) - img_pts, axis=1)
    
    mean_conf = float(confs.mean())
    num_inliers = len(inliers)
    mean_reproj = float(reproj_errors.mean())
    
    passed = (
        mean_conf >= args.gate_min_confidence_mean and
        num_inliers >= args.gate_min_inliers and
        mean_reproj <= args.gate_max_reproj_error
    )
    return passed

# =========================
# Online Training Loop
# =========================
def run_online_tta(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    student, teacher, input_size, heatmap_size = load_models(args.source_checkpoint, device)
    
    # Only train decoder
    for param in student.parameters(): param.requires_grad = False
    for param in student.decoder.parameters(): param.requires_grad = True
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, student.parameters()), lr=args.lr)
    scaler = GradScaler(enabled=device.type == "cuda")
    
    camera_matrix, dist_coeffs = load_camera_from_shirt(Path(args.data_root))
    roes = ["roe1", "roe2"] if args.roe == "all" else [args.roe]
    dataset = ShirtTtaDataset(args.data_root, roes, args.domain, input_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    
    memory_bank = deque(maxlen=args.memory_capacity)
    
    for batch in tqdm(loader, desc="Online TTA Streaming"):
        images = batch["image"].to(device)
        bbox = batch["bbox"]
        
        # 1. Teacher generates pseudolabels
        with torch.no_grad():
            teacher_heatmap = teacher(images)
            
        # 2. Geometric Gating
        is_reliable = evaluate_geometric_gate(
            teacher_heatmap, bbox, input_size, heatmap_size,
            camera_matrix, dist_coeffs, args
        )
        
        if is_reliable:
            memory_bank.append((images[0].cpu(), teacher_heatmap[0].cpu()))
            
        # 3. Train Student
        if len(memory_bank) >= args.min_reliable_samples:
            sample_size = min(len(memory_bank), args.memory_sample_size)
            sampled = random.sample(memory_bank, sample_size)
            
            mem_imgs = torch.stack([item[0] for item in sampled]).to(device)
            mem_tgts = torch.stack([item[1] for item in sampled]).to(device)
            
            student.train()
            optimizer.zero_grad()
            with autocast(device_type="cuda", enabled=device.type == "cuda"):
                pred = student(mem_imgs)
                # Self-training Loss
                loss_st = F.mse_loss(pred, mem_tgts)
                
                # Simple Masked Consistency (Drop random 20% patches)
                mask = torch.rand_like(pred) > 0.2
                loss_ca = F.mse_loss(pred * mask, mem_tgts * mask)
                
                total_loss = args.lambda_st * loss_st + args.lambda_ca * loss_ca

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 4. EMA Update Teacher
            update_ema_variables(student, teacher, args.ema_alpha)

    # Save final model
    torch.save(
        {
            "model_state_dict": student.state_dict(),
            "input_size": input_size,
            "heatmap_size": heatmap_size,
            "mid_channels": 256,
            "num_deconv_layers": 2,
        },
        output_dir / "tta_final.pth"
    )
    print(f"[OK] OTTA complete. Checkpoint saved to {output_dir / 'tta_final.pth'}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="SHIRT_Dataset")
    parser.add_argument("--source_checkpoint", type=str, required=True)
    parser.add_argument("--roe", type=str, default="all")
    parser.add_argument("--domain", type=str, default="lightbox")
    parser.add_argument("--output_dir", type=str, default="output/dinov2_heatmap_online_tta_geom")
    
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--memory_capacity", type=int, default=32)
    parser.add_argument("--memory_sample_size", type=int, default=16)
    parser.add_argument("--min_reliable_samples", type=int, default=4)
    parser.add_argument("--ema_alpha", type=float, default=0.999)
    parser.add_argument("--lambda_st", type=float, default=1.0)
    parser.add_argument("--lambda_ca", type=float, default=0.01)
    
    # Geometric Gate Params
    parser.add_argument("--pnp_min_confidence", type=float, default=0.05)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--min_points", type=int, default=6)
    parser.add_argument("--ransac_reproj_error", type=float, default=6.0)
    parser.add_argument("--ransac_iterations", type=int, default=100)
    parser.add_argument("--ransac_confidence", type=float, default=0.999)
    
    parser.add_argument("--gate_min_confidence_mean", type=float, default=0.75)
    parser.add_argument("--gate_min_inliers", type=int, default=6)
    parser.add_argument("--gate_max_reproj_error", type=float, default=12.0)
    
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()

if __name__ == "__main__":
    run_online_tta(parse_args())