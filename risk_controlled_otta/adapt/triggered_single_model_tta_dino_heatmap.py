from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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

from data.crop_and_heatmap import (
    SPEEDPLUS_3D_KEYPOINTS,
    load_camera,
)
from risk_controlled_otta.data.dino_heatmap_dataset import (
    SpeedPlusDinoHeatmapDataset,
    generate_gaussian_heatmap_from_crop_coords,
)
from risk_controlled_otta.eval.evaluate_dino_heatmap import (
    decode_heatmap_to_keypoints,
    map_crop_coords_to_image,
    solve_pose_robust,
)
from risk_controlled_otta.models.dino_pose_model import DinoHeatmapPoseModel


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


def load_source_model(checkpoint_path: str, device: torch.device) -> DinoHeatmapPoseModel:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = DinoHeatmapPoseModel(
        model_name=checkpoint.get("model_name", "vit_base_patch16_dinov3.lvd1689m"),
        input_size=int(checkpoint.get("input_size", 384)),
        num_keypoints=11,
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


def make_dataset(args, split: str) -> SpeedPlusDinoHeatmapDataset:
    return SpeedPlusDinoHeatmapDataset(
        data_root=args.data_root,
        split=split,
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        heatmap_sigma=args.heatmap_sigma,
        use_source_augmentation=False,
    )


def configure_trainable_parameters(model: DinoHeatmapPoseModel, update_scope: str) -> List[torch.nn.Parameter]:
    for param in model.parameters():
        param.requires_grad = False

    for param in model.decoder.parameters():
        param.requires_grad = True

    if update_scope == "decoder_last_block":
        blocks = getattr(model.encoder, "blocks", None)
        if blocks is None or len(blocks) == 0:
            raise ValueError("update_scope=decoder_last_block requires encoder.blocks.")
        for param in blocks[-1].parameters():
            param.requires_grad = True
    elif update_scope != "decoder":
        raise ValueError(f"Unsupported update_scope: {update_scope}")

    return [param for param in model.parameters() if param.requires_grad]


def heatmap_confidence_stats(heatmap: torch.Tensor) -> Tuple[float, float]:
    peaks = heatmap.detach().reshape(heatmap.shape[0], heatmap.shape[1], -1).amax(dim=-1)
    return float(peaks.mean().item()), float(peaks.min().item())


def tensor_image_name(value) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0])
    return str(value)


def tensor_bbox_to_tuple(bbox: torch.Tensor) -> Tuple[float, float, float, float]:
    bbox_np = bbox.detach().cpu().numpy()
    if bbox_np.ndim == 2:
        bbox_np = bbox_np[0]
    return tuple(float(item) for item in bbox_np.tolist())


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

    pose_debug: Dict[str, object]
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
        pose_failed = False
    except Exception as exc:
        rvec = None
        tvec = None
        pose_debug = {
            "selected_indices": [],
            "ransac_inliers": [],
            "used_fallback_epnp": True,
            "mean_reprojection_error": float("inf"),
            "solvepnp_error": str(exc),
        }
        pose_failed = True

    mean_conf, min_conf = heatmap_confidence_stats(heatmap)
    num_selected = int(pose_debug.get("num_selected_points", len(pose_debug.get("selected_indices", []))))
    num_inliers = int(len(pose_debug.get("ransac_inliers", [])))
    inlier_ratio = float(num_inliers / max(num_selected, 1))
    reproj_error = float(pose_debug.get("mean_reprojection_error", float("inf")))
    fallback = bool(pose_debug.get("used_fallback_epnp", False))

    trigger_reasons = []
    if mean_conf < args.trigger_confidence:
        trigger_reasons.append("low_confidence")
    if num_inliers < args.trigger_min_inliers:
        trigger_reasons.append("low_inliers")
    if reproj_error > args.trigger_reprojection_error:
        trigger_reasons.append("high_reprojection_error")
    if fallback:
        trigger_reasons.append("fallback_epnp")
    if pose_failed:
        trigger_reasons.append("pnp_failed")

    reproj_quality = 1.0 / (1.0 + min(reproj_error, args.quality_reprojection_cap))
    quality = float(max(mean_conf, 0.0) * max(inlier_ratio, 0.0) * reproj_quality)

    return {
        "rvec": rvec,
        "tvec": tvec,
        "crop_coords": crop_coords,
        "image_coords": image_coords,
        "confidences": confidences[0],
        "mean_confidence": mean_conf,
        "min_confidence": min_conf,
        "num_selected_points": num_selected,
        "num_ransac_inliers": num_inliers,
        "inlier_ratio": inlier_ratio,
        "mean_reprojection_error": reproj_error,
        "used_fallback_epnp": fallback,
        "quality": quality,
        "triggered": bool(trigger_reasons),
        "trigger_reasons": trigger_reasons,
        **pose_debug,
    }


def geometry_target_from_pose(
    rvec,
    tvec,
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
        objectPoints=SPEEDPLUS_3D_KEYPOINTS.astype(np.float64),
        rvec=np.asarray(rvec, dtype=np.float64),
        tvec=np.asarray(tvec, dtype=np.float64),
        cameraMatrix=camera_matrix.astype(np.float64),
        distCoeffs=dist_coeffs.astype(np.float64),
    )
    image_coords = projected.reshape(-1, 2).astype(np.float64)
    x1, y1, x2, y2 = bbox
    crop_coords = image_coords.copy()
    crop_coords[:, 0] = (crop_coords[:, 0] - x1) / max(x2 - x1, 1.0) * float(input_size)
    crop_coords[:, 1] = (crop_coords[:, 1] - y1) / max(y2 - y1, 1.0) * float(input_size)
    visible = (
        (crop_coords[:, 0] >= 0.0)
        & (crop_coords[:, 0] < float(input_size))
        & (crop_coords[:, 1] >= 0.0)
        & (crop_coords[:, 1] < float(input_size))
    ).astype(np.float32)
    if visible.sum() < 4:
        return None

    target = generate_gaussian_heatmap_from_crop_coords(
        keypoints=crop_coords.astype(np.float32),
        visible=visible,
        input_size=input_size,
        heatmap_size=heatmap_size,
        sigma=heatmap_sigma,
    )
    return target.unsqueeze(0).to(device)


def weighted_heatmap_mse(pred_heatmap: torch.Tensor, target_heatmap: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    per_sample = (pred_heatmap - target_heatmap).pow(2).mean(dim=(1, 2, 3))
    if weights is None:
        return per_sample.mean()
    return (per_sample * weights).mean()


def confidence_weighted_regularization(pred_heatmap: torch.Tensor, pseudo_heatmap: torch.Tensor, tau: float) -> torch.Tensor:
    peaks = pseudo_heatmap.detach().flatten(2).amax(dim=-1)
    weights = F.softmax(peaks / tau, dim=1).unsqueeze(-1).unsqueeze(-1)
    per_keypoint = (pred_heatmap - pseudo_heatmap.detach()).pow(2)
    return (weights * per_keypoint).mean()


@torch.no_grad()
def maybe_push_current_sample(
    memory_bank: QualityMemoryBank,
    image: torch.Tensor,
    heatmap: torch.Tensor,
    image_name: str,
    quality: float,
    args,
) -> bool:
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

    total_loss_value = 0.0
    loss_st_value = 0.0
    loss_geo_value = 0.0
    loss_reg_value = 0.0
    executed_step = False

    model.train()
    for _ in range(args.adapt_steps):
        optimizer.zero_grad(set_to_none=True)
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

            total_loss = args.lambda_self_training * loss_st + args.lambda_geo * loss_geo + args.lambda_reg * loss_reg

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group["params"]], args.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        executed_step = True

        total_loss_value = float(total_loss.item())
        loss_st_value = float(loss_st.item())
        loss_geo_value = float(loss_geo.item())
        loss_reg_value = float(loss_reg.item())

    with torch.no_grad():
        updated_mem_pred = model(mem_images)
        updated_quality = updated_mem_pred.detach().flatten(2).amax(dim=-1).mean(dim=1)
        if len(mem_indices) > 0:
            memory_bank.update_if_better(mem_indices, updated_mem_pred, updated_quality)

    return {
        "executed_step": executed_step,
        "total_loss": total_loss_value,
        "loss_self_training": loss_st_value,
        "loss_geometry": loss_geo_value,
        "loss_regularization": loss_reg_value,
    }


def run_triggered_single_model_tta(args) -> None:
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
    trainable_params = configure_trainable_parameters(model, args.update_scope)
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=device.type == "cuda")
    memory_bank = QualityMemoryBank(capacity=args.memory_capacity)

    history: List[Dict[str, object]] = []
    loss_history: List[Dict[str, object]] = []

    for step, batch in enumerate(tqdm(target_loader, desc="triggered_single_model_tta"), start=1):
        if args.max_samples is not None and step > args.max_samples:
            break

        image = batch["image"].to(device, non_blocking=True)
        bbox = tensor_bbox_to_tuple(batch["bbox"])
        image_name = tensor_image_name(batch["image_name"])

        model.eval()
        with torch.no_grad(), autocast(enabled=device.type == "cuda"):
            heatmap = model(image)

        diagnosis = diagnose_prediction(
            heatmap=heatmap,
            bbox=bbox,
            input_size=args.input_size,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            args=args,
        )

        pushed_to_memory = maybe_push_current_sample(
            memory_bank=memory_bank,
            image=image,
            heatmap=heatmap,
            image_name=image_name,
            quality=float(diagnosis["quality"]),
            args=args,
        )

        adapted = False
        losses = {
            "total_loss": 0.0,
            "loss_self_training": 0.0,
            "loss_geometry": 0.0,
            "loss_regularization": 0.0,
        }
        if bool(diagnosis["triggered"]) and len(memory_bank) >= args.min_memory_for_update:
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
            losses = adapt_single_trigger(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                memory_bank=memory_bank,
                current_image=image,
                current_pseudo=heatmap.detach(),
                geometry_target=geometry_target,
                args=args,
                device=device,
            )
            adapted = bool(losses.pop("executed_step", False))
            loss_history.append({"step": step, "image_name": image_name, **losses})

        history.append(
            {
                "step": step,
                "image_name": image_name,
                "triggered": bool(diagnosis["triggered"]),
                "trigger_reasons": diagnosis["trigger_reasons"],
                "optimizer_step_executed": adapted,
                "adapted": adapted,
                "pushed_to_memory": pushed_to_memory,
                "memory_size": len(memory_bank),
                "quality": float(diagnosis["quality"]),
                "mean_confidence": float(diagnosis["mean_confidence"]),
                "min_confidence": float(diagnosis["min_confidence"]),
                "num_ransac_inliers": int(diagnosis["num_ransac_inliers"]),
                "inlier_ratio": float(diagnosis["inlier_ratio"]),
                "mean_reprojection_error": float(diagnosis["mean_reprojection_error"]),
                "used_fallback_epnp": bool(diagnosis["used_fallback_epnp"]),
                **losses,
            }
        )

    summary = {
        "target_split": args.target_split,
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

    with (output_dir / "trigger_history.json").open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "history": history, "loss_history": loss_history}, handle, indent=2)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": args.model_name,
            "input_size": args.input_size,
            "heatmap_size": args.heatmap_size,
            "heatmap_sigma": args.heatmap_sigma,
            "mid_channels": args.mid_channels,
            "num_deconv_layers": args.num_deconv_layers,
            "adaptation": "triggered_single_model_tta",
            "update_scope": args.update_scope,
            "summary": summary,
        },
        output_dir / "tta_final.pth",
    )
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="speedplusv2")
    parser.add_argument("--source_checkpoint", type=str, required=True)
    parser.add_argument("--target_split", type=str, default="sunlamp")
    parser.add_argument("--output_dir", type=str, default="output/dinov3_heatmap_triggered_single_model_tta")
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_dinov3.lvd1689m")
    parser.add_argument("--input_size", type=int, default=384)
    parser.add_argument("--heatmap_size", type=int, default=96)
    parser.add_argument("--heatmap_sigma", type=float, default=3.0)
    parser.add_argument("--mid_channels", type=int, default=256)
    parser.add_argument("--num_deconv_layers", type=int, default=2)
    parser.add_argument("--update_scope", type=str, choices=["decoder", "decoder_last_block"], default="decoder")

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


