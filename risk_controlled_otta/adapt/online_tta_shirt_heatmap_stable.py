from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
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
        object_keypoints_3d: np.ndarray | None = None,
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
        self.object_keypoints_3d = np.asarray(
            SHIRT_KEYPOINTS_3D if object_keypoints_3d is None else object_keypoints_3d,
            dtype=np.float32,
        )

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

    def __getitem__(self, index: int) -> Dict[str, Any]:
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
            "quaternion": torch.tensor(quaternion, dtype=torch.float32),
            "translation": torch.tensor(translation, dtype=torch.float32),
        }


def random_discontinuous_mask(
    images: torch.Tensor,
    mask_ratio: float = 0.8,
    patch_size: int = 16,
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
        patch_mask
        .repeat_interleave(patch_size, dim=1)
        .repeat_interleave(patch_size, dim=2)
        .unsqueeze(1)
    )
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


def make_dataset(args, split: str, use_source_augmentation: bool = False) -> ShirtDinoHeatmapDataset:
    if args.roe == "all":
        roes = ["roe1", "roe2"]
    else:
        roes = [args.roe]

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
    plt.plot(steps[sampled], weighted_st[sampled], label=r"$10 \times L_{st}$", linewidth=1.5)
    plt.plot(steps[sampled], raw_mh[sampled], label=r"$L_{mh}$", linewidth=1.5)
    plt.plot(steps[sampled], weighted_ca[sampled], label=r"$0.01 \times L_{ca}$", linewidth=1.5)
    plt.xlabel("Training Step")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "otta_losses_weighted.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(steps[sampled], total_loss[sampled], label="total loss", linewidth=1.8)
    plt.plot(steps[sampled], raw_st[sampled], label=r"$L_{st}$", linewidth=1.5)
    plt.plot(steps[sampled], raw_mh[sampled], label=r"$L_{mh}$", linewidth=1.5)
    plt.plot(steps[sampled], raw_ca[sampled], label=r"$L_{ca}$", linewidth=1.5)
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

    source_dataset = make_dataset(args, args.source_split, use_source_augmentation=False)
    target_dataset = make_dataset(args, args.target_split, use_source_augmentation=False)

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
        "online_conf": [],
    }

    skipped_lowconf_push = 0
    skipped_unreliable_batch = 0

    for step, batch in enumerate(tqdm(target_loader, desc="online_tta_shirt_stable"), start=1):
        x_online = batch["image"].to(device, non_blocking=True)

        with torch.no_grad():
            teacher_online_heatmap = teacher_model(x_online)
            online_scores = heatmap_confidence_score(teacher_online_heatmap)
            best_index = int(online_scores.argmax().item())
            best_conf = float(online_scores[best_index].item())

            if best_conf >= args.bank_push_conf_thresh:
                memory_bank.push(x_online[best_index], teacher_online_heatmap[best_index])
            else:
                skipped_lowconf_push += 1

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
        loss_history["online_conf"].append(best_conf)

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
            "ema_alpha": args.ema_alpha,
            "bank_push_conf_thresh": args.bank_push_conf_thresh,
            "memory_conf_thresh": args.memory_conf_thresh,
            "min_reliable_samples": args.min_reliable_samples,
            "skipped_lowconf_push": skipped_lowconf_push,
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
        "ema_alpha": args.ema_alpha,
        "bank_push_conf_thresh": args.bank_push_conf_thresh,
        "memory_conf_thresh": args.memory_conf_thresh,
        "min_reliable_samples": args.min_reliable_samples,
        "skipped_lowconf_push": skipped_lowconf_push,
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
    parser.add_argument("--output_dir", type=str, default="output/dinov3_heatmap_shirt_tta_stable")

    parser.add_argument("--model_name", type=str, default="vit_base_patch16_dinov3.lvd1689m")
    parser.add_argument("--input_size", type=int, default=384)
    parser.add_argument("--heatmap_size", type=int, default=96)
    parser.add_argument("--heatmap_sigma", type=float, default=3.0)
    parser.add_argument("--expand_ratio", type=float, default=1.25)
    parser.add_argument("--mid_channels", type=int, default=256)
    parser.add_argument("--num_deconv_layers", type=int, default=2)

    parser.add_argument("--adapt_modules", type=str, default="decoder", choices=["decoder", "decoder_norm", "all"])
    parser.add_argument("--lr", type=float, default=2e-6)
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

    parser.add_argument("--lambda_st", type=float, default=10.0)
    parser.add_argument("--lambda_ca", type=float, default=0.01)

    parser.add_argument("--prototype_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--plot_stride", type=int, default=25)
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    online_adapt(parse_args())