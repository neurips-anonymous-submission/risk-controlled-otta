from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from risk_controlled_otta.data.dino_heatmap_dataset import SpeedPlusDinoHeatmapDataset
from risk_controlled_otta.models.dino_pose_model import DinoHeatmapPoseModel
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
        num_keypoints=11,
        mid_channels=int(checkpoint.get("mid_channels", 256)),
        num_deconv_layers=int(checkpoint.get("num_deconv_layers", 2)),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device)


def make_dataset(args, split: str, use_source_augmentation: bool = False) -> SpeedPlusDinoHeatmapDataset:
    return SpeedPlusDinoHeatmapDataset(
        data_root=args.data_root,
        split=split,
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        heatmap_sigma=args.heatmap_sigma,
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
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_dataset = make_dataset(args, "train", use_source_augmentation=False)
    target_dataset = make_dataset(args, args.target_split, use_source_augmentation=False)
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

    student_model = load_source_model(args.source_checkpoint, device)
    source_prototype = build_source_prototype(student_model, source_loader, device)

    memory_bank = DynamicMemoryBank(capacity=args.memory_capacity)
    initialize_memory_bank(source_loader, memory_bank, max_samples=args.memory_capacity)

    teacher_model = copy.deepcopy(student_model)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    optimizer = AdamW(student_model.parameters(), lr=args.lr, weight_decay=0.0)
    scaler = GradScaler(enabled=device.type == "cuda")
    student_model.train()

    lambda_st = 10.0
    lambda_ca = 0.01
    loss_history = {"step": [], "total_loss": [], "loss_st": [], "loss_mh": [], "loss_ca": []}

    for step, batch in enumerate(tqdm(target_loader, desc="online_tta_dino"), start=1):
        x_online = batch["image"].to(device, non_blocking=True)

        with torch.no_grad():
            online_heatmap = student_model(x_online)
            online_scores = heatmap_confidence_score(online_heatmap)
            best_index = int(online_scores.argmax().item())
            memory_bank.push(x_online[best_index], online_heatmap[best_index])

        x_memory, old_pseudo, mem_indices = sample_from_memory_bank(memory_bank, args.memory_sample_size, device)
        masked_memory = random_discontinuous_mask(x_memory, mask_ratio=args.mask_ratio, patch_size=args.patch_size)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=device.type == "cuda"):
            h_student, cls_student = student_model(x_memory, return_features=True)
            with torch.no_grad():
                h_teacher, cls_teacher = teacher_model(masked_memory, return_features=True)

            loss_st = self_training_loss(h_student, old_pseudo)
            loss_mh = masked_heatmap_consistency_loss(h_student, h_teacher, tau=args.tau)
            loss_ca = class_awareness_consistency_loss(cls_student, cls_teacher, source_prototype.to(device))
            total_loss = total_target_loss(loss_st, loss_mh, loss_ca, lambda_st=lambda_st, lambda_ca=lambda_ca)

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_history["step"].append(step)
        loss_history["total_loss"].append(float(total_loss.item()))
        loss_history["loss_st"].append(float(loss_st.item()))
        loss_history["loss_mh"].append(float(loss_mh.item()))
        loss_history["loss_ca"].append(float(loss_ca.item()))

        with torch.no_grad():
            update_memory_pseudolabels(memory_bank, mem_indices, h_student, old_pseudo)
            update_ema_teacher(student_model, teacher_model, alpha=args.ema_alpha)

    torch.save(
        {
            "student_state_dict": student_model.state_dict(),
            "teacher_state_dict": teacher_model.state_dict(),
            "model_name": args.model_name,
            "input_size": args.input_size,
            "heatmap_size": args.heatmap_size,
            "heatmap_sigma": args.heatmap_sigma,
            "mid_channels": args.mid_channels,
            "num_deconv_layers": args.num_deconv_layers,
        },
        output_dir / "tta_final.pth",
    )
    save_loss_curves(loss_history, output_dir, lambda_st=lambda_st, lambda_ca=lambda_ca, plot_stride=args.plot_stride)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="speedplusv2")
    parser.add_argument("--source_checkpoint", type=str, required=True)
    parser.add_argument("--target_split", type=str, default="sunlamp")
    parser.add_argument("--output_dir", type=str, default="output/dinov3_heatmap_tta")
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_dinov3.lvd1689m")
    parser.add_argument("--input_size", type=int, default=384)
    parser.add_argument("--heatmap_size", type=int, default=96)
    parser.add_argument("--heatmap_sigma", type=float, default=3.0)
    parser.add_argument("--mid_channels", type=int, default=256)
    parser.add_argument("--num_deconv_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--memory_capacity", type=int, default=16)
    parser.add_argument("--memory_sample_size", type=int, default=16)
    parser.add_argument("--mask_ratio", type=float, default=0.8)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--ema_alpha", type=float, default=0.999)
    parser.add_argument("--prototype_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--plot_stride", type=int, default=25)
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    online_adapt(parse_args())


