from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
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

    mask = patch_mask.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2).unsqueeze(1)

    # In case input_size is not an exact multiple of patch_size, pad/crop mask to match image size.
    if mask.shape[-2:] != images.shape[-2:]:
        mask = torch.nn.functional.interpolate(mask, size=images.shape[-2:], mode="nearest")
    return images * mask


@torch.no_grad()
def update_ema_teacher(student_model: torch.nn.Module, teacher_model: torch.nn.Module, alpha: float = 0.999) -> None:
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1.0 - alpha)


@torch.no_grad()
def build_source_prototype(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> torch.Tensor:
    model.eval()
    features = []
    for batch in tqdm(dataloader, desc="build_prototype", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        with autocast("cuda", enabled=device.type == "cuda"):
            _, cls_token = model(images, return_features=True)
        features.append(cls_token.detach().cpu())

    if not features:
        raise RuntimeError("Unable to build source prototype: source dataloader is empty.")

    prototype = torch.cat(features, dim=0).mean(dim=0)
    # Keep the historical two-branch prototype shape expected by class_awareness_consistency_loss.
    return torch.cat([prototype, prototype], dim=0)


@torch.no_grad()
def initialize_memory_bank(dataloader: DataLoader, memory_bank: DynamicMemoryBank, max_samples: int = 16) -> None:
    dataset = getattr(dataloader, "dataset", None)
    if dataset is None:
        raise ValueError("Memory-bank initialization requires access to the source dataset.")

    num_samples = min(max_samples, len(dataset))
    if num_samples <= 0:
        raise RuntimeError("Unable to initialize memory bank: source dataset is empty.")

    sampled_indices = np.random.choice(len(dataset), size=num_samples, replace=False)
    for sample_index in tqdm(sampled_indices, desc="init_memory", leave=False):
        sample = dataset[int(sample_index)]
        memory_bank.push(sample["image"], sample["heatmap"])


def _extract_state_dict(checkpoint: Dict) -> Dict[str, torch.Tensor]:
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "student_state_dict" in checkpoint:
        return checkpoint["student_state_dict"]
    return checkpoint


def _infer_input_size_from_hf_pos_embed(state_dict: Dict[str, torch.Tensor], patch_size: int = 14) -> int | None:
    pos_embed = state_dict.get("encoder.embeddings.position_embeddings")
    if pos_embed is None:
        pos_embed = state_dict.get("embeddings.position_embeddings")
    if pos_embed is None or pos_embed.ndim != 3:
        return None

    num_tokens = int(pos_embed.shape[1])
    num_patch_tokens = num_tokens - 1
    if num_patch_tokens <= 0:
        return None
    grid_size = int(round(num_patch_tokens ** 0.5))
    if grid_size * grid_size != num_patch_tokens:
        return None
    return int(grid_size * patch_size)


def load_source_model(checkpoint_path: str, device: torch.device, args) -> DinoHeatmapPoseModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(checkpoint)

    resolved_input_size = int(checkpoint.get("input_size") or args.input_size)
    inferred_input_size = _infer_input_size_from_hf_pos_embed(state_dict, patch_size=args.patch_size)
    if inferred_input_size is not None and inferred_input_size != resolved_input_size:
        print(
            f"[online_tta_dinov2][warn] checkpoint input_size={resolved_input_size}, "
            f"but inferred input_size={inferred_input_size} from position embeddings. Using inferred value."
        )
        resolved_input_size = inferred_input_size

    resolved_heatmap_size = int(checkpoint.get("heatmap_size") or args.heatmap_size)

    model = DinoHeatmapPoseModel(
        input_size=resolved_input_size,
        output_heatmap_size=resolved_heatmap_size,
        num_keypoints=args.num_keypoints,
        mid_channels=int(checkpoint.get("mid_channels", args.mid_channels)),
        num_deconv_layers=int(checkpoint.get("num_deconv_layers", args.num_deconv_layers)),
        pretrained=False,
    )
    model.load_state_dict(state_dict, strict=True)
    return model.to(device)


def make_dataset(
    args,
    split: str,
    input_size: int,
    heatmap_size: int,
    use_source_augmentation: bool = False,
) -> SpeedPlusDinoHeatmapDataset:
    return SpeedPlusDinoHeatmapDataset(
        data_root=args.data_root,
        split=split,
        input_size=input_size,
        heatmap_size=heatmap_size,
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
    plt.plot(steps[sampled], weighted_st[sampled], label=fr"{lambda_st:g} x L_st", linewidth=1.5)
    plt.plot(steps[sampled], raw_mh[sampled], label="L_mh", linewidth=1.5)
    plt.plot(steps[sampled], weighted_ca[sampled], label=fr"{lambda_ca:g} x L_ca", linewidth=1.5)
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
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    student_model = load_source_model(args.source_checkpoint, device, args)
    model_input_size = int(getattr(student_model, "input_size", args.input_size))
    model_heatmap_size = int(getattr(student_model, "output_heatmap_size", args.heatmap_size))

    source_dataset = make_dataset(args, "train", model_input_size, model_heatmap_size, use_source_augmentation=False)
    target_dataset = make_dataset(args, args.target_split, model_input_size, model_heatmap_size, use_source_augmentation=False)
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

    source_prototype = build_source_prototype(student_model, source_loader, device)

    memory_bank = DynamicMemoryBank(capacity=args.memory_capacity)
    initialize_memory_bank(source_loader, memory_bank, max_samples=args.memory_capacity)

    teacher_model = copy.deepcopy(student_model)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    if args.update_scope == "decoder":
        for param in student_model.parameters():
            param.requires_grad = False
        for param in student_model.decoder.parameters():
            param.requires_grad = True
        trainable_params = [p for p in student_model.decoder.parameters() if p.requires_grad]
    elif args.update_scope == "all":
        for param in student_model.parameters():
            param.requires_grad = True
        trainable_params = [p for p in student_model.parameters() if p.requires_grad]
    else:
        raise ValueError(f"Unsupported update_scope: {args.update_scope}")

    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    student_model.train()

    loss_history = {"step": [], "total_loss": [], "loss_st": [], "loss_mh": [], "loss_ca": []}

    for step, batch in enumerate(tqdm(target_loader, desc="online_tta_dinov2"), start=1):
        if args.max_samples is not None and step > args.max_samples:
            break

        x_online = batch["image"].to(device, non_blocking=True)

        with torch.no_grad():
            with autocast("cuda", enabled=device.type == "cuda"):
                online_heatmap = student_model(x_online)
            online_scores = heatmap_confidence_score(online_heatmap)
            best_index = int(online_scores.argmax().item())
            memory_bank.push(x_online[best_index], online_heatmap[best_index])

        x_memory, old_pseudo, mem_indices = sample_from_memory_bank(memory_bank, args.memory_sample_size, device)
        masked_memory = random_discontinuous_mask(x_memory, mask_ratio=args.mask_ratio, patch_size=args.patch_size)

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=device.type == "cuda"):
            h_student, cls_student = student_model(x_memory, return_features=True)
            with torch.no_grad():
                h_teacher, cls_teacher = teacher_model(masked_memory, return_features=True)

            loss_st = self_training_loss(h_student, old_pseudo)
            loss_mh = masked_heatmap_consistency_loss(h_student, h_teacher, tau=args.tau)
            loss_ca = class_awareness_consistency_loss(cls_student, cls_teacher, source_prototype.to(device))
            total_loss = total_target_loss(
                loss_st,
                loss_mh,
                loss_ca,
                lambda_st=args.lambda_st,
                lambda_ca=args.lambda_ca,
            )

        scaler.scale(total_loss).backward()
        if args.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip_norm)
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
            "input_size": model_input_size,
            "heatmap_size": model_heatmap_size,
            "heatmap_sigma": args.heatmap_sigma,
            "mid_channels": args.mid_channels,
            "num_deconv_layers": args.num_deconv_layers,
            "num_keypoints": args.num_keypoints,
            "adaptation": "online_tta_dinov2_heatmap",
            "update_scope": args.update_scope,
            "source_checkpoint": args.source_checkpoint,
        },
        output_dir / "tta_final.pth",
    )
    save_loss_curves(
        loss_history,
        output_dir,
        lambda_st=args.lambda_st,
        lambda_ca=args.lambda_ca,
        plot_stride=args.plot_stride,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="speedplusv2")
    parser.add_argument("--source_checkpoint", type=str, required=True)
    parser.add_argument("--target_split", type=str, default="sunlamp")
    parser.add_argument("--output_dir", type=str, default="output/dinov2_heatmap_online_tta_sunlamp")

    parser.add_argument("--input_size", type=int, default=384)
    parser.add_argument("--heatmap_size", type=int, default=96)
    parser.add_argument("--heatmap_sigma", type=float, default=3.0)
    parser.add_argument("--num_keypoints", type=int, default=11)
    parser.add_argument("--mid_channels", type=int, default=256)
    parser.add_argument("--num_deconv_layers", type=int, default=2)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--update_scope", type=str, choices=["decoder", "all"], default="decoder")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    parser.add_argument("--memory_capacity", type=int, default=16)
    parser.add_argument("--memory_sample_size", type=int, default=16)
    parser.add_argument("--mask_ratio", type=float, default=0.8)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--ema_alpha", type=float, default=0.999)
    parser.add_argument("--lambda_st", type=float, default=10.0)
    parser.add_argument("--lambda_ca", type=float, default=0.01)

    parser.add_argument("--prototype_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--plot_stride", type=int, default=25)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    online_adapt(parse_args())