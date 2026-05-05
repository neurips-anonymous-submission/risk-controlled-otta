from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from risk_controlled_otta.data.dino_heatmap_dataset import SpeedPlusDinoHeatmapDataset
from risk_controlled_otta.losses.heatmap_loss import heatmap_mse_loss
from risk_controlled_otta.models.dino_pose_model import DinoHeatmapPoseModel


def build_dataloader(
    data_root: str,
    split: str,
    batch_size: int,
    num_workers: int,
    input_size: int,
    heatmap_size: int,
    heatmap_sigma: float,
    use_source_augmentation: bool,
) -> DataLoader:
    dataset = SpeedPlusDinoHeatmapDataset(
        data_root=data_root,
        split=split,
        input_size=input_size,
        heatmap_size=heatmap_size,
        heatmap_sigma=heatmap_sigma,
        use_source_augmentation=use_source_augmentation,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )


def set_learning_rate(optimizer, epoch: int, global_step: int, warmup_steps: int) -> float:
    scale = 1.0
    if global_step < warmup_steps:
        scale = float(global_step + 1) / float(max(warmup_steps, 1))
    elif epoch >= 15:
        scale = 0.01
    elif epoch >= 10:
        scale = 0.1

    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["base_lr"] * scale
    return optimizer.param_groups[0]["lr"]


def build_optimizer(model: DinoHeatmapPoseModel, args) -> AdamW:
    return AdamW(
        [
            {
                "name": "encoder",
                "params": model.encoder.parameters(),
                "lr": args.encoder_lr,
                "base_lr": args.encoder_lr,
                "weight_decay": args.weight_decay,
            },
            {
                "name": "decoder",
                "params": model.decoder.parameters(),
                "lr": args.decoder_lr,
                "base_lr": args.decoder_lr,
                "weight_decay": args.weight_decay,
            },
        ]
    )


@torch.no_grad()
def validate(model, dataloader, device, args):
    model.eval()
    running_loss = 0.0
    for batch in tqdm(dataloader, desc="val", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        heatmaps = batch["heatmap"].to(device, non_blocking=True)
        with autocast(enabled=device.type == "cuda"):
            pred_heatmap = model(images)
            loss = heatmap_mse_loss(
                pred_heatmap,
                heatmaps,
                positive_weight=args.positive_weight,
                positive_threshold=args.positive_threshold,
            )
        running_loss += float(loss.item())
    return running_loss / max(len(dataloader), 1)


def set_encoder_trainable(model: DinoHeatmapPoseModel, trainable: bool) -> None:
    for param in model.encoder.parameters():
        param.requires_grad = trainable


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_dataloader(
        data_root=args.data_root,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        heatmap_sigma=args.heatmap_sigma,
        use_source_augmentation=True,
    )
    val_loader = build_dataloader(
        data_root=args.data_root,
        split="validation",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        heatmap_sigma=args.heatmap_sigma,
        use_source_augmentation=False,
    )

    model = DinoHeatmapPoseModel(
        model_name=args.model_name,
        input_size=args.input_size,
        num_keypoints=11,
        mid_channels=args.mid_channels,
        num_deconv_layers=args.num_deconv_layers,
        pretrained=(not args.no_pretrained),
        pretrained_path=args.pretrained_path,
    ).to(device)

    optimizer = build_optimizer(model, args)
    scaler = GradScaler(enabled=device.type == "cuda")

    best_val = float("inf")
    global_step = 0
    for epoch in range(args.epochs):
        encoder_trainable = epoch >= args.freeze_encoder_epochs
        set_encoder_trainable(model, encoder_trainable)
        model.train()
        running_loss = 0.0
        current_lr = optimizer.param_groups[0]["lr"]

        for batch in tqdm(train_loader, desc=f"train epoch {epoch + 1}", leave=False):
            current_lr = set_learning_rate(optimizer, epoch, global_step, args.warmup_steps)
            images = batch["image"].to(device, non_blocking=True)
            heatmaps = batch["heatmap"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                pred_heatmap = model(images)
                loss = heatmap_mse_loss(
                    pred_heatmap,
                    heatmaps,
                    positive_weight=args.positive_weight,
                    positive_threshold=args.positive_threshold,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            global_step += 1

        train_loss = running_loss / max(len(train_loader), 1)
        val_loss = validate(model, val_loader, device, args)
        print(
            f"epoch={epoch + 1} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"encoder_lr={optimizer.param_groups[0]['lr']:.8f} decoder_lr={optimizer.param_groups[1]['lr']:.8f} "
            f"encoder_trainable={encoder_trainable}"
        )

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "model_name": args.model_name,
            "input_size": args.input_size,
            "heatmap_size": args.heatmap_size,
            "heatmap_sigma": args.heatmap_sigma,
            "mid_channels": args.mid_channels,
            "num_deconv_layers": args.num_deconv_layers,
            "positive_weight": args.positive_weight,
            "positive_threshold": args.positive_threshold,
        }
        torch.save(checkpoint, output_dir / "last_source_dino_heatmap.pth")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, output_dir / "best_source_dino_heatmap.pth")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="speedplusv2")
    parser.add_argument("--output_dir", type=str, default="output/dinov2_heatmap_source")
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_dinov3.lvd1689m")
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--input_size", type=int, default=384)
    parser.add_argument("--heatmap_size", type=int, default=96)
    parser.add_argument("--heatmap_sigma", type=float, default=3.0)
    parser.add_argument("--mid_channels", type=int, default=256)
    parser.add_argument("--num_deconv_layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--encoder_lr", type=float, default=5e-5)
    parser.add_argument("--decoder_lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--freeze_encoder_epochs", type=int, default=0)
    parser.add_argument("--positive_weight", type=float, default=4.0)
    parser.add_argument("--positive_threshold", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())


