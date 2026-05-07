from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from data.crop_and_heatmap import (
    compute_expanded_bbox,
    crop_and_resize,
    normalize_image,
    project_keypoints,
    visible_keypoints_mask,
)
# 注意这里的导入路径改成了新的 dinov2_pose_model
from dinov2_heatmap_otta.models.dinov2_pose_model import DinoHeatmapPoseModel
from dinov2_heatmap_otta.losses.heatmap_loss import heatmap_mse_loss


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
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


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
        heatmaps[idx] = gaussian2d(
            (heatmap_size, heatmap_size),
            sigma=sigma,
            center=(x, y),
        )

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
        use_augmentation: bool = False,
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
        self.use_augmentation = bool(use_augmentation)

        self.camera_matrix, self.dist_coeffs = load_camera_from_shirt(self.data_root)
        self.object_keypoints_3d = np.asarray(
            SHIRT_KEYPOINTS_3D if object_keypoints_3d is None else object_keypoints_3d,
            dtype=np.float32,
        )
        assert self.object_keypoints_3d.ndim == 2 and self.object_keypoints_3d.shape[1] == 3
        assert self.object_keypoints_3d.shape[0] == 11, "SHIRT_KEYPOINTS_3D must be shape [11, 3]"

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

    def _apply_aug(self, image: np.ndarray) -> np.ndarray:
        if not self.use_augmentation:
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

        crop_image = self._apply_aug(crop_image)

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


@torch.no_grad()
def validate(model, loader, criterion, device) -> float:
    model.eval()
    losses: List[float] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        target = batch["heatmap"].to(device, non_blocking=True)

        pred = model(images)
        loss = criterion(pred, target)
        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else math.inf


def train_one_epoch(model, loader, optimizer, criterion, device, scaler) -> float:
    model.train()
    losses: List[float] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        target = batch["heatmap"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            pred = model(images)
            loss = criterion(pred, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else math.inf


def build_model(args) -> DinoHeatmapPoseModel:
    # 适配新的 DINOv2 初始化接口
    return DinoHeatmapPoseModel(
        input_size=args.input_size,
        output_heatmap_size=args.heatmap_size,
        num_keypoints=args.num_keypoints,
        mid_channels=args.mid_channels,
        num_deconv_layers=args.num_deconv_layers,
        pretrained=True,
        pretrained_path=args.pretrained_path,
    )


def main(args) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.roe == "all":
        roes = ["roe1", "roe2"]
    else:
        roes = [args.roe]

    train_dataset = ShirtDinoHeatmapDataset(
        data_root=args.data_root,
        roes=roes,
        domain="synthetic",
        split="train",
        val_ratio=args.val_ratio,
        seed=args.seed,
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        heatmap_sigma=args.heatmap_sigma,
        expand_ratio=args.expand_ratio,
        use_augmentation=True,
    )
    val_dataset = ShirtDinoHeatmapDataset(
        data_root=args.data_root,
        roes=roes,
        domain="synthetic",
        split="val",
        val_ratio=args.val_ratio,
        seed=args.seed,
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        heatmap_sigma=args.heatmap_sigma,
        expand_ratio=args.expand_ratio,
        use_augmentation=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = build_model(args).to(device)
    criterion = heatmap_mse_loss
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_val = math.inf
    history: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss = validate(model, val_loader, criterion, device)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        history.append(record)
        print(json.dumps(record))

        ckpt = {
            "model_state_dict": model.state_dict(),
            "input_size": args.input_size,
            "heatmap_size": args.heatmap_size,
            "heatmap_sigma": args.heatmap_sigma,
            "mid_channels": args.mid_channels,
            "num_deconv_layers": args.num_deconv_layers,
            "num_keypoints": args.num_keypoints,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "dataset": "SHIRT",
            "roes": roes,
            "source_domain": "synthetic",
            "pretrained_path": args.pretrained_path,
        }

        # 名字已更新为 dinov2
        torch.save(ckpt, output_dir / "last_shirt_source_dinov2_heatmap.pth")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, output_dir / "best_shirt_source_dinov2_heatmap.pth")

        save_json(
            {
                "args": vars(args),
                "best_val_loss": best_val,
                "history": history,
            },
            output_dir / "train_history.json",
        )

    print(f"[OK] best_val_loss={best_val:.6f}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="SHIRT_Dataset")
    parser.add_argument("--roe", type=str, choices=["roe1", "roe2", "all"], default="all")
    # 默认输出路径已更新为 dinov2
    parser.add_argument("--output_dir", type=str, default="output/dinov2_heatmap_shirt_source_v1")

    # 默认读取路径已更新为 DINOv2 本地目录
    parser.add_argument("--pretrained_path", type=str, default="pretrained/dinov2_base")

    parser.add_argument("--input_size", type=int, default=384)
    parser.add_argument("--heatmap_size", type=int, default=96)
    parser.add_argument("--heatmap_sigma", type=float, default=3.0)
    parser.add_argument("--num_keypoints", type=int, default=11)
    parser.add_argument("--mid_channels", type=int, default=256)
    parser.add_argument("--num_deconv_layers", type=int, default=2)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--expand_ratio", type=float, default=1.25)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())