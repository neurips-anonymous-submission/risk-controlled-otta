from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from risk_controlled_otta.data.dino_heatmap_dataset import SpeedPlusDinoHeatmapDataset
from risk_controlled_otta.eval.evaluate_dino_heatmap import decode_heatmap_to_keypoints
from risk_controlled_otta.models.dino_pose_model import DinoHeatmapPoseModel


def load_model(checkpoint_path: str, device: torch.device) -> DinoHeatmapPoseModel:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = DinoHeatmapPoseModel(
        model_name=checkpoint.get("model_name", "vit_base_patch16_dinov3.lvd1689m"),
        input_size=int(checkpoint.get("input_size", 384)),
        num_keypoints=11,
        mid_channels=int(checkpoint.get("mid_channels", 256)),
        num_deconv_layers=int(checkpoint.get("num_deconv_layers", 2)),
        pretrained=False,
    ).to(device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "student_state_dict" in checkpoint:
        state_dict = checkpoint["student_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def parse_indices(indices_text: str | None, num_examples: int, dataset_length: int) -> List[int]:
    if indices_text:
        indices = [int(item.strip()) for item in indices_text.split(",") if item.strip()]
    else:
        indices = list(range(min(num_examples, dataset_length)))
    return [index for index in indices if 0 <= index < dataset_length]


def normalize_map(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    values = values - values.min()
    denom = max(float(values.max()), 1e-6)
    return values / denom


def cls_token_to_tile(cls_token: torch.Tensor, tile_size: int = 16) -> np.ndarray:
    vector = cls_token.detach().cpu().numpy().astype(np.float32).reshape(-1)
    tile_elems = tile_size * tile_size
    if vector.shape[0] < tile_elems:
        padded = np.zeros(tile_elems, dtype=np.float32)
        padded[: vector.shape[0]] = vector
        vector = padded
    else:
        vector = vector[:tile_elems]
    tile = vector.reshape(tile_size, tile_size)
    return normalize_map(tile)


def make_heatmap_overlay(image: np.ndarray, heatmap: torch.Tensor) -> np.ndarray:
    heatmap_np = heatmap.detach().cpu().numpy()
    heatmap_map = heatmap_np.max(axis=0)
    heatmap_map = normalize_map(heatmap_map)
    heatmap_map = cv2.resize(heatmap_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    colored = cv2.applyColorMap((heatmap_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = (0.55 * image.astype(np.float32) + 0.45 * colored.astype(np.float32)).clip(0, 255).astype(np.uint8)
    return overlay


def draw_predicted_keypoints(image: np.ndarray, heatmap: torch.Tensor) -> np.ndarray:
    decoded, confidences = decode_heatmap_to_keypoints(heatmap.unsqueeze(0))
    keypoints = decoded[0]
    confidences = confidences[0]
    hm_h, hm_w = heatmap.shape[-2:]

    vis = image.copy()
    for index, (point, conf) in enumerate(zip(keypoints, confidences)):
        x = int(round(point[0] / hm_w * image.shape[1]))
        y = int(round(point[1] / hm_h * image.shape[0]))
        color = (0, 255, 0) if conf > 0.05 else (255, 64, 64)
        cv2.circle(vis, (x, y), 5, color, -1)
        cv2.putText(vis, f"{index}", (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return vis


def visualize_example(
    image: np.ndarray,
    heatmap: torch.Tensor,
    cls_token: torch.Tensor,
    image_name: str,
    output_path: Path,
) -> None:
    keypoint_vis = draw_predicted_keypoints(image, heatmap)
    heatmap_overlay = make_heatmap_overlay(image, heatmap)
    cls_tile = cls_token_to_tile(cls_token, tile_size=16)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image)
    axes[0].set_title("Input Crop")
    axes[0].axis("off")

    axes[1].imshow(keypoint_vis)
    axes[1].set_title("Decoded Keypoints")
    axes[1].axis("off")

    axes[2].imshow(heatmap_overlay)
    axes[2].set_title("Predicted Heatmap Overlay")
    axes[2].axis("off")

    im = axes[3].imshow(cls_tile, cmap="viridis")
    axes[3].set_title("CLS Feature Tile")
    axes[3].axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    fig.suptitle(image_name)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_contact_sheet(example_paths: List[Path], output_path: Path) -> None:
    images = [plt.imread(path) for path in example_paths if path.exists()]
    if not images:
        return
    fig, axes = plt.subplots(len(images), 1, figsize=(16, 4 * len(images)))
    if len(images) == 1:
        axes = [axes]
    for ax, image, path in zip(axes, images, example_paths):
        ax.imshow(image)
        ax.set_title(path.stem)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="speedplusv2")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="sunlamp")
    parser.add_argument("--output_dir", type=str, default="visualization_results_dinov3_heatmap")
    parser.add_argument("--num_examples", type=int, default=3)
    parser.add_argument("--indices", type=str, default=None, help="Comma-separated dataset indices, e.g. 0,10,25")
    parser.add_argument("--input_size", type=int, default=384)
    parser.add_argument("--heatmap_size", type=int, default=96)
    parser.add_argument("--heatmap_sigma", type=float, default=3.0)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = SpeedPlusDinoHeatmapDataset(
        data_root=args.data_root,
        split=args.split,
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        heatmap_sigma=args.heatmap_sigma,
        use_source_augmentation=False,
    )
    model = load_model(args.model_path, device)

    indices = parse_indices(args.indices, args.num_examples, len(dataset))
    saved_paths: List[Path] = []

    for order, index in enumerate(indices, start=1):
        sample = dataset[index]
        image = sample["image_uint8"].numpy().astype(np.uint8)
        image_name = str(sample["image_name"])

        image_tensor = sample["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            heatmap, cls_token = model(image_tensor, return_features=True)

        save_path = output_dir / f"{order:02d}_{Path(image_name).stem}_vis.png"
        visualize_example(
            image=image,
            heatmap=heatmap[0].cpu(),
            cls_token=cls_token[0].cpu(),
            image_name=image_name,
            output_path=save_path,
        )
        saved_paths.append(save_path)

    make_contact_sheet(saved_paths, output_dir / f"{args.split}_overview.png")
    print(f"Saved {len(saved_paths)} example visualizations to: {output_dir}")


if __name__ == "__main__":
    main()


