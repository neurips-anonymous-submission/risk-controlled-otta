from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from risk_controlled_otta.adapt.triggered_single_model_tta_dino_heatmap import (
    diagnose_prediction,
    tensor_bbox_to_tuple,
)
from risk_controlled_otta.data.dino_heatmap_dataset import SpeedPlusDinoHeatmapDataset
from risk_controlled_otta.visualize.visualize_dino_heatmaps_and_cls import (
    draw_predicted_keypoints,
    load_model,
    make_heatmap_overlay,
)
from data.crop_and_heatmap import load_camera


def build_diagnosis_args() -> SimpleNamespace:
    return SimpleNamespace(
        disable_nms=False,
        nms_kernel=3,
        disable_subpixel=False,
        subpixel_radius=2,
        min_confidence=0.05,
        top_k=8,
        min_points=6,
        ransac_reproj_error=6.0,
        ransac_iterations=100,
        ransac_confidence=0.999,
        disable_iterative_refine=False,
        trigger_confidence=0.15,
        trigger_min_inliers=5,
        trigger_reprojection_error=8.0,
        quality_reprojection_cap=50.0,
    )


def collect_ranked_samples(
    dataset: SpeedPlusDinoHeatmapDataset,
    model,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    device: torch.device,
    args: SimpleNamespace,
    max_scan: int | None,
) -> List[Dict[str, object]]:
    ranked: List[Dict[str, object]] = []
    total = len(dataset) if max_scan is None else min(len(dataset), int(max_scan))

    for index in tqdm(range(total), desc=f"score_{dataset.split}"):
        sample = dataset[index]
        image_uint8 = sample["image_uint8"].numpy().astype(np.uint8)
        image_name = str(sample["image_name"])
        bbox = tensor_bbox_to_tuple(sample["bbox"])

        image_tensor = sample["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            heatmap, cls_token = model(image_tensor, return_features=True)

        diagnosis = diagnose_prediction(
            heatmap=heatmap,
            bbox=bbox,
            input_size=dataset.input_size,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            args=args,
        )

        ranked.append(
            {
                "index": index,
                "image_name": image_name,
                "image_uint8": image_uint8,
                "heatmap": heatmap[0].detach().cpu(),
                "cls_token": cls_token[0].detach().cpu(),
                "diagnosis": diagnosis,
            }
        )

    ranked.sort(key=lambda item: float(item["diagnosis"]["quality"]), reverse=True)
    return ranked


def build_metric_text(item: Dict[str, object]) -> str:
    diagnosis = item["diagnosis"]
    return "\n".join(
        [
            f"quality: {float(diagnosis['quality']):.4f}",
            f"mean conf.: {float(diagnosis['mean_confidence']):.4f}",
            f"inliers: {int(diagnosis['num_ransac_inliers'])}",
            f"inlier ratio: {float(diagnosis['inlier_ratio']):.4f}",
            f"mean reproj.: {float(diagnosis['mean_reprojection_error']):.4f}",
            f"fallback: {bool(diagnosis['used_fallback_epnp'])}",
        ]
    )


def draw_sample_panel(ax_row, item: Dict[str, object], title_prefix: str) -> None:
    image = item["image_uint8"]
    heatmap = item["heatmap"]

    keypoint_vis = draw_predicted_keypoints(image, heatmap)
    heatmap_overlay = make_heatmap_overlay(image, heatmap)
    metrics_text = build_metric_text(item)

    ax_row[0].imshow(image)
    ax_row[0].set_title(f"{title_prefix}: Input")
    ax_row[0].axis("off")

    ax_row[1].imshow(keypoint_vis)
    ax_row[1].set_title(f"{title_prefix}: Keypoints")
    ax_row[1].axis("off")

    ax_row[2].imshow(heatmap_overlay)
    ax_row[2].set_title(f"{title_prefix}: Heatmap")
    ax_row[2].axis("off")

    ax_row[3].axis("off")
    ax_row[3].text(
        0.02,
        0.98,
        f"{item['image_name']}\n\n{metrics_text}",
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )
    ax_row[3].set_title(f"{title_prefix}: Metrics")


def save_pair_figure(
    reliable_item: Dict[str, object],
    unreliable_item: Dict[str, object],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    draw_sample_panel(axes[0], reliable_item, "Reliable")
    draw_sample_panel(axes[1], unreliable_item, "Unreliable")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_overview(pairs: List[Path], output_path: Path) -> None:
    if not pairs:
        return
    images = [plt.imread(path) for path in pairs]
    fig, axes = plt.subplots(len(images), 1, figsize=(16, 8 * len(images)))
    if len(images) == 1:
        axes = [axes]
    for ax, image, path in zip(axes, images, pairs):
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
    parser.add_argument("--output_dir", type=str, default="visualization_results_dinov3_heatmap/reliable_vs_unreliable")
    parser.add_argument("--num_pairs", type=int, default=3)
    parser.add_argument("--max_scan", type=int, default=200, help="How many samples to score before selecting top/bottom examples")
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
    camera_matrix, dist_coeffs = load_camera(Path(args.data_root))
    model = load_model(args.model_path, device)
    diagnosis_args = build_diagnosis_args()

    ranked = collect_ranked_samples(
        dataset=dataset,
        model=model,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        device=device,
        args=diagnosis_args,
        max_scan=args.max_scan,
    )

    num_pairs = max(1, min(int(args.num_pairs), len(ranked) // 2))
    reliable_samples = ranked[:num_pairs]
    unreliable_samples = list(reversed(ranked[-num_pairs:]))

    saved_pairs: List[Path] = []
    for pair_index, (reliable_item, unreliable_item) in enumerate(zip(reliable_samples, unreliable_samples), start=1):
        output_path = output_dir / f"{pair_index:02d}_reliable_vs_unreliable.png"
        save_pair_figure(reliable_item, unreliable_item, output_path)
        saved_pairs.append(output_path)

    save_overview(saved_pairs, output_dir / f"{args.split}_reliable_vs_unreliable_overview.png")
    print(f"Saved {len(saved_pairs)} comparison figures to: {output_dir}")


if __name__ == "__main__":
    main()


