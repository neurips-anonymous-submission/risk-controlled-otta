from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


METHOD_STYLES = {
    "source_only": {"label": "Source-only/No TTA", "color": "#2ca02c"},
    "continuous_otta": {"label": "Continuous OTTA", "color": "#ff7f0e"},
    "ours": {"label": "Risk-Controlled-OTTA (Dual-Branch)", "color": "#1f4aff"},
}


def load_records(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        if "history" in payload and isinstance(payload["history"], list):
            return payload["history"]
        if "records" in payload and isinstance(payload["records"], list):
            return payload["records"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported history format in {path}")


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    out = np.zeros_like(values, dtype=np.float64)
    for idx in range(len(values)):
        left = max(0, idx - window + 1)
        out[idx] = values[left : idx + 1].mean()
    return out


def load_metric_series(path: Path, metric_key: str) -> np.ndarray:
    history = load_records(path)
    values = np.asarray([float(item[metric_key]) for item in history], dtype=np.float64)
    return values


def plot_figure(
    series_by_method: Dict[str, np.ndarray],
    output_path: Path,
    metric_label: str,
    collapse_threshold: float,
    global_window: int,
    zoom_window: int,
    zoom_start: int,
    zoom_length: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), gridspec_kw={"width_ratios": [1.9, 1.0]})
    ax_global, ax_zoom = axes

    total_steps = max(len(series) for series in series_by_method.values())
    zoom_left = max(1, int(zoom_start))
    zoom_right = min(total_steps, int(zoom_start + zoom_length - 1))

    for method_name in ("source_only", "continuous_otta", "ours"):
        if method_name not in series_by_method:
            continue
        style = METHOD_STYLES[method_name]
        raw = series_by_method[method_name]
        steps = np.arange(1, len(raw) + 1, dtype=np.int32)

        smooth_global = rolling_mean(raw, global_window)
        ax_global.plot(
            steps,
            smooth_global,
            color=style["color"],
            linewidth=3.0 if method_name == "ours" else 2.2,
            alpha=1.0 if method_name == "ours" else 0.9,
            label=style["label"],
        )

        zoom_slice = slice(zoom_left - 1, zoom_right)
        zoom_steps = steps[zoom_slice]
        zoom_raw = raw[zoom_slice]
        zoom_smooth = rolling_mean(zoom_raw, zoom_window)
        ax_zoom.plot(
            zoom_steps,
            zoom_raw,
            color=style["color"],
            linewidth=1.0,
            alpha=0.18 if method_name != "ours" else 0.25,
        )
        ax_zoom.plot(
            zoom_steps,
            zoom_smooth,
            color=style["color"],
            linewidth=3.0 if method_name == "ours" else 2.2,
            alpha=1.0 if method_name == "ours" else 0.9,
        )

    for ax in axes:
        ax.axhline(collapse_threshold, color="#777777", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.grid(axis="y", alpha=0.22)
        ax.tick_params(labelsize=11)

    ax_global.axvspan(zoom_left, zoom_right, color="#b0b0b0", alpha=0.15)
    ax_global.set_title("(a) Global Trend", fontsize=13)
    ax_global.set_xlabel("Test-time adaptation step", fontsize=13)
    ax_global.set_ylabel(metric_label, fontsize=14)
    ax_global.legend(frameon=False, fontsize=12, loc="upper left")

    ax_zoom.set_title("(b) Failure Episode Zoom-in", fontsize=13)
    ax_zoom.set_xlabel("Step", fontsize=13)
    ax_zoom.set_xlim(zoom_left, zoom_right)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_only_history", type=str, required=True)
    parser.add_argument("--continuous_otta_history", type=str, required=True)
    parser.add_argument("--ours_history", type=str, required=True)
    parser.add_argument("--metric_key", type=str, default="e_star_p")
    parser.add_argument("--metric_label", type=str, default=r"Pose Error ($E_p^*$)")
    parser.add_argument("--collapse_threshold", type=float, default=0.1)
    parser.add_argument("--global_window", type=int, default=50)
    parser.add_argument("--zoom_window", type=int, default=12)
    parser.add_argument("--zoom_start", type=int, default=6610)
    parser.add_argument("--zoom_length", type=int, default=100)
    parser.add_argument(
        "--output_path",
        type=str,
        default="visualization_results_dinov3_heatmap/lightbox_pose_error_global_and_zoom.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    series_by_method = {
        "source_only": load_metric_series(Path(args.source_only_history), args.metric_key),
        "continuous_otta": load_metric_series(Path(args.continuous_otta_history), args.metric_key),
        "ours": load_metric_series(Path(args.ours_history), args.metric_key),
    }
    plot_figure(
        series_by_method=series_by_method,
        output_path=Path(args.output_path),
        metric_label=args.metric_label,
        collapse_threshold=float(args.collapse_threshold),
        global_window=int(args.global_window),
        zoom_window=int(args.zoom_window),
        zoom_start=int(args.zoom_start),
        zoom_length=int(args.zoom_length),
    )


if __name__ == "__main__":
    main()


