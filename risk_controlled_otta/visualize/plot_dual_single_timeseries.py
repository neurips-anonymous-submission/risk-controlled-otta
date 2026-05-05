from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SERIES_SPECS = {
    "trigger_score": {
        "source": "step",
        "label": r"Risk Score $s_t$",
        "color": "#55a868",
        "group": "Behavior",
        "clip": None,
    },
    "mean_confidence": {
        "source": "step",
        "label": "Keypoint Localization Reliability",
        "color": "#55a868",
        "group": "Behavior",
        "clip": None,
    },
    "num_ransac_inliers": {
        "source": "step",
        "label": "RANSAC Inliers",
        "color": "#55a868",
        "group": "Behavior",
        "clip": None,
    },
    "total_loss": {
        "source": "learn",
        "label": "Total Loss",
        "color": "#4c72b0",
        "group": "Optimization",
        "clip": 99.5,
    },
    "loss_geometry": {
        "source": "learn",
        "label": r"$L_{geo}$",
        "color": "#4c72b0",
        "group": "Optimization",
        "clip": 99.5,
    },
    "e_star_p": {
        "source": "step",
        "label": r"Pose Error $E_p^*$",
        "color": "#e15759",
        "group": "Performance",
        "clip": 99.5,
    },
    "e_star_q_deg": {
        "source": "step",
        "label": r"Rotation Error $E_q^*$ [deg]",
        "color": "#e15759",
        "group": "Performance",
        "clip": 99.5,
    },
    "mean_reprojection_error": {
        "source": "step",
        "label": "Reprojection Error",
        "color": "#e15759",
        "group": "Performance",
        "clip": 97.5,
    },
    "inlier_failure_rate": {
        "source": "step",
        "label": r"Inlier Failure Rate $(1-\rho_t)$",
        "color": "#e15759",
        "group": "Performance",
        "clip": None,
    },
}


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="same")


def moving_envelope(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    out = np.empty_like(values)
    for i in range(len(values)):
        out[i] = np.mean(padded[i : i + window])
    return out


def binned_quantile_band(values: np.ndarray, bin_size: int, low_q: float = 0.15, high_q: float = 0.85):
    if bin_size <= 1 or len(values) == 0:
        x = np.arange(1, len(values) + 1, dtype=np.float64)
        return x, values, values
    xs = []
    lows = []
    highs = []
    for start in range(0, len(values), bin_size):
        chunk = values[start : start + bin_size]
        if len(chunk) == 0:
            continue
        xs.append(start + len(chunk) / 2.0)
        lows.append(float(np.quantile(chunk, low_q)))
        highs.append(float(np.quantile(chunk, high_q)))
    return np.asarray(xs), np.asarray(lows), np.asarray(highs)


def load_list(path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj["history"] if isinstance(obj, dict) and "history" in obj else obj


def finite_array(values) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return arr
    finite_mask = np.isfinite(arr)
    if finite_mask.all():
        return arr
    if not finite_mask.any():
        return np.zeros_like(arr)
    fill = float(np.nanmedian(arr[finite_mask]))
    arr[~finite_mask] = fill
    return arr


def plot_metric(
    step_history: Path,
    learnable_history: Path,
    metric: str,
    output_path: Path,
    rolling_window: int,
    title: str | None,
    raw_alpha: float,
    clip_percentile: float,
) -> None:
    spec = SERIES_SPECS[metric]
    step_data = load_list(step_history)
    learn_data = load_list(learnable_history)
    data = step_data if spec["source"] == "step" else learn_data
    if metric == "inlier_failure_rate":
        values = finite_array(1.0 - float(item.get("inlier_ratio", 0.0)) for item in data)
    else:
        values = finite_array(item.get(metric, 0.0) for item in data)
    steps = np.arange(1, len(values) + 1, dtype=np.int32)
    smooth = moving_average(values, rolling_window)

    display_values = values.copy()
    effective_clip = clip_percentile if spec["clip"] is not None else None
    if effective_clip is not None and len(display_values):
        upper = float(np.percentile(display_values, effective_clip))
        upper = max(upper, float(display_values.max()) * 0.15)
        display_values = np.clip(display_values, None, upper)
        smooth = moving_average(display_values, rolling_window)
    else:
        upper = float(display_values.max()) if len(display_values) else 1.0

    envelope = moving_envelope(display_values, max(11, rolling_window // 3))
    band_x, band_low, band_high = binned_quantile_band(display_values, max(32, rolling_window // 2))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 2.35), constrained_layout=True)
    ax.fill_between(band_x, band_low, band_high, color=spec["color"], alpha=0.18, linewidth=0)
    ax.plot(steps, envelope, color=spec["color"], alpha=max(0.18, raw_alpha * 0.55), linewidth=0.9)
    ax.plot(steps, smooth, color="black", linewidth=1.6)

    ax.set_xlabel(r"Test-time adaptation step $(t)$", fontsize=12)
    ax.set_ylabel(spec["label"], fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#fcfcfc")
    y_top = float(max(display_values.max() if len(display_values) else 1.0, smooth.max() if len(smooth) else 1.0))
    ax.set_ylim(0.0, y_top * 1.12 if y_top > 0 else 1.0)

    ax.text(
        0.995,
        0.9,
        spec["group"],
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color=spec["color"],
        fontweight="bold",
    )
    if title:
        ax.set_title(title, fontsize=12, pad=6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {metric} to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a single dual-branch OTTA timeseries.")
    parser.add_argument("--step_history", type=Path, required=True)
    parser.add_argument("--learnable_history", type=Path, required=True)
    parser.add_argument("--metric", type=str, choices=sorted(SERIES_SPECS.keys()), required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--rolling_window", type=int, default=181)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--raw_alpha", type=float, default=0.4)
    parser.add_argument("--clip_percentile", type=float, default=97.5)
    args = parser.parse_args()
    plot_metric(
        args.step_history,
        args.learnable_history,
        args.metric,
        args.output_path,
        args.rolling_window,
        args.title,
        args.raw_alpha,
        args.clip_percentile,
    )


if __name__ == "__main__":
    main()

