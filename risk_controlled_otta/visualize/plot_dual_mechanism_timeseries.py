from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="same")


def finite_array(values: Iterable[float]) -> np.ndarray:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot dual-branch OTTA mechanism timeseries.")
    parser.add_argument("--step_history", type=Path, required=True)
    parser.add_argument("--learnable_history", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--rolling_window", type=int, default=101)
    parser.add_argument("--title", type=str, default="Risk-Controlled-OTTA (Dual-Branch) on Lightbox")
    args = parser.parse_args()

    step_history = load_json(args.step_history)
    learnable = load_json(args.learnable_history)
    learnable_history = learnable["history"] if isinstance(learnable, dict) else learnable

    n = min(len(step_history), len(learnable_history))
    step_history = step_history[:n]
    learnable_history = learnable_history[:n]
    steps = np.arange(1, n + 1, dtype=np.int32)

    series = [
        {
            "label": "Risk Score $s_t$",
            "values": finite_array(item["trigger_score"] for item in step_history),
            "color": "#55a868",
            "group": "Behavior",
        },
        {
            "label": "Adaptation Event",
            "values": finite_array(float(item["adapted"]) for item in step_history),
            "color": "#55a868",
            "group": "Behavior",
        },
        {
            "label": "Total Loss",
            "values": finite_array(item["total_loss"] for item in learnable_history),
            "color": "#4c72b0",
            "group": "Optimization",
        },
        {
            "label": "$L_{geo}$",
            "values": finite_array(item["loss_geometry"] for item in learnable_history),
            "color": "#4c72b0",
            "group": "Optimization",
        },
        {
            "label": "Pose Error $E_p^*$",
            "values": finite_array(item["e_star_p"] for item in step_history),
            "color": "#e15759",
            "group": "Performance",
        },
        {
            "label": "Reprojection Error",
            "values": finite_array(item["mean_reprojection_error"] for item in step_history),
            "color": "#e15759",
            "group": "Performance",
        },
        {
            "label": "Inlier Failure Rate $(1-\\rho_t)$",
            "values": finite_array(1.0 - float(item["inlier_ratio"]) for item in step_history),
            "color": "#e15759",
            "group": "Performance",
        },
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        len(series),
        1,
        figsize=(14, 8.4),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"hspace": 0.06},
    )

    for idx, (ax, spec) in enumerate(zip(axes, series)):
        values = spec["values"]
        smooth = moving_average(values, args.rolling_window)
        color = spec["color"]

        if spec["label"] == "Adaptation Event":
            ax.vlines(steps[values > 0.5], ymin=0.0, ymax=1.0, color=color, alpha=0.45, linewidth=0.7)
            ax.plot(steps, smooth, color="black", linewidth=1.4)
            ax.set_ylim(-0.02, 1.05)
        else:
            ax.plot(steps, values, color=color, alpha=0.55, linewidth=0.85)
            ax.plot(steps, smooth, color="black", linewidth=1.4)

        ax.set_ylabel(spec["label"], fontsize=11)
        ax.tick_params(axis="both", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("#fcfcfc")

        group = spec["group"]
        ax.text(
            0.995,
            0.86,
            group,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color=color,
            fontweight="bold",
        )

        if spec["label"] == "Pose Error $E_p^*$":
            collapsed = np.asarray([1.0 if item["collapse"] else 0.0 for item in step_history], dtype=np.float64)
            collapse_steps = steps[collapsed > 0.5]
            if collapse_steps.size:
                sample = collapse_steps[:: max(1, collapse_steps.size // 150)]
                ax.vlines(sample, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color="#8c1d18", alpha=0.12, linewidth=0.8)

    axes[-1].set_xlabel("Test-time adaptation step $(t)$", fontsize=12)
    fig.suptitle(args.title, fontsize=13, y=0.995)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {args.output_path}")


if __name__ == "__main__":
    main()


