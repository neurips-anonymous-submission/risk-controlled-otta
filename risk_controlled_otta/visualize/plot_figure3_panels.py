from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


ROOT = Path(r"D:\code")
OUT_DIR = ROOT / "visualization_results_dinov3_heatmap"
MIXED_ROOT = ROOT / "mixed_domain_stream_results" / "table8_full"


METHODS = {
    "source_only": {
        "label": "Source-only / No TTA",
        "color": "#4C78A8",
        "summary_dir": MIXED_ROOT / "source_only" / "source_only",
    },
    "original_otta": {
        "label": "Continuous OTTA",
        "color": "#F58518",
        "summary_dir": MIXED_ROOT / "original_otta" / "original_otta",
    },
    "cotta": {
        "label": "CoTTA",
        "color": "#54A24B",
        "summary_dir": MIXED_ROOT / "cotta" / "cotta",
    },
    "eata": {
        "label": "EATA",
        "color": "#9C755F",
        "summary_dir": MIXED_ROOT / "eata" / "eata",
    },
    "ours": {
        "label": "Risk-Controlled-OTTA (Dual-Branch)",
        "color": "#E45756",
        "summary_dir": MIXED_ROOT / "ours_dual_branch" / "ours",
    },
}

STREAMS = ["forward", "reverse", "cyclic", "shifted"]
STREAM_MARKERS = {
    "forward": "o",
    "reverse": "s",
    "cyclic": "^",
    "shifted": "D",
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def plot_tradeoff(output_path: Path) -> None:
    fig, (ax_l, ax_r) = plt.subplots(
        1,
        2,
        figsize=(9.5, 5.8),
        dpi=220,
        gridspec_kw={"width_ratios": [3.8, 1.4], "wspace": 0.06},
        sharey=True,
    )

    # Subtle emphasis for the useful region.
    ax_l.axvspan(0.0, 0.30, color="#EEF6EE", alpha=0.95, zorder=0)

    marker_handles = [
        plt.Line2D([0], [0], marker=STREAM_MARKERS[stream], linestyle="None", color="black", markersize=6.5, label=stream.capitalize())
        for stream in STREAMS
    ]
    method_handles = []

    x_break = 0.92
    label_offsets = {
        "Source-only / No TTA": (0.012, 0.02),
        "EATA": (-0.04, -0.015),
        "Risk-Controlled-OTTA (Dual-Branch)": (0.012, -0.028),
        "Continuous OTTA": (-0.055, 0.12),
        "CoTTA": (0.015, 0.06),
    }

    for method_key, meta in METHODS.items():
        points = []
        for stream in STREAMS:
            summary = load_json(meta["summary_dir"] / stream / "summary.json")
            points.append(
                (
                    float(summary["adapt_ratio"]),
                    float(summary["p95_e_star_p"]),
                    stream,
                )
            )
        points_xy = np.asarray([(x, y) for x, y, _ in points], dtype=np.float64)
        center_x = float(points_xy[:, 0].mean())
        center_y = float(points_xy[:, 1].mean())
        xerr = np.vstack(
            [
                [center_x - float(points_xy[:, 0].min())],
                [float(points_xy[:, 0].max()) - center_x],
            ]
        )
        yerr = np.vstack(
            [
                [center_y - float(points_xy[:, 1].min())],
                [float(points_xy[:, 1].max()) - center_y],
            ]
        )

        target_ax = ax_l if center_x < x_break else ax_r
        target_ax.errorbar(
            center_x,
            center_y,
            xerr=xerr,
            yerr=yerr,
            fmt="none",
            ecolor=meta["color"],
            elinewidth=1.1,
            alpha=0.55,
            capsize=0,
            zorder=1,
        )

        for x, y, stream in points:
            scatter_ax = ax_l if x < x_break else ax_r
            scatter_ax.scatter(
                x,
                y,
                s=38,
                marker=STREAM_MARKERS[stream],
                color=meta["color"],
                edgecolors="white",
                linewidths=0.7,
                alpha=0.35,
                zorder=2,
            )

        target_ax.scatter(
            center_x,
            center_y,
            s=110 if method_key == "ours" else 90,
            marker="o",
            color=meta["color"],
            edgecolors="white",
            linewidths=1.0,
            zorder=4,
        )
        dx, dy = label_offsets[meta["label"]]
        target_ax.text(
            center_x + dx,
            center_y + dy,
            meta["label"],
            fontsize=9,
            color=meta["color"],
            weight="bold" if method_key == "ours" else None,
        )
        method_handles.append(
            plt.Line2D([0], [0], marker="o", linestyle="None", color=meta["color"], markersize=7, label=meta["label"])
        )

    ymax = max(
        float(load_json(meta["summary_dir"] / stream / "summary.json")["p95_e_star_p"])
        for meta in METHODS.values()
        for stream in STREAMS
    )
    for ax in (ax_l, ax_r):
        ax.grid(True, linewidth=0.5, alpha=0.35)
        ax.set_ylim(0.0, min(float(ymax) * 1.08, 2.6))

    ax_l.set_xlim(-0.03, 0.35)
    ax_r.set_xlim(0.94, 1.02)
    ax_l.set_xlabel("Adaptation Rate", fontsize=11)
    ax_r.set_xlabel("Adaptation Rate", fontsize=11)
    ax_l.set_ylabel(r"$p95\ E_p^*$", fontsize=11)
    ax_l.set_title("Figure 3(a)  Stability-Adaptation Trade-off", fontsize=12)

    ax_r.yaxis.set_visible(False)
    ax_r.spines["left"].set_visible(False)
    ax_l.spines["right"].set_visible(False)

    d = 0.008
    kwargs = dict(transform=ax_l.transAxes, color="k", clip_on=False, linewidth=0.8)
    ax_l.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax_l.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax_r.transAxes)
    ax_r.plot((-d, +d), (-d, +d), **kwargs)
    ax_r.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    ax_l.legend(handles=marker_handles, loc="upper left", frameon=True, fontsize=8, title="Protocols")
    ax_r.legend(handles=method_handles, loc="upper right", frameon=True, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_dynamics(output_path: Path, stream: str = "forward", window: int = 200) -> None:
    chosen = {
        "source_only": METHODS["source_only"],
        "original_otta": METHODS["original_otta"],
        "eata": METHODS["eata"],
        "ours": METHODS["ours"],
    }
    histories = {
        key: load_json(meta["summary_dir"] / stream / "step_history.json")
        for key, meta in chosen.items()
    }

    fig = plt.figure(figsize=(10.6, 5.3), dpi=220)
    gs = fig.add_gridspec(2, 1, height_ratios=[4.3, 0.36], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax_strip = fig.add_subplot(gs[1], sharex=ax)

    line_styles = {
        "source_only": "--",
        "original_otta": "-",
        "eata": "-.",
        "ours": "-",
    }
    zorders = {
        "source_only": 4,
        "original_otta": 2,
        "eata": 3,
        "ours": 5,
    }
    for key, meta in chosen.items():
        values = np.asarray([float(item["e_star_p"]) for item in histories[key]], dtype=np.float64)
        smooth = moving_average(values, window=window)
        x = np.arange(1, len(values) + 1)
        ax.plot(
            x,
            smooth,
            color=meta["color"],
            linewidth=2.2 if key == "ours" else 1.7,
            linestyle=line_styles[key],
            label=meta["label"],
            alpha=0.95 if key == "ours" else 0.9,
            zorder=zorders[key],
        )

    ax.set_ylabel(r"Moving Avg. $E_p^*$", fontsize=11)
    ax.set_title(f"Figure 3(b)  Online Target-Stream Dynamics ({stream.capitalize()})", fontsize=12)
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.legend(loc="upper right", frameon=True, fontsize=8)

    # Keep the main scale informative and show the collapse spike in an inset.
    ax.set_ylim(0.0, 0.72)

    ref_history = histories["ours"]
    domains = [str(item["domain"]).lower() for item in ref_history]
    domain_to_idx = {domain: idx for idx, domain in enumerate(["sunlamp", "lightbox", "shirt"])}
    strip = np.asarray([[domain_to_idx[d] for d in domains]], dtype=np.float64)
    cmap = plt.matplotlib.colors.ListedColormap(["#4C78A8", "#F58518", "#9E9E9E"])
    ax_strip.imshow(strip, aspect="auto", cmap=cmap, interpolation="nearest", extent=[1, len(domains), 0, 1])
    ax_strip.set_yticks([])
    ax_strip.set_xlabel("Online target-stream step", fontsize=11)
    ax_strip.set_xlim(1, len(domains))
    ax_strip.set_title("Target domain stream", fontsize=9, pad=2)

    boundaries = []
    current = domains[0]
    for idx, domain in enumerate(domains[1:], start=2):
        if domain != current:
            boundaries.append(idx - 0.5)
            current = domain
    for x in boundaries:
        ax.axvline(x, color="black", linestyle="--", linewidth=0.6, alpha=0.55)
        ax_strip.axvline(x, color="white", linestyle="-", linewidth=0.8, alpha=0.9)

    # Domain legend as compact colored patches.
    domain_handles = [
        plt.matplotlib.patches.Patch(color="#4C78A8", label="Sunlamp"),
        plt.matplotlib.patches.Patch(color="#F58518", label="Lightbox"),
        plt.matplotlib.patches.Patch(color="#9E9E9E", label="SHIRT"),
    ]
    ax_strip.legend(handles=domain_handles, loc="center right", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.995, -0.35))

    # Continuous OTTA tail spike inset.
    inset = inset_axes(ax, width="25%", height="34%", loc="upper left", borderpad=1.2)
    cont = np.asarray([float(item["e_star_p"]) for item in histories["original_otta"]], dtype=np.float64)
    cont_smooth = moving_average(cont, window=window)
    x = np.arange(1, len(cont) + 1)
    start = max(1, len(cont) - 700)
    inset.plot(x[start - 1 :], cont_smooth[start - 1 :], color=METHODS["original_otta"]["color"], linewidth=1.4)
    inset.set_title("Continuous tail spike", fontsize=7)
    inset.tick_params(labelsize=6)
    inset.grid(True, linewidth=0.4, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_qualitative(output_path: Path) -> None:
    source_path = OUT_DIR / "qualitative_comparison_main.png"
    image = Image.open(source_path).convert("RGB")
    arr = np.asarray(image)
    h, w = arr.shape[:2]

    # Bottom zoom row from the existing qualitative figure, one panel per method.
    y0 = int(h * 0.43)
    y1 = int(h * 0.885)
    x_edges = [
        (int(w * 0.03), int(w * 0.325)),
        (int(w * 0.345), int(w * 0.645)),
        (int(w * 0.665), int(w * 0.965)),
    ]
    crops = [arr[y0:y1, x0:x1] for x0, x1 in x_edges]
    titles = ["Source-only / No TTA", "Continuous OTTA", "Risk-Controlled-OTTA (Dual-Branch)"]

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9), dpi=220)
    for idx, (ax, crop, title) in enumerate(zip(axes, crops, titles)):
        ax.imshow(crop)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle("Figure 3(c)  Qualitative Keypoint Visualization", fontsize=12, y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_tradeoff(OUT_DIR / "figure3a_stability_adaptation_tradeoff.png")
    plot_dynamics(OUT_DIR / "figure3b_online_target_stream_dynamics.png", stream="forward", window=200)
    plot_qualitative(OUT_DIR / "figure3c_qualitative_keypoint_visualization.png")


if __name__ == "__main__":
    main()


