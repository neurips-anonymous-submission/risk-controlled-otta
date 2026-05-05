from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
DOMAIN_COLORS = {
    "sunlamp": "#4C78A8",
    "lightbox": "#F58518",
    "shirt": "#9E9E9E",
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


def load_summary(method_key: str, stream: str) -> dict:
    return load_json(METHODS[method_key]["summary_dir"] / stream / "summary.json")


def load_history(method_key: str, stream: str) -> list[dict]:
    return load_json(METHODS[method_key]["summary_dir"] / stream / "step_history.json")


def plot_tradeoff_relative(output_path: Path) -> None:
    order = ["source_only", "eata", "ours", "cotta", "original_otta"]
    labels = [METHODS[key]["label"] for key in order]
    colors = [METHODS[key]["color"] for key in order]

    stability_values: dict[str, list[float]] = {}
    adapt_values: dict[str, list[float]] = {}
    for method_key in order:
        stability_values[method_key] = []
        adapt_values[method_key] = []
        for stream in STREAMS:
            source_summary = load_summary("source_only", stream)
            summary = load_summary(method_key, stream)
            stability_values[method_key].append(
                float(summary["p95_e_star_p"]) - float(source_summary["p95_e_star_p"])
            )
            adapt_values[method_key].append(float(summary["adapt_ratio"]))

    stability_mean, stability_low, stability_high = [], [], []
    adapt_mean, adapt_low, adapt_high = [], [], []
    for method_key in order:
        vals = np.asarray(stability_values[method_key], dtype=np.float64)
        mean = float(vals.mean())
        stability_mean.append(mean)
        stability_low.append(mean - float(vals.min()))
        stability_high.append(float(vals.max()) - mean)

        vals = np.asarray(adapt_values[method_key], dtype=np.float64)
        mean = float(vals.mean())
        adapt_mean.append(mean)
        adapt_low.append(mean - float(vals.min()))
        adapt_high.append(float(vals.max()) - mean)

    x = np.arange(len(order), dtype=np.float64)
    width = 0.68
    edgecolors = ["white"] * len(order)
    linew = [1.0] * len(order)
    alphas = [0.82] * len(order)
    ours_idx = order.index("ours")
    alphas[ours_idx] = 0.98
    linew[ours_idx] = 1.3

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(8.8, 6.4),
        dpi=220,
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 0.9], "hspace": 0.12},
    )

    err_kw = dict(ecolor="#555555", elinewidth=1.0, capsize=3, capthick=1.0)

    top_bars = ax_top.bar(
        x,
        stability_mean,
        width=width,
        color=colors,
        alpha=0.9,
        edgecolor=edgecolors,
        linewidth=1.0,
        yerr=np.vstack([stability_low, stability_high]),
        error_kw=err_kw,
        zorder=3,
    )
    top_bars[ours_idx].set_alpha(1.0)
    top_bars[ours_idx].set_linewidth(1.3)
    ax_top.axhline(0.0, color="#777777", linestyle="--", linewidth=1.0, alpha=0.8, zorder=1)
    ax_top.set_ylabel(r"Mean $\Delta p95\ E_p^*$" "\nvs. Source-only", fontsize=11)
    ax_top.set_title("Figure 3(a)  Stability--Adaptation Summary", fontsize=12)
    ax_top.grid(True, axis="y", linewidth=0.5, alpha=0.30, zorder=0)
    ax_top.set_axisbelow(True)
    ax_top.set_ylim(-0.06, 0.86)

    bottom_bars = ax_bottom.bar(
        x,
        adapt_mean,
        width=width,
        color=colors,
        alpha=0.9,
        edgecolor=edgecolors,
        linewidth=1.0,
        yerr=np.vstack([adapt_low, adapt_high]),
        error_kw=err_kw,
        zorder=3,
    )
    bottom_bars[ours_idx].set_alpha(1.0)
    bottom_bars[ours_idx].set_linewidth(1.3)
    ax_bottom.set_ylabel("Mean Adaptation Rate", fontsize=11)
    ax_bottom.set_xlabel("Method", fontsize=11)
    ax_bottom.grid(True, axis="y", linewidth=0.5, alpha=0.30, zorder=0)
    ax_bottom.set_axisbelow(True)
    ax_bottom.set_ylim(0.0, 1.08)

    ax_bottom.set_xticks(x, labels, fontsize=10)
    plt.setp(ax_bottom.get_xticklabels(), rotation=0, ha="center")

    for ax in (ax_top, ax_bottom):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_top.text(
        0.01,
        0.96,
        "Error bars show min--max across Forward / Reverse / Cyclic / Shifted.",
        transform=ax_top.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#444444",
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_stream_gap(output_path: Path, stream: str = "cyclic", window: int = 200) -> None:
    histories = {
        key: load_history(key, stream)
        for key in ["source_only", "original_otta", "eata", "ours"]
    }
    x = np.arange(1, len(histories["source_only"]) + 1)
    source_values = np.asarray([float(item["e_star_p"]) for item in histories["source_only"]], dtype=np.float64)
    source_ma = moving_average(source_values, window)

    fig = plt.figure(figsize=(10.6, 4.9), dpi=220)
    gs = fig.add_gridspec(2, 1, height_ratios=[4.2, 0.34], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax_strip = fig.add_subplot(gs[1], sharex=ax)

    for key in ["original_otta", "eata", "ours"]:
        values = np.asarray([float(item["e_star_p"]) for item in histories[key]], dtype=np.float64)
        ma = moving_average(values, window)
        gap = ma - source_ma
        style = {
            "original_otta": ("-", 1.9, 3),
            "eata": ("-.", 1.7, 2),
            "ours": ("-", 2.2, 4),
        }[key]
        ax.plot(
            x,
            gap,
            color=METHODS[key]["color"],
            linestyle=style[0],
            linewidth=style[1],
            label=METHODS[key]["label"],
            zorder=style[2],
        )

    ax.axhline(0.0, color=METHODS["source_only"]["color"], linestyle="--", linewidth=1.6, label=METHODS["source_only"]["label"])
    ax.set_ylabel(r"Moving Avg. $\Delta E_p^*$", fontsize=11)
    ax.set_title(f"Figure 3(b)  Representative Mixed-Domain Stream Dynamics ({stream.capitalize()})", fontsize=12)
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    ax.set_ylim(-0.09, 0.78)

    domains = [str(item["domain"]).lower() for item in histories["source_only"]]
    domain_to_idx = {"sunlamp": 0, "lightbox": 1, "shirt": 2}
    strip = np.asarray([[domain_to_idx[d] for d in domains]], dtype=np.float64)
    cmap = plt.matplotlib.colors.ListedColormap(
        [DOMAIN_COLORS["sunlamp"], DOMAIN_COLORS["lightbox"], DOMAIN_COLORS["shirt"]]
    )
    ax_strip.imshow(strip, aspect="auto", cmap=cmap, interpolation="nearest", extent=[1, len(domains), 0, 1])
    ax_strip.set_yticks([])
    ax_strip.set_xlabel("Online target-stream step", fontsize=11)
    ax_strip.set_title("Target domain stream", fontsize=9, pad=2)

    # Continuous tail inset on the gap curve.
    cont_values = np.asarray([float(item["e_star_p"]) for item in histories["original_otta"]], dtype=np.float64)
    cont_gap = moving_average(cont_values, window) - source_ma
    inset = inset_axes(ax, width="26%", height="36%", loc="upper left", borderpad=1.2)
    start = max(1, len(x) - 700)
    inset.plot(x[start - 1 :], cont_gap[start - 1 :], color=METHODS["original_otta"]["color"], linewidth=1.4)
    inset.axhline(0.0, color="#888888", linestyle="--", linewidth=0.8)
    inset.set_title("Continuous tail spike", fontsize=7)
    inset.tick_params(labelsize=6)
    inset.grid(True, linewidth=0.35, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_stream_heatmap_score(output_path: Path, stream: str = "cyclic", window: int = 200) -> None:
    histories = {
        key: load_history(key, stream)
        for key in ["source_only", "original_otta", "eata", "ours"]
    }
    x = np.arange(1, len(histories["source_only"]) + 1)

    fig = plt.figure(figsize=(10.6, 4.9), dpi=220)
    gs = fig.add_gridspec(2, 1, height_ratios=[4.2, 0.34], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax_strip = fig.add_subplot(gs[1], sharex=ax)

    style = {
        "source_only": ("--", 1.8, 3),
        "original_otta": ("-", 1.8, 1),
        "eata": ("-.", 1.7, 2),
        "ours": ("-", 2.25, 4),
    }
    order = ["source_only", "original_otta", "eata", "ours"]
    for key in order:
        values = np.asarray([float(item["mean_confidence"]) for item in histories[key]], dtype=np.float64)
        ma = moving_average(values, window)
        ax.plot(
            x,
            ma,
            color=METHODS[key]["color"],
            linestyle=style[key][0],
            linewidth=style[key][1],
            label=METHODS[key]["label"],
            zorder=style[key][2],
        )

    ax.set_ylabel("Moving Avg. Mean Keypoint\nHeatmap Score", fontsize=11)
    ax.set_title(f"Figure 3(b-alt)  Mixed-Domain Stream Dynamics ({stream.capitalize()})", fontsize=12)
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.legend(loc="upper right", frameon=True, fontsize=8)

    domains = [str(item["domain"]).lower() for item in histories["source_only"]]
    domain_to_idx = {"sunlamp": 0, "lightbox": 1, "shirt": 2}
    strip = np.asarray([[domain_to_idx[d] for d in domains]], dtype=np.float64)
    cmap = plt.matplotlib.colors.ListedColormap(
        [DOMAIN_COLORS["sunlamp"], DOMAIN_COLORS["lightbox"], DOMAIN_COLORS["shirt"]]
    )
    ax_strip.imshow(strip, aspect="auto", cmap=cmap, interpolation="nearest", extent=[1, len(domains), 0, 1])
    ax_strip.set_yticks([])
    ax_strip.set_xlabel("Online target-stream step", fontsize=11)
    ax_strip.set_title("Target domain stream", fontsize=9, pad=2)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_stream_reprojection_gap(output_path: Path, stream: str = "cyclic", window: int = 200) -> None:
    histories = {
        key: load_history(key, stream)
        for key in ["source_only", "original_otta", "cotta", "eata", "ours"]
    }
    x = np.arange(1, len(histories["source_only"]) + 1)
    source_values = np.asarray([float(item["mean_reprojection_error"]) for item in histories["source_only"]], dtype=np.float64)
    source_ma = moving_average(source_values, window)

    fig = plt.figure(figsize=(10.6, 4.5), dpi=220)
    ax = fig.add_subplot(111)

    style = {
        "original_otta": ("-", 1.9, 1),
        "cotta": (":", 1.9, 2),
        "eata": ("-.", 1.7, 2),
        "ours": ("-", 2.25, 5),
    }
    for key in ["original_otta", "cotta", "eata", "ours"]:
        values = np.asarray([float(item["mean_reprojection_error"]) for item in histories[key]], dtype=np.float64)
        ma = moving_average(values, window)
        gap = ma - source_ma
        ax.plot(
            x,
            gap,
            color=METHODS[key]["color"],
            linestyle=style[key][0],
            linewidth=style[key][1],
            label=METHODS[key]["label"],
            zorder=style[key][2],
        )

    ax.axhline(0.0, color=METHODS["source_only"]["color"], linestyle="--", linewidth=1.6, label=METHODS["source_only"]["label"])
    ax.set_ylabel("Moving Avg. Mean\nReprojection Error Gap", fontsize=11)
    ax.set_title(f"Figure 3(b-reproj)  Geometric Consistency Dynamics ({stream.capitalize()})", fontsize=12)
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    ax.set_xlabel("Online target-stream step", fontsize=11)

    cont_values = np.asarray([float(item["mean_reprojection_error"]) for item in histories["original_otta"]], dtype=np.float64)
    cont_gap = moving_average(cont_values, window) - source_ma
    inset = inset_axes(ax, width="26%", height="36%", loc="upper left", borderpad=1.2)
    start = max(1, len(x) - 700)
    inset.plot(x[start - 1 :], cont_gap[start - 1 :], color=METHODS["original_otta"]["color"], linewidth=1.4)
    inset.axhline(0.0, color="#888888", linestyle="--", linewidth=0.8)
    inset.set_title("Continuous tail gap", fontsize=7)
    inset.tick_params(labelsize=6)
    inset.grid(True, linewidth=0.35, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_hard_segment_reprojection_gap(
    output_path: Path,
    stream: str = "forward",
    start: int = 2400,
    end: int = 3200,
    window: int = 80,
) -> None:
    histories = {
        key: load_history(key, stream)
        for key in ["source_only", "original_otta", "cotta", "eata", "ours"]
    }
    x_full = np.arange(1, len(histories["source_only"]) + 1)
    source_values = np.asarray(
        [float(item["mean_reprojection_error"]) for item in histories["source_only"]],
        dtype=np.float64,
    )
    source_ma = moving_average(source_values, window)

    fig, ax = plt.subplots(figsize=(10.8, 4.8), dpi=220)

    emphasis = {
        "source_only": {"linestyle": "--", "linewidth": 1.8, "alpha": 0.95, "zorder": 4},
        "eata": {"linestyle": "-.", "linewidth": 1.9, "alpha": 0.95, "zorder": 5},
        "ours": {"linestyle": "-", "linewidth": 2.4, "alpha": 1.0, "zorder": 6},
        "cotta": {"linestyle": ":", "linewidth": 1.4, "alpha": 0.55, "zorder": 2},
        "original_otta": {"linestyle": "-", "linewidth": 1.5, "alpha": 0.55, "zorder": 1},
    }

    # Subtle domain shading and switch markers so the figure stays readable
    # without a separate domain strip panel.
    domains = [str(item["domain"]).lower() for item in histories["source_only"]]
    run_start = start
    current_domain = domains[start]
    switch_points: list[int] = []
    for idx in range(start + 1, end):
        if domains[idx] != current_domain:
            ax.axvspan(
                run_start + 1,
                idx,
                color=DOMAIN_COLORS[current_domain],
                alpha=0.06,
                linewidth=0,
                zorder=0,
            )
            mid = (run_start + idx + 1) / 2.0
            ax.text(
                mid,
                0.985,
                current_domain.capitalize(),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8.5,
                color="#444444",
            )
            switch_points.append(idx + 1)
            run_start = idx
            current_domain = domains[idx]
    ax.axvspan(
        run_start + 1,
        end,
        color=DOMAIN_COLORS[current_domain],
        alpha=0.06,
        linewidth=0,
        zorder=0,
    )
    mid = (run_start + end + 1) / 2.0
    ax.text(
        mid,
        0.985,
        current_domain.capitalize(),
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=8.5,
        color="#444444",
    )
    for switch_x in switch_points:
        ax.axvline(switch_x, color="#666666", linestyle="--", linewidth=1.0, alpha=0.7, zorder=0)

    # Source-only is the zero baseline; other methods are shown relative to it.
    ax.axhline(
        0.0,
        color=METHODS["source_only"]["color"],
        linestyle=emphasis["source_only"]["linestyle"],
        linewidth=emphasis["source_only"]["linewidth"],
        alpha=emphasis["source_only"]["alpha"],
        zorder=emphasis["source_only"]["zorder"],
        label=METHODS["source_only"]["label"],
    )

    for key in ["original_otta", "cotta", "eata", "ours"]:
        values = np.asarray(
            [float(item["mean_reprojection_error"]) for item in histories[key]],
            dtype=np.float64,
        )
        ma = moving_average(values, window)
        gap = ma - source_ma
        style = emphasis[key]
        ax.plot(
            x_full[start:end],
            gap[start:end],
            color=METHODS[key]["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
            label=METHODS[key]["label"],
            zorder=style["zorder"],
        )

    ax.set_xlim(start + 1, end)
    ax.set_ylabel("Moving Avg. Mean\nReprojection Error Gap", fontsize=11)
    ax.set_xlabel("Online target-stream step", fontsize=11)
    ax.set_title(
        "Figure 3(b-candidate)  Representative Hard Segment Around a Domain Switch",
        fontsize=12,
    )
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.legend(loc="upper left", frameon=True, fontsize=8)

    # Keep the scale focused on the region that separates Ours from EATA and
    # Source-only; the high-update baselines remain visible but less dominant.
    ax.set_ylim(-3.4, 6.4)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_tradeoff_relative(OUT_DIR / "figure3a_protocol_aggregated_tradeoff_v2.png")
    plot_hard_segment_reprojection_gap(
        OUT_DIR / "figure3b_hard_segment_reprojection_gap_v2.png",
        stream="forward",
        start=2400,
        end=3200,
        window=80,
    )


if __name__ == "__main__":
    main()


