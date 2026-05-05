from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


METHOD_STYLES = {
    "no_tta": {"label": "No TTA", "color": "#2ca02c", "marker": "o", "linestyle": "--"},
    "source_only": {"label": "Source-only/No TTA", "color": "#2ca02c", "marker": "o", "linestyle": "--"},
    "original_otta": {"label": "Continuous OTTA", "color": "#ff7f0e", "marker": "^", "linestyle": "-."},
    "cotta": {"label": "CoTTA", "color": "#9467bd", "marker": "^", "linestyle": "-."},
    "eata": {"label": "EATA", "color": "#8c564b", "marker": "D", "linestyle": ":"},
    "ours": {"label": "Risk-Controlled-OTTA (Dual-Branch)", "color": "#1f4aff", "marker": "x", "linestyle": "-"},
    "mlp_geo": {"label": "Learnable-Geo", "color": "#ff4d4d", "marker": "P", "linestyle": "--"},
    "dual_branch": {"label": "Risk-Controlled-OTTA (Dual-Branch)", "color": "#1f4aff", "marker": "x", "linestyle": "-"},
    "triggered_single": {"label": "Triggered Single", "color": "#ff7f0e", "marker": "X", "linestyle": "--"},
    "strict_ltta": {"label": "L-TTA", "color": "#bcbd22", "marker": "v", "linestyle": "--"},
    "strict_petta": {"label": "PeTTA", "color": "#ff7f0e", "marker": "^", "linestyle": "-."},
    "strict_hybrid": {"label": "Hybrid-TTA-lite", "color": "#d62728", "marker": "P", "linestyle": ":"},
}


def finite_series(values: Sequence[float], fill_percentile: float = 99.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite_mask = np.isfinite(arr)
    if finite_mask.any():
        cap = float(np.percentile(arr[finite_mask], fill_percentile))
    else:
        cap = 1.0
    arr[~finite_mask] = cap
    return arr


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


def get_numeric_series(history: Sequence[dict], key: str) -> np.ndarray | None:
    values: List[float] = []
    for item in history:
        if key not in item:
            return None
        try:
            values.append(float(item.get(key, 0.0)))
        except (TypeError, ValueError):
            values.append(float("nan"))
    return np.asarray(values, dtype=np.float64)


def compute_geometric_reliability(
    history: Sequence[dict],
    reprojection_cap: float,
    include_confidence: bool,
) -> np.ndarray:
    mean_conf = get_numeric_series(history, "mean_confidence")
    inlier_ratio = get_numeric_series(history, "inlier_ratio")
    mean_reproj = get_numeric_series(history, "mean_reprojection_error")
    if mean_conf is None or inlier_ratio is None or mean_reproj is None:
        missing = [
            key
            for key, value in {
                "mean_confidence": mean_conf,
                "inlier_ratio": inlier_ratio,
                "mean_reprojection_error": mean_reproj,
            }.items()
            if value is None
        ]
        raise KeyError(f"Required geometric keys missing from history: {missing}")

    reproj_term = 1.0 + np.minimum(mean_reproj, float(reprojection_cap))
    if include_confidence:
        reliability = mean_conf * inlier_ratio / reproj_term
        return finite_series(reliability)
    reliability = inlier_ratio / reproj_term
    return finite_series(reliability)


def pick_metric(history: Sequence[dict], requested_key: str, reprojection_cap: float) -> tuple[np.ndarray, str, str]:
    if requested_key == "inlier_ratio_proxy":
        inlier_ratio = get_numeric_series(history, "inlier_ratio")
        if inlier_ratio is None:
            raise KeyError("Metric key 'inlier_ratio' was not found in every record.")
        proxy = 1.0 - np.clip(inlier_ratio, 0.0, 1.0)
        return finite_series(proxy), "inlier_ratio_proxy", "1 - inlier ratio"

    if requested_key == "geo_inlier_over_reproj":
        return (
            compute_geometric_reliability(history, reprojection_cap=reprojection_cap, include_confidence=False),
            "geo_inlier_over_reproj",
            "Geometric reliability",
        )

    if requested_key == "geo_conf_inlier_over_reproj":
        return (
            compute_geometric_reliability(history, reprojection_cap=reprojection_cap, include_confidence=True),
            "geo_conf_inlier_over_reproj",
            "Confidence-weighted geometric reliability",
        )

    if requested_key == "no_fallback":
        fallback = get_numeric_series(history, "used_fallback_epnp")
        if fallback is None:
            raise KeyError("Metric key 'used_fallback_epnp' was not found in every record.")
        reliability = 1.0 - np.clip(fallback, 0.0, 1.0)
        return finite_series(reliability), "no_fallback", "PnP success reliability"

    if requested_key == "quality_proxy":
        quality = get_numeric_series(history, "quality")
        if quality is None:
            raise KeyError("Metric key 'quality' was not found in every record.")
        finite_quality = quality[np.isfinite(quality)]
        if finite_quality.size == 0:
            raise ValueError("Metric key 'quality' exists but has no finite values.")
        normalizer = float(np.percentile(finite_quality, 95))
        if normalizer <= 1e-12:
            normalizer = float(max(finite_quality.max(), 1.0))
        proxy = 1.0 - np.clip(quality / normalizer, 0.0, 1.0)
        return finite_series(proxy), "quality_proxy", "1 - normalized quality"

    if requested_key == "quality_like":
        return (
            compute_geometric_reliability(history, reprojection_cap=8.0, include_confidence=True),
            "quality_like",
            "Keypoint localization reliability",
        )

    if requested_key != "auto":
        series = get_numeric_series(history, requested_key)
        if series is None:
            raise KeyError(f"Metric key '{requested_key}' was not found in every record.")
        return finite_series(series), requested_key, requested_key

    e_star_p = get_numeric_series(history, "e_star_p")
    if e_star_p is not None and np.isfinite(e_star_p).any():
        return finite_series(e_star_p), "e_star_p", "E_p*"

    reproj = get_numeric_series(history, "mean_reprojection_error")
    if reproj is not None and np.isfinite(reproj).any():
        return finite_series(reproj), "mean_reprojection_error", "Mean Reprojection Error"

    quality = get_numeric_series(history, "quality")
    if quality is not None and np.isfinite(quality).any():
        finite_quality = quality[np.isfinite(quality)]
        normalizer = float(np.percentile(finite_quality, 95))
        if normalizer <= 1e-12:
            normalizer = float(max(finite_quality.max(), 1.0))
        proxy = 1.0 - np.clip(quality / normalizer, 0.0, 1.0)
        return finite_series(proxy), "quality_proxy", "1 - normalized quality"

    inlier_ratio = get_numeric_series(history, "inlier_ratio")
    if inlier_ratio is not None and np.isfinite(inlier_ratio).any():
        proxy = 1.0 - np.clip(inlier_ratio, 0.0, 1.0)
        return finite_series(proxy), "inlier_ratio_proxy", "1 - inlier ratio"

    raise ValueError(
        "Could not infer a plottable metric. Tried e_star_p, mean_reprojection_error, quality, and inlier_ratio."
    )


def rolling_stats(values: np.ndarray, window: int, lower_q: float, upper_q: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if window <= 1:
        return values.copy(), values.copy(), values.copy()
    means = np.zeros_like(values)
    lowers = np.zeros_like(values)
    uppers = np.zeros_like(values)
    for idx in range(len(values)):
        left = max(0, idx - window + 1)
        chunk = values[left : idx + 1]
        means[idx] = chunk.mean()
        lowers[idx] = np.percentile(chunk, lower_q)
        uppers[idx] = np.percentile(chunk, upper_q)
    return means, lowers, uppers


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    window = int(window)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def load_series(path: Path, metric_key: str, collapse_threshold: float, reprojection_cap: float) -> Dict[str, np.ndarray]:
    history = load_records(path)
    raw, metric_source_key, metric_source_label = pick_metric(history, metric_key, reprojection_cap=reprojection_cap)
    adapted = np.asarray([1.0 if item.get("adapted", False) else 0.0 for item in history], dtype=np.float64)
    triggered = np.asarray(
        [
            1.0
            if item.get("triggered", item.get("heuristic_triggered", False))
            else 0.0
            for item in history
        ],
        dtype=np.float64,
    )
    if all("collapse" in item for item in history):
        collapsed = np.asarray([1.0 if item.get("collapse", False) else 0.0 for item in history], dtype=np.float64)
    elif metric_source_key == "e_star_p":
        collapsed = np.asarray([1.0 if value > collapse_threshold else 0.0 for value in raw], dtype=np.float64)
    else:
        collapsed = np.zeros_like(raw, dtype=np.float64)
    steps = np.arange(1, len(raw) + 1, dtype=np.int32)
    domains = [str(item.get("domain", "")).lower() for item in history]
    return {
        "steps": steps,
        "metric": raw,
        "adapted": adapted,
        "triggered": triggered,
        "collapsed": collapsed,
        "domains": domains,
        "metric_source_key": np.asarray([metric_source_key] * len(raw), dtype=object),
        "metric_source_label": np.asarray([metric_source_label] * len(raw), dtype=object),
    }


def plot_curves(
    curves: Dict[str, Dict[str, np.ndarray]],
    metric_label: str,
    output_path: Path,
    rolling_window: int,
    domain_block: int | None,
    auto_domain_boundaries: bool,
    highlight_method: str,
    label_overrides: Dict[str, str],
    band_alpha_scale: float,
    start_step: int,
    band_lower_q: float,
    band_upper_q: float,
    line_smooth_window: int,
    marker_every: int,
    marker_size: float,
    marker_alpha: float,
    line_width_main: float,
    line_width_other: float,
    auto_tight_ylim: bool,
    tight_ylim_pad: float,
    y_min: float | None,
    y_max: float | None,
    legend_labelspacing: float,
    legend_handlelength: float,
    legend_borderpad: float,
    annotate_gap_to_highlight: bool,
    gap_annotation_window: int,
    gap_decimals: int,
    right_label_pad_ratio: float,
    gap_use_last_point: bool,
    gap_font_size: float,
    gap_line_width: float,
    gap_dot_size: float,
    annotate_end_values: bool,
    end_value_decimals: int,
    end_value_font_size: float,
    end_value_line_width: float,
    end_value_dot_size: float,
    end_value_right_pad_ratio: float,
    end_value_min_sep_ratio: float,
    end_value_min_stem_ratio: float,
) -> None:
    plt.figure(figsize=(11, 5.8))

    y_curve_min = float("inf")
    y_curve_max = float("-inf")
    plotted_curves: Dict[str, Dict[str, np.ndarray | dict | str]] = {}
    for method_name, data in curves.items():
        style = METHOD_STYLES.get(method_name, {"label": method_name, "color": None, "marker": None, "linestyle": "-"})
        legend_label = label_overrides.get(method_name) or style["label"]
        is_highlight = bool(highlight_method) and method_name == highlight_method
        line_alpha = 1.0 if is_highlight else 0.82
        fill_alpha = (0.20 if is_highlight else 0.10) * max(0.0, float(band_alpha_scale))
        line_width = float(line_width_main if is_highlight else line_width_other)
        mean_curve, lower_curve, upper_curve = rolling_stats(
            data["metric"],
            window=rolling_window,
            lower_q=band_lower_q,
            upper_q=band_upper_q,
        )
        mean_curve = smooth_series(mean_curve, line_smooth_window)
        lower_curve = smooth_series(lower_curve, max(1, line_smooth_window // 2))
        upper_curve = smooth_series(upper_curve, max(1, line_smooth_window // 2))
        start_index = max(0, int(start_step) - 1)
        steps = data["steps"][start_index:]
        mean_curve = mean_curve[start_index:]
        lower_curve = lower_curve[start_index:]
        upper_curve = upper_curve[start_index:]
        y_curve_min = min(y_curve_min, float(mean_curve.min()))
        y_curve_max = max(y_curve_max, float(mean_curve.max()))
        if fill_alpha > 0:
            plt.fill_between(
                steps,
                lower_curve,
                upper_curve,
                alpha=fill_alpha,
                color=style["color"],
                linewidth=0,
            )
        plt.plot(
            steps,
            mean_curve,
            color=style["color"],
            linewidth=line_width,
            alpha=line_alpha,
            linestyle=style.get("linestyle", "-"),
            label=legend_label,
            marker=style.get("marker"),
            markevery=max(1, int(marker_every)) if int(marker_every) > 0 else None,
            markersize=float(marker_size),
            markerfacecolor="white" if not is_highlight else style["color"],
            markeredgecolor=style["color"],
            markeredgewidth=1.8 if style.get("marker") in {"x", "X", "+"} else 1.0,
            dash_capstyle="round",
            solid_capstyle="round",
            zorder=4 if is_highlight else 3,
        )
        plotted_curves[method_name] = {
            "steps": steps,
            "mean_curve": mean_curve,
            "style": style,
            "label": legend_label,
        }

        collapse_idx = np.where(data["collapsed"][start_index:] > 0.5)[0]
        if len(collapse_idx) > 0:
            sampled = collapse_idx[:: max(1, len(collapse_idx) // 35)]
            plt.scatter(
                steps[sampled],
                mean_curve[sampled],
                s=10,
                color=style["color"],
                alpha=0.9,
                marker="x",
            )

    if domain_block is not None and domain_block > 0:
        total_steps = max(len(data["steps"]) for data in curves.values())
        for boundary in range(domain_block, total_steps, domain_block):
            plt.axvline(boundary, color="k", linestyle="--", linewidth=1.0, alpha=0.7)
    elif auto_domain_boundaries:
        first_domains = next(iter(curves.values())).get("domains", [])
        for idx in range(1, len(first_domains)):
            if first_domains[idx] and first_domains[idx] != first_domains[idx - 1]:
                plt.axvline(idx + 1, color="k", linestyle="--", linewidth=1.0, alpha=0.55)

    plt.xlabel("Test-time adaptation step", fontsize=16)
    plt.ylabel(metric_label, fontsize=18)
    if auto_tight_ylim and np.isfinite(y_curve_min) and np.isfinite(y_curve_max):
        pad = max((y_curve_max - y_curve_min) * float(tight_ylim_pad), 1e-4)
        plt.ylim(y_curve_min - pad, y_curve_max + pad)
    if y_min is not None or y_max is not None:
        current_min, current_max = plt.ylim()
        plt.ylim(current_min if y_min is None else y_min, current_max if y_max is None else y_max)
    if (
        annotate_gap_to_highlight
        and highlight_method
        and highlight_method in plotted_curves
        and len(plotted_curves) > 1
    ):
        highlight_curve = plotted_curves[highlight_method]
        highlight_steps = np.asarray(highlight_curve["steps"], dtype=np.float64)
        highlight_values = np.asarray(highlight_curve["mean_curve"], dtype=np.float64)
        last_n = max(1, min(int(gap_annotation_window), len(highlight_values)))
        if gap_use_last_point:
            highlight_y = float(highlight_values[-1])
        else:
            highlight_y = float(np.mean(highlight_values[-last_n:]))
        x_start = float(highlight_steps[-1])
        x_min = float(min(np.asarray(item["steps"], dtype=np.float64)[0] for item in plotted_curves.values()))
        x_span = max(x_start - x_min, 1.0)
        x_anchor = x_start + x_span * float(right_label_pad_ratio)
        x_text = x_anchor + x_span * max(float(right_label_pad_ratio) * 0.55, 0.035)
        plt.xlim(x_min, x_text + x_span * 0.02)

        ours_style = highlight_curve["style"]
        plt.scatter(
            [x_start],
            [highlight_y],
            s=float(gap_dot_size),
            color=ours_style["color"],
            marker=ours_style.get("marker", "o"),
            zorder=5,
        )

        for method_name, item in plotted_curves.items():
            if method_name == highlight_method:
                continue
            comp_values = np.asarray(item["mean_curve"], dtype=np.float64)
            if gap_use_last_point:
                comp_y = float(comp_values[-1])
            else:
                comp_y = float(np.mean(comp_values[-last_n:]))
            comp_style = item["style"]
            color = comp_style["color"]
            delta = highlight_y - comp_y

            plt.plot(
                [x_start, x_anchor],
                [comp_y, comp_y],
                color=color,
                linewidth=float(gap_line_width) * 0.9,
                linestyle=":",
                alpha=0.8,
                zorder=2,
            )
            plt.plot(
                [x_anchor, x_anchor],
                [min(comp_y, highlight_y), max(comp_y, highlight_y)],
                color=color,
                linewidth=float(gap_line_width),
                linestyle="-",
                alpha=0.82,
                zorder=2,
            )
            plt.text(
                x_text,
                0.5 * (comp_y + highlight_y),
                f"+{delta:.{int(gap_decimals)}f}",
                color=color,
                fontsize=float(gap_font_size),
                fontweight="semibold",
                va="center",
                ha="left",
            )

        plt.text(
            x_text,
            highlight_y,
            "vs ours",
            color=ours_style["color"],
            fontsize=float(gap_font_size),
            fontweight="semibold",
            va="bottom",
            ha="left",
        )

    if annotate_end_values and plotted_curves:
        x_min = float(min(np.asarray(item["steps"], dtype=np.float64)[0] for item in plotted_curves.values()))
        x_last = float(max(np.asarray(item["steps"], dtype=np.float64)[-1] for item in plotted_curves.values()))
        x_span = max(x_last - x_min, 1.0)
        x_anchor = x_last + x_span * float(end_value_right_pad_ratio)
        x_text = x_anchor + x_span * max(float(end_value_right_pad_ratio) * 0.45, 0.030)

        ylim_now = plt.ylim()
        y_low, y_high = float(ylim_now[0]), float(ylim_now[1])
        y_span = max(y_high - y_low, 1e-6)
        min_sep = y_span * float(end_value_min_sep_ratio)
        endpoints = []
        for method_name, item in plotted_curves.items():
            mean_curve = np.asarray(item["mean_curve"], dtype=np.float64)
            steps = np.asarray(item["steps"], dtype=np.float64)
            endpoints.append(
                {
                    "method_name": method_name,
                    "x": float(steps[-1]),
                    "y": float(mean_curve[-1]),
                    "style": item["style"],
                }
            )

        endpoints.sort(key=lambda d: d["y"])
        adjusted_ys = []
        current_y = None
        for endpoint in endpoints:
            desired_y = endpoint["y"]
            if current_y is None:
                current_y = desired_y
            else:
                current_y = max(desired_y, current_y + min_sep)
            adjusted_ys.append(current_y)

        top_margin = y_span * 0.04
        bottom_margin = y_span * 0.04
        if adjusted_ys:
            overflow = adjusted_ys[-1] - (y_high - top_margin)
            if overflow > 0:
                adjusted_ys = [y - overflow for y in adjusted_ys]
            underflow = (y_low + bottom_margin) - adjusted_ys[0]
            if underflow > 0:
                adjusted_ys = [y + underflow for y in adjusted_ys]
            for idx in range(1, len(adjusted_ys)):
                adjusted_ys[idx] = max(adjusted_ys[idx], adjusted_ys[idx - 1] + min_sep)
            for idx in range(len(adjusted_ys) - 2, -1, -1):
                adjusted_ys[idx] = min(adjusted_ys[idx], adjusted_ys[idx + 1] - min_sep)

        plt.xlim(x_min, x_text + x_span * 0.02)

        guide_low = min(adjusted_ys) - y_span * 0.008
        guide_high = max(adjusted_ys) + y_span * 0.008
        plt.plot(
            [x_anchor, x_anchor],
            [guide_low, guide_high],
            color="#6e6e6e",
            linewidth=0.95,
            linestyle="-",
            alpha=0.85,
            zorder=1,
        )

        for endpoint, y_text in zip(endpoints, adjusted_ys):
            color = endpoint["style"]["color"]
            marker = endpoint["style"].get("marker", "o")
            plt.scatter(
                [endpoint["x"]],
                [endpoint["y"]],
                s=float(end_value_dot_size),
                color=color,
                marker=marker,
                zorder=5,
            )
            plt.plot(
                [endpoint["x"], x_anchor],
                [endpoint["y"], endpoint["y"]],
                color=color,
                linewidth=float(end_value_line_width) * 0.9,
                linestyle=":",
                alpha=0.78,
                zorder=2,
            )
            plt.text(
                x_text,
                y_text,
                f"{endpoint['y']:.{int(end_value_decimals)}f}",
                color=color,
                fontsize=float(end_value_font_size),
                fontweight="semibold",
                va="center",
                ha="left",
            )

    plt.grid(True, axis="both", linestyle="--", linewidth=1.05, alpha=0.52)
    plt.legend(
        frameon=False,
        fontsize=14,
        loc="upper left",
        labelspacing=float(legend_labelspacing),
        handlelength=float(legend_handlelength),
        borderaxespad=float(legend_borderpad),
        handletextpad=0.6,
    )
    plt.tick_params(labelsize=13)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_only_history", type=str, default="")
    parser.add_argument("--original_otta_history", type=str, default="")
    parser.add_argument("--ours_history", type=str, default="")
    parser.add_argument("--cotta_history", type=str, default="")
    parser.add_argument("--eata_history", type=str, default="")
    parser.add_argument("--no_tta_history", type=str, default="")
    parser.add_argument("--mlp_geo_history", type=str, default="")
    parser.add_argument("--dual_branch_history", type=str, default="")
    parser.add_argument("--triggered_single_history", type=str, default="")
    parser.add_argument("--strict_ltta_history", type=str, default="")
    parser.add_argument("--strict_petta_history", type=str, default="")
    parser.add_argument("--strict_hybrid_history", type=str, default="")
    parser.add_argument("--metric_key", type=str, default="auto")
    parser.add_argument("--metric_label", type=str, default="")
    parser.add_argument("--reprojection_cap", type=float, default=50.0)
    parser.add_argument("--collapse_threshold", type=float, default=20.0)
    parser.add_argument("--rolling_window", type=int, default=25)
    parser.add_argument("--domain_block", type=int, default=0)
    parser.add_argument("--auto_domain_boundaries", action="store_true")
    parser.add_argument("--highlight_method", type=str, default="ours")
    parser.add_argument("--band_alpha_scale", type=float, default=1.0)
    parser.add_argument("--band_lower_q", type=float, default=25.0)
    parser.add_argument("--band_upper_q", type=float, default=75.0)
    parser.add_argument("--line_smooth_window", type=int, default=1)
    parser.add_argument("--marker_every", type=int, default=0)
    parser.add_argument("--marker_size", type=float, default=6.0)
    parser.add_argument("--marker_alpha", type=float, default=0.95)
    parser.add_argument("--line_width_main", type=float, default=3.0)
    parser.add_argument("--line_width_other", type=float, default=2.0)
    parser.add_argument("--auto_tight_ylim", action="store_true")
    parser.add_argument("--tight_ylim_pad", type=float, default=0.12)
    parser.add_argument("--y_min", type=float, default=None)
    parser.add_argument("--y_max", type=float, default=None)
    parser.add_argument("--legend_labelspacing", type=float, default=0.35)
    parser.add_argument("--legend_handlelength", type=float, default=2.2)
    parser.add_argument("--legend_borderpad", type=float, default=0.2)
    parser.add_argument("--annotate_gap_to_highlight", action="store_true")
    parser.add_argument("--gap_annotation_window", type=int, default=120)
    parser.add_argument("--gap_decimals", type=int, default=3)
    parser.add_argument("--right_label_pad_ratio", type=float, default=0.035)
    parser.add_argument("--gap_use_last_point", action="store_true")
    parser.add_argument("--gap_font_size", type=float, default=11.5)
    parser.add_argument("--gap_line_width", type=float, default=1.4)
    parser.add_argument("--gap_dot_size", type=float, default=32.0)
    parser.add_argument("--annotate_end_values", action="store_true")
    parser.add_argument("--end_value_decimals", type=int, default=3)
    parser.add_argument("--end_value_font_size", type=float, default=13.0)
    parser.add_argument("--end_value_line_width", type=float, default=1.8)
    parser.add_argument("--end_value_dot_size", type=float, default=40.0)
    parser.add_argument("--end_value_right_pad_ratio", type=float, default=0.040)
    parser.add_argument("--end_value_min_sep_ratio", type=float, default=0.040)
    parser.add_argument("--end_value_min_stem_ratio", type=float, default=0.028)
    parser.add_argument("--start_step", type=int, default=1)
    parser.add_argument("--source_only_label", type=str, default="")
    parser.add_argument("--original_otta_label", type=str, default="")
    parser.add_argument("--ours_label", type=str, default="")
    parser.add_argument("--cotta_label", type=str, default="")
    parser.add_argument("--eata_label", type=str, default="")
    parser.add_argument("--strict_ltta_label", type=str, default="")
    parser.add_argument("--strict_petta_label", type=str, default="")
    parser.add_argument("--strict_hybrid_label", type=str, default="")
    parser.add_argument("--output_path", type=str, default="visualization_results_dinov3_heatmap/motivation_curve.png")
    return parser.parse_args()


def main():
    args = parse_args()
    curves = {}
    for name, path_str in {
        "source_only": args.source_only_history,
        "original_otta": args.original_otta_history,
        "ours": args.ours_history,
        "cotta": args.cotta_history,
        "eata": args.eata_history,
        "no_tta": args.no_tta_history,
        "mlp_geo": args.mlp_geo_history,
        "dual_branch": args.dual_branch_history,
        "triggered_single": args.triggered_single_history,
        "strict_ltta": args.strict_ltta_history,
        "strict_petta": args.strict_petta_history,
        "strict_hybrid": args.strict_hybrid_history,
    }.items():
        if not path_str:
            continue
        path = Path(path_str)
        if not path.is_file():
            raise FileNotFoundError(f"Missing history file: {path}")
        curves[name] = load_series(
            path,
            args.metric_key,
            collapse_threshold=args.collapse_threshold,
            reprojection_cap=args.reprojection_cap,
        )

    if not curves:
        raise ValueError("No input history files were provided.")

    if args.metric_label:
        metric_label = args.metric_label
    else:
        first_curve = next(iter(curves.values()))
        metric_label = f"Testing Error ({first_curve['metric_source_label'][0]})"

    for method_name, data in curves.items():
        source_label = str(data["metric_source_label"][0])
        print(f"[plot] {method_name}: using metric source '{source_label}' from {len(data['metric'])} steps")

    block = None if args.domain_block <= 0 else int(args.domain_block)
    label_overrides = {
        "source_only": str(args.source_only_label or ""),
        "original_otta": str(args.original_otta_label or ""),
        "ours": str(args.ours_label or ""),
        "cotta": str(args.cotta_label or ""),
        "eata": str(args.eata_label or ""),
        "strict_ltta": str(args.strict_ltta_label or ""),
        "strict_petta": str(args.strict_petta_label or ""),
        "strict_hybrid": str(args.strict_hybrid_label or ""),
    }
    plot_curves(
        curves=curves,
        metric_label=metric_label,
        output_path=Path(args.output_path),
        rolling_window=int(args.rolling_window),
        domain_block=block,
        auto_domain_boundaries=bool(args.auto_domain_boundaries),
        highlight_method=str(args.highlight_method or ""),
        label_overrides=label_overrides,
        band_alpha_scale=float(args.band_alpha_scale),
        start_step=int(args.start_step),
        band_lower_q=float(args.band_lower_q),
        band_upper_q=float(args.band_upper_q),
        line_smooth_window=int(args.line_smooth_window),
        marker_every=int(args.marker_every),
        marker_size=float(args.marker_size),
        marker_alpha=float(args.marker_alpha),
        line_width_main=float(args.line_width_main),
        line_width_other=float(args.line_width_other),
        auto_tight_ylim=bool(args.auto_tight_ylim),
        tight_ylim_pad=float(args.tight_ylim_pad),
        y_min=args.y_min,
        y_max=args.y_max,
        legend_labelspacing=float(args.legend_labelspacing),
        legend_handlelength=float(args.legend_handlelength),
        legend_borderpad=float(args.legend_borderpad),
        annotate_gap_to_highlight=bool(args.annotate_gap_to_highlight),
        gap_annotation_window=int(args.gap_annotation_window),
        gap_decimals=int(args.gap_decimals),
        right_label_pad_ratio=float(args.right_label_pad_ratio),
        gap_use_last_point=bool(args.gap_use_last_point),
        gap_font_size=float(args.gap_font_size),
        gap_line_width=float(args.gap_line_width),
        gap_dot_size=float(args.gap_dot_size),
        annotate_end_values=bool(args.annotate_end_values),
        end_value_decimals=int(args.end_value_decimals),
        end_value_font_size=float(args.end_value_font_size),
        end_value_line_width=float(args.end_value_line_width),
        end_value_dot_size=float(args.end_value_dot_size),
        end_value_right_pad_ratio=float(args.end_value_right_pad_ratio),
        end_value_min_sep_ratio=float(args.end_value_min_sep_ratio),
        end_value_min_stem_ratio=float(args.end_value_min_stem_ratio),
    )


if __name__ == "__main__":
    main()


