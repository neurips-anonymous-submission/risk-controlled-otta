from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from data.crop_and_heatmap import load_camera
from risk_controlled_otta.adapt.learnable_trigger_single_model_tta_dino_heatmap import (
    FeatureQualityMemoryBank,
    build_geo_features,
    build_source_prototype,
)
from risk_controlled_otta.adapt.triggered_single_model_tta_dino_heatmap import load_source_model
from risk_controlled_otta.experiments.mixed_domain_stream_eval import (
    infer_pose,
    load_domain_refs,
    prepare_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Risk-prediction analysis (Table 1).")
    parser.add_argument("--data_root", type=str, default="speedplusv2")
    parser.add_argument("--domains", nargs="+", default=["sunlamp", "lightbox", "shirt"])
    parser.add_argument("--source_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/analysis_table1_risk")
    parser.add_argument("--label_mode", choices=["source_error", "beneficial_update"], default="source_error")
    parser.add_argument("--epsilon", type=float, default=0.4)
    parser.add_argument("--benefit_delta", type=float, default=0.05)
    parser.add_argument("--benefit_history", type=str, default="output/step_history_ours_dual_{domain}/step_history.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)

    parser.add_argument("--model_name", type=str, default="vit_base_patch16_dinov3.lvd1689m")
    parser.add_argument("--input_size", type=int, default=384)
    parser.add_argument("--heatmap_size", type=int, default=96)
    parser.add_argument("--heatmap_sigma", type=float, default=3.0)
    parser.add_argument("--mid_channels", type=int, default=256)
    parser.add_argument("--num_deconv_layers", type=int, default=2)
    parser.add_argument("--num_keypoints", type=int, default=11)

    parser.add_argument("--memory_capacity", type=int, default=32)
    parser.add_argument("--memory_min_quality", type=float, default=0.01)
    parser.add_argument("--quality_reprojection_cap", type=float, default=50.0)

    parser.add_argument("--trigger_confidence", type=float, default=0.15)
    parser.add_argument("--trigger_min_inliers", type=int, default=5)
    parser.add_argument("--trigger_reprojection_error", type=float, default=8.0)
    parser.add_argument("--feature_reprojection_cap", type=float, default=50.0)
    parser.add_argument("--feature_tvec_norm_cap", type=float, default=20.0)

    parser.add_argument("--prototype_batch_size", type=int, default=32)
    parser.add_argument("--prototype_max_samples", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_cuda", action="store_true")

    parser.add_argument("--nms_kernel", type=int, default=3)
    parser.add_argument("--disable_nms", action="store_true")
    parser.add_argument("--disable_subpixel", action="store_true")
    parser.add_argument("--subpixel_radius", type=int, default=2)
    parser.add_argument("--min_confidence", type=float, default=0.05)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--min_points", type=int, default=6)
    parser.add_argument("--ransac_reproj_error", type=float, default=6.0)
    parser.add_argument("--ransac_iterations", type=int, default=100)
    parser.add_argument("--ransac_confidence", type=float, default=0.999)
    parser.add_argument("--disable_iterative_refine", action="store_true")

    parser.add_argument("--trigger_mode", type=str, default="dual_branch")
    args = parser.parse_args()
    args.max_samples = None if args.max_samples <= 0 else int(args.max_samples)
    return args


def compute_quality(mean_conf: float, inlier_ratio: float, mean_reproj: float, reproj_cap: float) -> float:
    reproj_quality = 1.0 / (1.0 + min(float(mean_reproj), float(reproj_cap)))
    return float(mean_conf) * float(inlier_ratio) * reproj_quality


def evaluate_signal(X: np.ndarray, y: np.ndarray, folds: int, seed: int) -> Dict[str, float]:
    positives = int(y.sum())
    negatives = int((1 - y).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("Risk labels contain only one class; cannot compute AUROC/AUPRC.")

    splits = max(2, min(int(folds), positives, negatives))
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    aucs: List[float] = []
    aprs: List[float] = []

    for train_idx, val_idx in cv.split(X, y):
        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(max_iter=1000)),
            ]
        )
        clf.fit(X[train_idx], y[train_idx])
        prob = clf.predict_proba(X[val_idx])[:, 1]
        aucs.append(float(roc_auc_score(y[val_idx], prob)))
        aprs.append(float(average_precision_score(y[val_idx], prob)))

    return {
        "auroc_mean": float(np.mean(aucs)),
        "auroc_std": float(np.std(aucs)),
        "auprc_mean": float(np.mean(aprs)),
        "auprc_std": float(np.std(aprs)),
        "num_folds": int(splits),
    }


def records_to_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_step_history(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("steps"), list):
            return data["steps"]
        if isinstance(data.get("history"), list):
            return data["history"]
    raise ValueError(f"Unsupported step-history format: {path}")


def analyze_domain(args, domain: str, model, device: torch.device, camera_matrix, dist_coeffs) -> Dict[str, object]:
    refs = load_domain_refs(Path(args.data_root), domain)
    if args.max_samples is not None:
        refs = refs[: int(args.max_samples)]

    source_prototype = build_source_prototype(model, args, device)
    memory_bank = FeatureQualityMemoryBank(capacity=args.memory_capacity)
    rows: List[Dict[str, object]] = []
    benefit_map: Dict[str, Dict[str, object]] = {}

    if args.label_mode == "beneficial_update":
        benefit_path = Path(args.benefit_history.format(domain=domain))
        if not benefit_path.is_file():
            raise FileNotFoundError(f"Missing benefit-history file for {domain}: {benefit_path}")
        benefit_rows = load_step_history(benefit_path)
        benefit_map = {str(item["image_name"]): item for item in benefit_rows}

    for step, sample_ref in enumerate(tqdm(refs, desc=f"risk_{domain}"), start=1):
        sample = prepare_sample(sample_ref, camera_matrix, dist_coeffs, args.input_size)
        heatmap, cls_token, diagnosis, metrics = infer_pose(
            model=model,
            sample=sample,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            args=args,
            device=device,
            return_features=True,
        )
        geo_features = build_geo_features(diagnosis, args, device).squeeze(0)
        feat_features = memory_bank.feature_distances(cls_token.squeeze(0), source_prototype)

        mean_conf = float(diagnosis.get("mean_confidence", 0.0))
        inlier_ratio = float(diagnosis.get("inlier_ratio", 0.0))
        mean_reproj = float(diagnosis.get("mean_reprojection_error", 0.0))
        quality = compute_quality(mean_conf, inlier_ratio, mean_reproj, args.quality_reprojection_cap)
        e_star_p = float(metrics["e_star_p"])

        if args.label_mode == "source_error":
            high_risk = int(e_star_p > float(args.epsilon))
            benefit_ep = None
            benefit_gain = None
        else:
            benefit_row = benefit_map.get(str(sample["image_name"]))
            if benefit_row is None:
                raise KeyError(f"Missing aligned benefit-history sample for {sample['image_name']}")
            benefit_ep = float(benefit_row["e_star_p"])
            benefit_gain = e_star_p - benefit_ep
            high_risk = int(benefit_gain > float(args.benefit_delta))

        rows.append(
            {
                "step": int(step),
                "image_name": str(sample["image_name"]),
                "e_star_p": e_star_p,
                "high_risk": high_risk,
                "mean_confidence": mean_conf,
                "num_ransac_inliers": int(diagnosis.get("num_ransac_inliers", 0)),
                "inlier_ratio": inlier_ratio,
                "mean_reprojection_error": mean_reproj,
                "max_reprojection_error": float(diagnosis.get("max_reprojection_error", mean_reproj)),
                "used_fallback_epnp": int(bool(diagnosis.get("used_fallback_epnp", False))),
                "pose_failed": int(bool(diagnosis.get("rvec") is None or diagnosis.get("tvec") is None)),
                "quality": quality,
                "benefit_target_e_star_p": benefit_ep,
                "benefit_gain": benefit_gain,
                **{f"z_{index}": float(value) for index, value in enumerate(geo_features.detach().cpu().tolist())},
                **{f"d_{index}": float(value) for index, value in enumerate(feat_features.detach().cpu().tolist())},
            }
        )

        if quality >= args.memory_min_quality:
            memory_bank.push(
                image=sample["image"].squeeze(0).cpu(),
                pseudo_heatmap=heatmap.squeeze(0).detach().cpu(),
                feature=cls_token.squeeze(0).detach().cpu(),
                quality=quality,
                image_name=str(sample["image_name"]),
            )

    y = np.asarray([int(item["high_risk"]) for item in rows], dtype=np.int64)
    x_conf = np.asarray([[float(item["mean_confidence"])] for item in rows], dtype=np.float64)
    x_geo = np.asarray([[float(item[f"z_{index}"]) for index in range(8)] for item in rows], dtype=np.float64)
    x_dual = np.asarray(
        [[float(item[f"z_{index}"]) for index in range(8)] + [float(item[f"d_{index}"]) for index in range(3)] for item in rows],
        dtype=np.float64,
    )

    signal_results = {
        "Prediction-level: mean peak score c_t only": evaluate_signal(x_conf, y, args.folds, args.seed),
        "Geometry-only: z_t": evaluate_signal(x_geo, y, args.folds, args.seed),
        "Dual-branch: (z_t, d_t)": evaluate_signal(x_dual, y, args.folds, args.seed),
        "Oracle (using ground-truth pose error)": {
            "auroc_mean": 1.0,
            "auroc_std": 0.0,
            "auprc_mean": 1.0,
            "auprc_std": 0.0,
            "num_folds": 0,
        },
    }

    return {
        "domain": domain,
        "label_mode": args.label_mode,
        "epsilon": float(args.epsilon),
        "benefit_delta": float(args.benefit_delta),
        "num_samples": int(len(rows)),
        "num_high_risk": int(y.sum()),
        "positive_ratio": float(y.mean()),
        "signals": signal_results,
        "rows": rows,
    }


def build_latex(domain_results: Dict[str, Dict[str, object]]) -> str:
    order = [
        "Prediction-level: mean peak score c_t only",
        "Geometry-only: z_t",
        "Dual-branch: (z_t, d_t)",
        "Oracle (using ground-truth pose error)",
    ]
    domains = list(domain_results.keys())
    lines = []
    for signal_name in order:
        parts = [signal_name]
        for domain in domains:
            metrics = domain_results[domain]["signals"][signal_name]
            parts.append(f"{metrics['auroc_mean']:.3f}")
            parts.append(f"{metrics['auprc_mean']:.3f}")
        lines.append(" & ".join(parts) + r" \\")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    camera_matrix, dist_coeffs = load_camera(Path(args.data_root))
    model = load_source_model(args.source_checkpoint, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, object] = {
        "label_mode": args.label_mode,
        "epsilon": float(args.epsilon),
        "benefit_delta": float(args.benefit_delta),
        "folds": int(args.folds),
        "domains": {},
    }

    for domain in [item.lower() for item in args.domains]:
        result = analyze_domain(args, domain, model, device, camera_matrix, dist_coeffs)
        rows = result.pop("rows")
        records_to_csv(output_dir / f"{domain}_risk_records.csv", rows)
        with (output_dir / f"{domain}_risk_analysis.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        all_results["domains"][domain] = result

    latex = build_latex(all_results["domains"])
    with (output_dir / "table1_risk_prediction_analysis.tex").open("w", encoding="utf-8") as handle:
        handle.write(latex + "\n")
    with (output_dir / "table1_risk_prediction_analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2)

    print("\n=== TABLE 1 LATEX ROWS ===")
    print(latex)


if __name__ == "__main__":
    main()


