from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main(args) -> None:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    min_confidences = parse_float_list(args.min_confidences)
    top_ks = parse_int_list(args.top_ks)
    reproj_errors = parse_float_list(args.ransac_reproj_errors)
    subpixel_radii = parse_int_list(args.subpixel_radii)

    rows = []
    for min_confidence, top_k, reproj_error, subpixel_radius in itertools.product(
        min_confidences,
        top_ks,
        reproj_errors,
        subpixel_radii,
    ):
        run_name = f"mc{min_confidence:g}_top{top_k}_rp{reproj_error:g}_sr{subpixel_radius}"
        run_dir = output_root / run_name
        command = [
            sys.executable,
            "-m",
            "dinov2_heatmap_otta.eval.evaluate_shirt_dino_heatmap",
            "--data_root",
            args.data_root,
            "--model_path",
            args.model_path,
            "--roe",
            args.roe,
            "--domain",
            args.domain,
            "--split",
            args.split,
            "--val_ratio",
            str(args.val_ratio),
            "--seed",
            str(args.seed),
            "--output_dir",
            str(run_dir),
            "--input_size",
            str(args.input_size),
            "--heatmap_size",
            str(args.heatmap_size),
            "--expand_ratio",
            str(args.expand_ratio),
            "--num_vis",
            str(args.num_vis),
            "--min_confidence",
            str(min_confidence),
            "--top_k",
            str(top_k),
            "--min_points",
            str(args.min_points),
            "--ransac_reproj_error",
            str(reproj_error),
            "--ransac_iterations",
            str(args.ransac_iterations),
            "--ransac_confidence",
            str(args.ransac_confidence),
            "--subpixel_radius",
            str(subpixel_radius),
        ]

        if args.disable_nms:
            command.append("--disable_nms")
        if args.disable_subpixel:
            command.append("--disable_subpixel")
        if args.disable_iterative_refine:
            command.append("--disable_iterative_refine")
        if args.no_cuda:
            command.append("--no_cuda")

        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)

        result_path = run_dir / f"{args.split}_results.json"
        with result_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)

        metrics.update(
            {
                "min_confidence": min_confidence,
                "top_k": top_k,
                "ransac_reproj_error": reproj_error,
                "subpixel_radius": subpixel_radius,
                "run_dir": str(run_dir),
            }
        )
        rows.append(metrics)

    rows = sorted(
        rows,
        key=lambda row: (
            row.get("avg_eq_deg", float("inf")),
            row.get("avg_et", float("inf")),
            -row.get("success_ratio", 0.0),
        ),
    )

    with (output_root / "sweep_results.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    print("Top results:")
    for row in rows[: min(10, len(rows))]:
        print(
            "top_k={top_k} min_conf={min_confidence} reproj={ransac_reproj_error} "
            "subpix={subpixel_radius} avg_et={avg_et:.6f} avg_eq_deg={avg_eq_deg:.6f} "
            "success={success_ratio:.3f} inliers={avg_num_ransac_inliers:.3f}".format(**row)
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="SHIRT_Dataset")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--roe", type=str, choices=["roe1", "roe2", "all"], default="all")
    parser.add_argument("--domain", type=str, default="synthetic")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "all"])
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output_root", type=str, default="evaluation_results_shirt_dino_heatmap/sweep")
    parser.add_argument("--input_size", type=int, default=384)
    parser.add_argument("--heatmap_size", type=int, default=96)
    parser.add_argument("--expand_ratio", type=float, default=1.25)

    parser.add_argument("--min_confidences", type=str, default="0.03,0.05,0.08,0.10")
    parser.add_argument("--top_ks", type=str, default="6,7,8,9,10,11")
    parser.add_argument("--ransac_reproj_errors", type=str, default="3,4,5,6,8")
    parser.add_argument("--subpixel_radii", type=str, default="1,2,3")

    parser.add_argument("--min_points", type=int, default=6)
    parser.add_argument("--ransac_iterations", type=int, default=100)
    parser.add_argument("--ransac_confidence", type=float, default=0.999)

    parser.add_argument("--num_vis", type=int, default=0)
    parser.add_argument("--disable_nms", action="store_true")
    parser.add_argument("--disable_subpixel", action="store_true")
    parser.add_argument("--disable_iterative_refine", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())