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
            "risk_controlled_otta.eval.evaluate_dino_heatmap",
            "--data_root",
            args.data_root,
            "--model_path",
            args.model_path,
            "--split",
            args.split,
            "--output_dir",
            str(run_dir),
            "--num_vis",
            str(args.num_vis),
            "--min_confidence",
            str(min_confidence),
            "--top_k",
            str(top_k),
            "--ransac_reproj_error",
            str(reproj_error),
            "--subpixel_radius",
            str(subpixel_radius),
        ]
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

    rows = sorted(rows, key=lambda row: (row.get("avg_eq_deg", float("inf")), row.get("avg_et", float("inf"))))
    with (output_root / "sweep_results.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    print("Top results:")
    for row in rows[: min(10, len(rows))]:
        print(
            "top_k={top_k} min_conf={min_confidence} reproj={ransac_reproj_error} "
            "subpix={subpixel_radius} avg_et={avg_et:.6f} avg_eq_deg={avg_eq_deg:.6f} "
            "inliers={avg_num_ransac_inliers:.3f}".format(**row)
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="speedplusv2")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--output_root", type=str, default="evaluation_results_dinov3_heatmap/sweep")
    parser.add_argument("--min_confidences", type=str, default="0.03,0.05,0.08,0.10")
    parser.add_argument("--top_ks", type=str, default="6,7,8,9,10,11")
    parser.add_argument("--ransac_reproj_errors", type=str, default="3,4,5,6,8")
    parser.add_argument("--subpixel_radii", type=str, default="1,2,3")
    parser.add_argument("--num_vis", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())


