from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.crop_and_heatmap import load_camera
from risk_controlled_otta.experiments.external_tta_methods.cotta_mixed_stream import CoTTAHeatmapMethod

from table2_redo_external_baselines.common import resolve_device, run_single_domain


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="speedplusv2")
    parser.add_argument("--domain", choices=["sunlamp", "lightbox", "shirt"], required=True)
    parser.add_argument("--source_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--collapse_threshold", type=float, default=0.1)

    parser.add_argument("--input_size", type=int, default=384)
    parser.add_argument("--heatmap_size", type=int, default=96)
    parser.add_argument("--update_scope", choices=["decoder", "decoder_last_block", "full_model"], default="decoder")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--anchor_conf_threshold", type=float, default=0.60)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--restore_prob", type=float, default=0.01)
    parser.add_argument("--aug_times", type=int, default=8)
    parser.add_argument("--aug_noise_std", type=float, default=0.03)

    parser.add_argument("--trigger_confidence", type=float, default=0.15)
    parser.add_argument("--trigger_min_inliers", type=int, default=5)
    parser.add_argument("--trigger_reprojection_error", type=float, default=8.0)
    parser.add_argument("--quality_reprojection_cap", type=float, default=50.0)
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
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()
    args.max_samples = None if args.max_samples <= 0 else int(args.max_samples)
    return args


def main():
    args = parse_args()
    device = resolve_device(args.no_cuda)
    camera_matrix, dist_coeffs = load_camera(Path(args.data_root))
    method = CoTTAHeatmapMethod(args, device, camera_matrix, dist_coeffs)
    result = run_single_domain(args=args, method=method, method_name="cotta")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()


