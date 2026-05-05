from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from risk_controlled_otta.experiments.mixed_domain_stream_eval import save_stream_outputs, summarize_stream
from risk_controlled_otta.experiments.strict_mixed_domain_stream_eval import (
    load_domain_refs,
    make_method,
    prepare_raw_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate single-domain step history for strict LTTA / PeTTA / Hybrid-TTA-lite.")
    parser.add_argument("--data_root", type=str, default="speedplusv2")
    parser.add_argument("--source_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--method", type=str, choices=["strict_ltta", "strict_petta", "strict_hybrid_tta_lite"], required=True)
    parser.add_argument("--domain", type=str, choices=["sunlamp", "lightbox", "shirt"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--collapse_threshold", type=float, default=0.1)

    parser.add_argument("--model_name", type=str, default="vit_base_patch16_dinov3.lvd1689m")
    parser.add_argument("--input_size", type=int, default=384)
    parser.add_argument("--heatmap_size", type=int, default=96)
    parser.add_argument("--heatmap_sigma", type=float, default=3.0)
    parser.add_argument("--num_keypoints", type=int, default=11)
    parser.add_argument("--mid_channels", type=int, default=256)
    parser.add_argument("--num_deconv_layers", type=int, default=2)

    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adapt_steps", type=int, default=1)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lambda_entropy", type=float, default=1.0)
    parser.add_argument("--lambda_confidence", type=float, default=0.05)
    parser.add_argument("--lambda_dwt", type=float, default=0.1)
    parser.add_argument("--pseudo_bbox_min_confidence", type=float, default=0.05)
    parser.add_argument("--pseudo_bbox_expand_ratio", type=float, default=1.50)
    parser.add_argument("--pseudo_bbox_min_size", type=float, default=96.0)
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

    parser.add_argument("--update_scope", type=str, default="stem")

    parser.add_argument("--normal_update_scope", type=str, default="stem")
    parser.add_argument("--guarded_update_scope", type=str, default="decoder")
    parser.add_argument("--guarded_lr", type=float, default=1e-6)
    parser.add_argument("--guarded_weight_decay", type=float, default=0.0)
    parser.add_argument("--guarded_adapt_steps", type=int, default=1)
    parser.add_argument("--lambda_mim", type=float, default=0.25)
    parser.add_argument("--lambda_anchor", type=float, default=0.01)
    parser.add_argument("--mim_mask_ratio", type=float, default=0.35)
    parser.add_argument("--mim_patch_size", type=int, default=32)
    parser.add_argument("--teacher_momentum", type=float, default=0.999)
    parser.add_argument("--guarded_grad_clip_norm", type=float, default=0.5)
    parser.add_argument("--petta_window_size", type=int, default=32)
    parser.add_argument("--petta_warmup_steps", type=int, default=8)
    parser.add_argument("--petta_threshold", type=float, default=0.75)
    parser.add_argument("--petta_freeze_threshold", type=float, default=1.25)
    parser.add_argument("--petta_quality_weight", type=float, default=1.0)
    parser.add_argument("--petta_confidence_weight", type=float, default=0.5)
    parser.add_argument("--petta_entropy_weight", type=float, default=0.25)
    parser.add_argument("--petta_reprojection_weight", type=float, default=0.5)
    parser.add_argument("--petta_inlier_weight", type=float, default=0.5)
    parser.add_argument("--petta_fallback_weight", type=float, default=0.5)
    parser.add_argument("--petta_min_inlier_ratio", type=float, default=0.5)
    parser.add_argument("--soft_reset_momentum", type=float, default=0.25)
    parser.add_argument("--reset_to_last_stable", action="store_true")
    parser.add_argument("--use_ema_teacher", action="store_true")

    parser.add_argument("--efficient_update_scope", type=str, default="stem")
    parser.add_argument("--full_update_scope", type=str, default="stem_decoder")
    parser.add_argument("--full_lr", type=float, default=2e-6)
    parser.add_argument("--full_weight_decay", type=float, default=0.0)
    parser.add_argument("--full_grad_clip_norm", type=float, default=0.5)
    parser.add_argument("--ddsd_window_size", type=int, default=32)
    parser.add_argument("--ddsd_warmup_steps", type=int, default=8)
    parser.add_argument("--ddsd_threshold", type=float, default=0.75)
    parser.add_argument("--ddsd_cooldown_steps", type=int, default=0)
    parser.add_argument("--ddsd_quality_weight", type=float, default=1.0)
    parser.add_argument("--ddsd_confidence_weight", type=float, default=0.5)
    parser.add_argument("--ddsd_entropy_weight", type=float, default=0.25)
    parser.add_argument("--ddsd_reprojection_weight", type=float, default=0.5)
    parser.add_argument("--ddsd_inlier_weight", type=float, default=0.5)
    parser.add_argument("--ddsd_min_inlier_ratio", type=float, default=0.5)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()
    args.max_samples = None if args.max_samples <= 0 else int(args.max_samples)
    return args


def maybe_trim_refs(refs, seed: int, max_samples: int | None):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(refs)) if refs else []
    refs = [refs[int(index)] for index in order] if refs else []
    if max_samples is not None:
        refs = refs[: int(max_samples)]
    return refs


def run_single_domain(args: argparse.Namespace) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    refs = load_domain_refs(Path(args.data_root), args.domain)
    refs = maybe_trim_refs(refs, args.seed, args.max_samples)
    method = make_method(args.method, args, device, None, None)

    records: List[Dict[str, object]] = []
    for step, sample_ref in enumerate(tqdm(refs, desc=f"{args.method}_{args.domain}"), start=1):
        sample = prepare_raw_sample(sample_ref)
        records.append(method.process_sample(sample, step))

    summary = summarize_stream(records, collapse_threshold=args.collapse_threshold)
    summary.update(
        {
            "method": args.method,
            "domain": args.domain.lower(),
            "seed": int(args.seed),
            "source_checkpoint": args.source_checkpoint,
        }
    )
    result = {"summary": summary, "records": records}
    save_stream_outputs(Path(args.output_dir), result)
    return result


def main() -> None:
    args = parse_args()
    result = run_single_domain(args)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()


