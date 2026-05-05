from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from data.crop_and_heatmap import load_camera
from risk_controlled_otta.experiments.mixed_domain_stream_eval import (
    load_domain_refs,
    make_method,
    prepare_sample,
    summarize_stream,
)


def parse_common_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="speedplusv2")
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--source_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--collapse_threshold", type=float, default=0.1)

    parser.add_argument("--model_name", type=str, default="vit_base_patch16_dinov3.lvd1689m")
    parser.add_argument("--input_size", type=int, default=384)
    parser.add_argument("--heatmap_size", type=int, default=96)
    parser.add_argument("--heatmap_sigma", type=float, default=3.0)
    parser.add_argument("--mid_channels", type=int, default=256)
    parser.add_argument("--num_deconv_layers", type=int, default=2)
    parser.add_argument("--num_keypoints", type=int, default=11)
    parser.add_argument("--update_scope", type=str, choices=["decoder", "decoder_last_block", "full_model"], default="decoder")

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adapt_steps", type=int, default=1)
    parser.add_argument("--memory_capacity", type=int, default=32)
    parser.add_argument("--memory_sample_size", type=int, default=8)
    parser.add_argument("--min_memory_for_update", type=int, default=4)
    parser.add_argument("--memory_min_quality", type=float, default=0.01)
    parser.add_argument("--lambda_self_training", type=float, default=1.0)
    parser.add_argument("--lambda_geo", type=float, default=0.1)
    parser.add_argument("--lambda_reg", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

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

    parser.add_argument("--trigger_mode", type=str, choices=["threshold", "mlp_geo", "dual_branch", "always", "never"], default="mlp_geo")
    parser.add_argument("--gate_usage", type=str, choices=["hard", "soft_loss", "soft_lr"], default="hard")
    parser.add_argument("--gate_threshold", type=float, default=0.5)
    parser.add_argument("--min_soft_gate_weight", type=float, default=0.05)
    parser.add_argument("--min_lr_gate_scale", type=float, default=0.05)
    parser.add_argument("--gate_hidden_dim", type=int, default=16)
    parser.add_argument("--gate_lr", type=float, default=1e-3)
    parser.add_argument("--gate_weight_decay", type=float, default=0.0)
    parser.add_argument("--gate_warmup_steps", type=int, default=128)
    parser.add_argument("--train_gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prototype_batch_size", type=int, default=32)
    parser.add_argument("--prototype_max_samples", type=int, default=256)
    parser.add_argument("--feature_reprojection_cap", type=float, default=50.0)
    parser.add_argument("--feature_tvec_norm_cap", type=float, default=20.0)

    parser.add_argument("--original_lr", type=float, default=1e-5)
    parser.add_argument("--original_memory_capacity", type=int, default=16)
    parser.add_argument("--original_memory_sample_size", type=int, default=16)
    parser.add_argument("--mask_ratio", type=float, default=0.8)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--ema_alpha", type=float, default=0.999)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()
    args.max_samples = None if args.max_samples <= 0 else int(args.max_samples)
    return args


def load_single_domain_refs(data_root: Path, domain: str, seed: int, shuffle: bool, max_samples: int | None) -> List:
    refs = load_domain_refs(data_root, domain)
    if shuffle:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(refs))
        refs = [refs[int(index)] for index in order]
    if max_samples is not None and max_samples > 0:
        refs = refs[: int(max_samples)]
    return refs


def save_outputs(output_dir: Path, summary: Dict[str, object], records: List[Dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (output_dir / "step_history.json").open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)


def run_single_domain_history(args: argparse.Namespace, method_name: str) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    data_root = Path(args.data_root)
    output_root = Path(args.output_dir)
    refs = load_single_domain_refs(data_root, args.domain, args.seed, bool(args.shuffle), args.max_samples)
    camera_matrix, dist_coeffs = load_camera(data_root)
    method = make_method(method_name, args, device, camera_matrix, dist_coeffs)

    records: List[Dict[str, object]] = []
    desc = f"{method_name}_{args.domain.lower()}"
    for step, sample_ref in enumerate(tqdm(refs, desc=desc), start=1):
        sample = prepare_sample(sample_ref, camera_matrix, dist_coeffs, args.input_size)
        record = method.process_sample(sample, step)
        records.append(record)

    summary = summarize_stream(records, collapse_threshold=args.collapse_threshold)
    summary.update(
        {
            "method": method_name,
            "domain": args.domain.lower(),
            "source_checkpoint": args.source_checkpoint,
            "num_samples_requested": None if args.max_samples is None else int(args.max_samples),
            "num_samples_emitted": int(len(records)),
            "shuffle": bool(args.shuffle),
            "seed": int(args.seed),
            "trigger_mode": args.trigger_mode if method_name in {"learnable_trigger", "ours"} else None,
            "gate_usage": args.gate_usage if method_name in {"learnable_trigger", "ours"} else None,
            "update_scope": args.update_scope if method_name != "source_only" else None,
        }
    )

    save_outputs(output_root, summary, records)
    print(json.dumps(summary, indent=2))
    return {"summary": summary, "records": records}


