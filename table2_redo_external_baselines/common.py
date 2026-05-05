from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from tqdm import tqdm

from data.crop_and_heatmap import load_camera
from risk_controlled_otta.experiments.mixed_domain_stream_eval import (
    load_domain_refs,
    prepare_sample,
    summarize_stream,
)


def load_single_domain_refs(
    data_root: Path,
    domain: str,
    seed: int,
    shuffle: bool,
    max_samples: int | None,
):
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


def run_single_domain(
    *,
    args,
    method,
    method_name: str,
) -> Dict[str, object]:
    data_root = Path(args.data_root)
    output_root = Path(args.output_dir)
    refs = load_single_domain_refs(data_root, args.domain, args.seed, bool(args.shuffle), args.max_samples)
    camera_matrix, dist_coeffs = load_camera(data_root)

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
            "update_scope": getattr(args, "update_scope", None),
        }
    )
    save_outputs(output_root, summary, records)
    print(json.dumps(summary, indent=2))
    return {"summary": summary, "records": records}


def resolve_device(no_cuda: bool) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")


