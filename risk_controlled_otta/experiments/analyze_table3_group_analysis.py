from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


DEFAULT_METHODS = [
    ("Continuous OTTA (Original)", "output/step_history_original_otta_{domain}/step_history.json"),
    ("CoTTA", "output/table2_redo_external/cotta_{domain}/step_history.json"),
    ("EATA", "output/table2_redo_external/eata_{domain}/step_history.json"),
    ("Risk-Controlled-OTTA (Threshold)", "output/step_history_ours_threshold_{domain}/step_history.json"),
    ("Risk-Controlled-OTTA (Learnable-Geo)", "output/step_history_ours_mlp_geo_{domain}/step_history.json"),
    ("Risk-Controlled-OTTA (Dual-Branch)", "output/step_history_ours_dual_{domain}/step_history.json"),
    ("LTTA", "output/step_history_strict_ltta_{domain}/step_history.json"),
    ("PeTTA", "output/step_history_strict_petta_{domain}/step_history.json"),
    ("Hybrid-TTA-lite", "output/step_history_strict_hybrid_{domain}/step_history.json"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Easy/Medium/Hard group analysis (Table 3).")
    parser.add_argument("--project_root", type=str, default="D:/code")
    parser.add_argument("--domains", nargs="+", default=["sunlamp", "lightbox"])
    parser.add_argument("--source_history", type=str, default="output/step_history_source_only_{domain}/step_history.json")
    parser.add_argument("--easy_threshold", type=float, default=0.1)
    parser.add_argument("--hard_threshold", type=float, default=0.3)
    parser.add_argument("--skip_missing", action="store_true", default=True)
    parser.add_argument("--output_dir", type=str, default="output/analysis_table3_group")
    return parser.parse_args()


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


def to_record_map(records: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    return {str(item["image_name"]): item for item in records}


def group_name(e_star_p: float, easy_threshold: float, hard_threshold: float) -> str:
    if e_star_p <= easy_threshold:
        return "Easy"
    if e_star_p <= hard_threshold:
        return "Medium"
    return "Hard"


def compute_method_row(
    source_records: List[Dict[str, object]],
    method_records: List[Dict[str, object]],
    easy_threshold: float,
    hard_threshold: float,
) -> Dict[str, object]:
    method_map = to_record_map(method_records)
    groups: Dict[str, List[float]] = {"Easy": [], "Medium": [], "Hard": []}
    harmful_num = 0
    harmful_den = 0
    matched = 0

    for source in source_records:
        image_name = str(source["image_name"])
        target = method_map.get(image_name)
        if target is None:
            continue
        matched += 1
        src_ep = float(source["e_star_p"])
        tgt_ep = float(target["e_star_p"])
        delta = tgt_ep - src_ep
        groups[group_name(src_ep, easy_threshold, hard_threshold)].append(delta)
        if bool(target.get("adapted", False)):
            harmful_den += 1
            if delta > 0:
                harmful_num += 1

    if matched == 0:
        raise ValueError("No overlapping samples between source-only and method histories.")

    return {
        "num_matched": matched,
        "easy_delta_ep": float(np.mean(groups["Easy"])) if groups["Easy"] else 0.0,
        "medium_delta_ep": float(np.mean(groups["Medium"])) if groups["Medium"] else 0.0,
        "hard_delta_ep": float(np.mean(groups["Hard"])) if groups["Hard"] else 0.0,
        "harmful_update_rate": float(harmful_num / harmful_den) if harmful_den > 0 else 0.0,
        "group_counts": {key: int(len(value)) for key, value in groups.items()},
        "num_updated": int(harmful_den),
    }


def format_signed(value: float) -> str:
    return f"{value:+.3f}"


def build_latex_rows(domain: str, rows: List[Tuple[str, Dict[str, object]]]) -> str:
    lines = []
    for method_name, metrics in rows:
        lines.append(
            f"{method_name} & {format_signed(metrics['easy_delta_ep'])} & "
            f"{format_signed(metrics['medium_delta_ep'])} & "
            f"{format_signed(metrics['hard_delta_ep'])} & "
            f"{metrics['harmful_update_rate']:.3f} \\\\"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, object] = {
        "easy_threshold": float(args.easy_threshold),
        "hard_threshold": float(args.hard_threshold),
        "domains": {},
    }

    for domain in [item.lower() for item in args.domains]:
        source_path = project_root / args.source_history.format(domain=domain)
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source-only step history for {domain}: {source_path}")

        source_records = load_step_history(source_path)
        domain_rows: List[Tuple[str, Dict[str, object]]] = []
        missing_methods: List[Dict[str, str]] = []

        for method_name, rel_path in DEFAULT_METHODS:
            method_path = project_root / rel_path.format(domain=domain)
            if not method_path.is_file():
                missing_methods.append({"method": method_name, "path": str(method_path)})
                if args.skip_missing:
                    continue
                raise FileNotFoundError(f"Missing method history for {domain}: {method_path}")

            metrics = compute_method_row(
                source_records=source_records,
                method_records=load_step_history(method_path),
                easy_threshold=args.easy_threshold,
                hard_threshold=args.hard_threshold,
            )
            domain_rows.append((method_name, metrics))

        latex_rows = build_latex_rows(domain, domain_rows)
        all_results["domains"][domain] = {
            "source_history": str(source_path),
            "missing_methods": missing_methods,
            "rows": [
                {"method": method_name, **metrics}
                for method_name, metrics in domain_rows
            ],
            "latex_rows": latex_rows,
        }

        with (output_dir / f"{domain}_table3_group_analysis.json").open("w", encoding="utf-8") as handle:
            json.dump(all_results["domains"][domain], handle, indent=2)
        with (output_dir / f"{domain}_table3_group_analysis.tex").open("w", encoding="utf-8") as handle:
            handle.write(latex_rows + "\n")

        print(f"\n=== {domain.upper()} ===")
        if missing_methods:
            print("Missing methods:")
            for item in missing_methods:
                print(f"  - {item['method']}: {item['path']}")
        print(latex_rows)

    with (output_dir / "table3_group_analysis_all_domains.json").open("w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2)


if __name__ == "__main__":
    main()


