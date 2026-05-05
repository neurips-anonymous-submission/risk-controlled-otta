from __future__ import annotations

import argparse
import json
from pathlib import Path


DOMAINS = ("sunlamp", "lightbox", "shirt")


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def row(method_label: str, summaries: dict[str, dict]) -> str:
    fields = [method_label]
    for domain in DOMAINS:
        summary = summaries[domain]
        fields.extend(
            [
                f"{summary['p95_e_star_p']:.6f}",
                f"{summary['collapse_rate']:.6f}",
                f"{summary['adapt_ratio']:.6f}",
            ]
        )
    return " & ".join(fields) + r" \\"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Root output dir containing cotta_* and eata_* subdirs.")
    args = parser.parse_args()

    root = Path(args.root)
    cotta = {domain: load_summary(root / f"cotta_{domain}" / "summary.json") for domain in DOMAINS}
    eata = {domain: load_summary(root / f"eata_{domain}" / "summary.json") for domain in DOMAINS}

    print("CoTTA")
    print(row("CoTTA", cotta))
    print()
    print("EATA")
    print(row("EATA", eata))


if __name__ == "__main__":
    main()

