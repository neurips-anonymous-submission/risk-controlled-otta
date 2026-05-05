from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def load_annotations(json_path: Path) -> list[dict]:
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected list annotations in {json_path}")
    return data


def merge_shirt_domain(input_root: Path, output_root: Path) -> None:
    image_out = output_root / "images"
    ensure_dir(image_out)

    merged: list[dict] = []
    for roe_name in ("roe1", "roe2"):
        roe_root = input_root / roe_name
        ann_path = roe_root / f"{roe_name}.json"
        image_root = roe_root / "lightbox" / "images"
        if not ann_path.is_file():
            raise FileNotFoundError(f"Missing annotation file: {ann_path}")
        if not image_root.is_dir():
            raise FileNotFoundError(f"Missing image directory: {image_root}")

        annotations = load_annotations(ann_path)
        for item in annotations:
            original_name = str(item["filename"])
            merged_name = f"{roe_name}_{original_name}"
            src = image_root / original_name
            dst = image_out / merged_name
            if not src.is_file():
                raise FileNotFoundError(f"Missing image referenced by {ann_path}: {src}")
            safe_link_or_copy(src, dst)

            merged_item = dict(item)
            merged_item["filename"] = merged_name
            merged.append(merged_item)

    ensure_dir(output_root)
    with (output_root / "test.json").open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, default="shirtv1")
    parser.add_argument("--output_root", type=str, default="speedplusv2/shirt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_shirt_domain(Path(args.input_root), Path(args.output_root))
    print(
        json.dumps(
            {
                "input_root": str(Path(args.input_root)),
                "output_root": str(Path(args.output_root)),
                "status": "ok",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

