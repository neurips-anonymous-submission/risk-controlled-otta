from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_original_module() -> ModuleType:
    source_path = Path(__file__).with_name("ltta_single_model_tta_speed_dino_heatmap (2).py")
    spec = importlib.util.spec_from_file_location("ltta_single_model_tta_speed_dino_heatmap_orig", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load original L-TTA module from {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ORIG = _load_original_module()

for _name in dir(_ORIG):
    if _name.startswith("_"):
        continue
    globals()[_name] = getattr(_ORIG, _name)


if __name__ == "__main__":
    run_ltta_single_model_tta(parse_args())

