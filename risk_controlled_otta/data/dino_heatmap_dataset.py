from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.crop_and_heatmap import (
    SPEEDPLUS_3D_KEYPOINTS,
    apply_source_augmentation,
    build_source_augmentation,
    compute_expanded_bbox,
    crop_and_resize,
    load_camera,
    normalize_image,
    project_keypoints,
    visible_keypoints_mask,
)


def generate_gaussian_heatmap_from_crop_coords(
    keypoints: np.ndarray,
    visible: np.ndarray,
    input_size: int = 384,
    heatmap_size: int = 48,
    sigma: float = 3.0,
) -> torch.Tensor:
    num_keypoints = keypoints.shape[0]
    yy, xx = np.meshgrid(np.arange(heatmap_size), np.arange(heatmap_size), indexing="ij")
    heatmaps = np.zeros((num_keypoints, heatmap_size, heatmap_size), dtype=np.float32)

    scale = float(heatmap_size) / float(input_size)
    for keypoint_index in range(num_keypoints):
        if visible[keypoint_index] <= 0:
            continue
        x = keypoints[keypoint_index, 0] * scale
        y = keypoints[keypoint_index, 1] * scale
        heatmaps[keypoint_index] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma ** 2))

    return torch.from_numpy(heatmaps)


class SpeedPlusDinoHeatmapDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        input_size: int = 384,
        heatmap_size: int = 96,
        heatmap_sigma: float = 3.0,
        bbox_expand_ratio: float = 1.25,
        use_source_augmentation: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.input_size = int(input_size)
        self.heatmap_size = int(heatmap_size)
        self.heatmap_sigma = float(heatmap_sigma)
        self.bbox_expand_ratio = float(bbox_expand_ratio)
        self.use_source_augmentation = use_source_augmentation
        self.source_augmentation = build_source_augmentation() if use_source_augmentation else None

        self.camera_matrix, self.dist_coeffs = load_camera(self.data_root)
        self.annotations = self._load_annotations()
        self.image_dir = self._resolve_image_dir()

    def _load_annotations(self) -> List[Dict]:
        if self.split == "train":
            annotation_path = self.data_root / "synthetic" / "train.json"
        elif self.split == "validation":
            annotation_path = self.data_root / "synthetic" / "validation.json"
        elif self.split in {"sunlamp", "sunlamp_test"}:
            annotation_path = self.data_root / "sunlamp" / "test.json"
        elif self.split in {"lightbox", "lightbox_test"}:
            annotation_path = self.data_root / "lightbox" / "test.json"
        elif self.split in {"shirt", "shirt_test"}:
            annotation_path = self.data_root / "shirt" / "test.json"
        else:
            raise ValueError(f"Unsupported split: {self.split}")

        with annotation_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else data["images"]

    def _resolve_image_dir(self) -> Path:
        if self.split in {"train", "validation"}:
            return self.data_root / "synthetic" / "images"
        if self.split in {"sunlamp", "sunlamp_test"}:
            return self.data_root / "sunlamp" / "images"
        if self.split in {"shirt", "shirt_test"}:
            return self.data_root / "shirt" / "images"
        return self.data_root / "lightbox" / "images"

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        annotation = self.annotations[index]
        image_path = self.image_dir / annotation["filename"]
        image = np.array(Image.open(image_path).convert("RGB"))

        quaternion = np.asarray(annotation["q_vbs2tango_true"], dtype=np.float32)
        translation = np.asarray(annotation["r_Vo2To_vbs_true"], dtype=np.float32)
        keypoints_2d = project_keypoints(
            points_3d=SPEEDPLUS_3D_KEYPOINTS,
            quaternion=quaternion,
            translation=translation,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
        )
        visibility = visible_keypoints_mask(keypoints_2d, image_size=(image.shape[1], image.shape[0]))
        bbox = compute_expanded_bbox(
            keypoints_2d=keypoints_2d,
            visible=visibility,
            image_size=(image.shape[1], image.shape[0]),
            expand_ratio=self.bbox_expand_ratio,
        )
        crop_image, crop_keypoints = crop_and_resize(
            image=image,
            keypoints_2d=keypoints_2d,
            bbox=bbox,
            output_size=self.input_size,
        )

        if self.use_source_augmentation:
            crop_image = apply_source_augmentation(crop_image, self.source_augmentation)

        heatmap = generate_gaussian_heatmap_from_crop_coords(
            keypoints=crop_keypoints,
            visible=visibility,
            input_size=self.input_size,
            heatmap_size=self.heatmap_size,
            sigma=self.heatmap_sigma,
        )

        return {
            "image": normalize_image(crop_image).float(),
            "image_uint8": torch.from_numpy(crop_image.copy()),
            "heatmap": heatmap.float(),
            "keypoints_2d": torch.from_numpy(crop_keypoints).float(),
            "visibility": torch.from_numpy(visibility).float(),
            "bbox": torch.tensor(bbox, dtype=torch.float32),
            "quaternion": torch.from_numpy(quaternion).float(),
            "translation": torch.from_numpy(translation).float(),
            "image_name": annotation["filename"],
        }

