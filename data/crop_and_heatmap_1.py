"""
Dataset and preprocessing helpers for the paper-style reproduction.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset


DEFAULT_SPEEDPLUS_3D_KEYPOINTS = np.array([
    [-0.3700, -0.3850, 0.3215],
    [-0.3700, 0.3850, 0.3215],
    [0.3700, 0.3850, 0.3215],
    [0.3700, -0.3850, 0.3215],
    [-0.3700, -0.2640, 0.0000],
    [-0.3700, 0.3040, 0.0000],
    [0.3700, 0.3040, 0.0000],
    [0.3700, -0.2640, 0.0000],
    [-0.5427, 0.4877, 0.2535],
    [0.5427, 0.4877, 0.2591],
    [0.3050, -0.5790, 0.2515],
], dtype=np.float32)


def load_tango_3d_keypoints(mat_path: str | Path | None = None) -> np.ndarray:
    """
    Load official Tango 3D keypoints from a MAT file.

    Expected variable name: ``tango3Dpoints`` with shape [3, 11] or [11, 3].
    Falls back to the baked-in official baseline points when the file is absent.
    """
    if mat_path is None:
        mat_path = Path(__file__).resolve().parents[1] / "tangoPoints.mat"
    else:
        mat_path = Path(mat_path)

    if not mat_path.exists():
        return DEFAULT_SPEEDPLUS_3D_KEYPOINTS.copy()

    vertices = loadmat(mat_path)["tango3Dpoints"]
    vertices = np.asarray(vertices, dtype=np.float32)
    if vertices.shape == (3, 11):
        vertices = vertices.T
    if vertices.shape != (11, 3):
        raise ValueError(f"Unexpected tango3Dpoints shape: {vertices.shape}")
    return vertices


SPEEDPLUS_3D_KEYPOINTS = load_tango_3d_keypoints()


def load_camera(data_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    with (data_root / "camera.json").open("r", encoding="utf-8") as handle:
        camera_data = json.load(handle)
    camera_matrix = np.asarray(camera_data["cameraMatrix"], dtype=np.float32)
    dist_coeffs = np.asarray(camera_data.get("distCoeffs", [0, 0, 0, 0, 0]), dtype=np.float32)
    return camera_matrix, dist_coeffs


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = quaternion.astype(np.float32)
    quaternion = quaternion / np.linalg.norm(quaternion)
    q0, q1, q2, q3 = quaternion

    dcm = np.zeros((3, 3), dtype=np.float32)
    dcm[0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
    dcm[1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
    dcm[2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2

    dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2
    dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1
    dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1
    return dcm


def project_keypoints(
    points_3d: np.ndarray,
    quaternion: np.ndarray,
    translation: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    if points_3d.shape[0] != 3:
        points_3d = points_3d.T

    keypoints = np.vstack((points_3d.astype(np.float64), np.ones((1, points_3d.shape[1]), dtype=np.float64)))

    # Match the official SPEED+ baseline projection chain:
    # pose_mat = [quat2dcm(q_vbs2tango)^T | r_Vo2To_vbs]
    pose_mat = np.hstack((
        quaternion_to_rotation_matrix(quaternion).T.astype(np.float64),
        translation.reshape(3, 1).astype(np.float64),
    ))
    xyz = pose_mat @ keypoints
    x0 = xyz[0, :] / xyz[2, :]
    y0 = xyz[1, :] / xyz[2, :]

    r2 = x0 * x0 + y0 * y0
    cdist = 1 + dist_coeffs[0] * r2 + dist_coeffs[1] * r2 * r2 + dist_coeffs[4] * r2 * r2 * r2
    x = x0 * cdist + dist_coeffs[2] * 2 * x0 * y0 + dist_coeffs[3] * (r2 + 2 * x0 * x0)
    y = y0 * cdist + dist_coeffs[2] * (r2 + 2 * y0 * y0) + dist_coeffs[3] * 2 * x0 * y0

    image_points = np.vstack((
        camera_matrix[0, 0] * x + camera_matrix[0, 2],
        camera_matrix[1, 1] * y + camera_matrix[1, 2],
    )).T
    return image_points.astype(np.float32)


def visible_keypoints_mask(keypoints_2d: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    width, height = image_size
    visible = (
        (keypoints_2d[:, 0] >= 0)
        & (keypoints_2d[:, 0] < width)
        & (keypoints_2d[:, 1] >= 0)
        & (keypoints_2d[:, 1] < height)
    )
    return visible.astype(np.float32)


def compute_expanded_bbox(
    keypoints_2d: np.ndarray,
    visible: np.ndarray,
    image_size: Tuple[int, int],
    expand_ratio: float = 1.25,
) -> Tuple[float, float, float, float]:
    width, height = image_size
    valid = keypoints_2d[visible > 0]
    if len(valid) == 0:
        return 0.0, 0.0, float(width), float(height)

    min_xy = valid.min(axis=0)
    max_xy = valid.max(axis=0)
    center = (min_xy + max_xy) * 0.5
    side = float(max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1], 1.0) * expand_ratio)
    half = side * 0.5

    x1 = max(0.0, center[0] - half)
    y1 = max(0.0, center[1] - half)
    x2 = min(float(width), center[0] + half)
    y2 = min(float(height), center[1] + half)

    if x2 <= x1:
        x2 = min(float(width), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(height), y1 + 1.0)
    return x1, y1, x2, y2


def crop_and_resize(
    image: np.ndarray,
    keypoints_2d: np.ndarray,
    bbox: Tuple[float, float, float, float],
    output_size: int = 384,
) -> Tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = bbox
    crop = image[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))]
    if crop.size == 0:
        crop = image
        x1, y1, x2, y2 = 0.0, 0.0, float(image.shape[1]), float(image.shape[0])

    crop_h, crop_w = crop.shape[:2]
    resized = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)

    keypoints = keypoints_2d.copy()
    keypoints[:, 0] = (keypoints[:, 0] - x1) * output_size / max(crop_w, 1)
    keypoints[:, 1] = (keypoints[:, 1] - y1) * output_size / max(crop_h, 1)
    return resized, keypoints


def apply_source_augmentation(image: np.ndarray) -> np.ndarray:
    augmented = image.copy()
    if random.random() < 0.5:
        try:
            flare_center = (random.randint(0, image.shape[1] - 1), random.randint(0, image.shape[0] - 1))
            augmented = cv2.addWeighted(augmented, 1.0, np.full_like(augmented, 50), 0.15, 0)
            cv2.circle(augmented, flare_center, random.randint(20, 100), (255, 255, 255), -1)
        except ValueError:
            pass
    if random.random() < 0.5:
        yuv = cv2.cvtColor(augmented, cv2.COLOR_RGB2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        augmented = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    if random.random() < 0.5:
        augmented = cv2.GaussianBlur(augmented, (5, 5), 0)
    if random.random() < 0.5:
        alpha = random.uniform(0.8, 1.2)
        beta = random.uniform(-20, 20)
        augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=beta)
    if random.random() < 0.5:
        noise = np.random.normal(loc=0.0, scale=8.0, size=augmented.shape).astype(np.float32)
        augmented = np.clip(augmented.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return augmented


def normalize_image(image: np.ndarray) -> torch.Tensor:
    image = image.astype(np.float32) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return torch.from_numpy(image.transpose(2, 0, 1))


def generate_gaussian_heatmap(
    keypoints: np.ndarray,
    visible: np.ndarray,
    heatmap_size: int = 96,
    sigma: float = 3.0,
) -> torch.Tensor:
    num_keypoints = keypoints.shape[0]
    yy, xx = np.meshgrid(np.arange(heatmap_size), np.arange(heatmap_size), indexing="ij")
    heatmaps = np.zeros((num_keypoints, heatmap_size, heatmap_size), dtype=np.float32)

    for keypoint_index in range(num_keypoints):
        if visible[keypoint_index] <= 0:
            continue
        x = keypoints[keypoint_index, 0] / 384.0 * heatmap_size
        y = keypoints[keypoint_index, 1] / 384.0 * heatmap_size
        heatmaps[keypoint_index] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))

    return torch.from_numpy(heatmaps)


class SpeedPlusCropHeatmapDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        input_size: int = 384,
        heatmap_size: int = 96,
        bbox_expand_ratio: float = 1.25,
        use_augmentation: bool = False,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.bbox_expand_ratio = bbox_expand_ratio
        self.use_augmentation = use_augmentation

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
        visible = visible_keypoints_mask(keypoints_2d, image_size=(image.shape[1], image.shape[0]))
        bbox = compute_expanded_bbox(
            keypoints_2d=keypoints_2d,
            visible=visible,
            image_size=(image.shape[1], image.shape[0]),
            expand_ratio=self.bbox_expand_ratio,
        )
        cropped_image, cropped_keypoints = crop_and_resize(
            image=image,
            keypoints_2d=keypoints_2d,
            bbox=bbox,
            output_size=self.input_size,
        )

        if self.use_augmentation:
            cropped_image = apply_source_augmentation(cropped_image)

        heatmap = generate_gaussian_heatmap(
            keypoints=cropped_keypoints,
            visible=visible,
            heatmap_size=self.heatmap_size,
            sigma=3.0,
        )

        return {
            "image": normalize_image(cropped_image),
            "heatmap": heatmap.float(),
            "keypoints_2d": torch.from_numpy(cropped_keypoints).float(),
            "visibility": torch.from_numpy(visible).float(),
            "bbox": torch.tensor(bbox, dtype=torch.float32),
            "quaternion": torch.from_numpy(quaternion).float(),
            "translation": torch.from_numpy(translation).float(),
            "image_name": annotation["filename"],
        }

