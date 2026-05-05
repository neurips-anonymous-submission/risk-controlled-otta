from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.crop_and_heatmap import (
    SPEEDPLUS_3D_KEYPOINTS,
    compute_expanded_bbox,
    crop_and_resize,
    load_camera,
    normalize_image,
    project_keypoints,
    visible_keypoints_mask,
)
from risk_controlled_otta.adapt.learnable_trigger_single_model_tta_dino_heatmap import (
    FeatureQualityMemoryBank,
    adapt_with_gate,
    build_geo_features,
    build_source_prototype as build_learnable_source_prototype,
    choose_gate_decision,
    gate_probability,
    heuristic_risk_label,
    make_gate,
)
from risk_controlled_otta.adapt.online_tta_dino_heatmap import (
    build_source_prototype as build_original_source_prototype,
    initialize_memory_bank,
    random_discontinuous_mask,
    update_ema_teacher,
)
from risk_controlled_otta.adapt.triggered_single_model_tta_dino_heatmap import (
    QualityMemoryBank,
    adapt_single_trigger,
    configure_trainable_parameters,
    diagnose_prediction,
    geometry_target_from_pose,
    load_source_model,
    maybe_push_current_sample,
)
from risk_controlled_otta.data.dino_heatmap_dataset import SpeedPlusDinoHeatmapDataset
from risk_controlled_otta.eval.evaluate_dino_heatmap import (
    compute_metrics,
    rotation_vector_to_quaternion,
)
from risk_controlled_otta.models.dino_pose_model import DinoHeatmapPoseModel
from losses.otta_losses import (
    class_awareness_consistency_loss,
    masked_heatmap_consistency_loss,
    self_training_loss,
    total_target_loss,
)
from memory.dynamic_bank import (
    DynamicMemoryBank,
    heatmap_confidence_score,
    sample_from_memory_bank,
    update_memory_pseudolabels,
)


DOMAIN_ALIASES = {
    "sunlamp": ["sunlamp", "Sunlamp", "SUNLAMP"],
    "lightbox": ["lightbox", "Lightbox", "LIGHTBOX"],
    "shirt": ["shirt", "SHIRT", "Shirt"],
}

STREAM_NAMES = ("forward", "reverse", "cyclic", "shifted")


@dataclass
class DomainSampleRef:
    domain: str
    filename: str
    gt_quaternion: List[float]
    gt_translation: List[float]
    image_path: Path


def finite_float(value, default: float = 0.0, cap: float = 1e6) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default
    if not np.isfinite(result):
        result = cap
    return min(result, cap)


def resolve_domain_dir(data_root: Path, domain: str) -> Path:
    candidates = DOMAIN_ALIASES.get(domain.lower(), [domain, domain.lower(), domain.upper(), domain.capitalize()])
    for candidate in candidates:
        domain_dir = data_root / candidate
        if domain_dir.is_dir():
            return domain_dir
    raise FileNotFoundError(f"Unable to resolve domain directory for {domain!r} under {data_root}.")


def load_domain_refs(data_root: Path, domain: str) -> List[DomainSampleRef]:
    domain_dir = resolve_domain_dir(data_root, domain)
    annotation_path = domain_dir / "test.json"
    image_dir = domain_dir / "images"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing annotation file: {annotation_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    with annotation_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    annotations = data if isinstance(data, list) else data["images"]
    return [
        DomainSampleRef(
            domain=domain.lower(),
            filename=str(annotation["filename"]),
            gt_quaternion=[float(item) for item in annotation["q_vbs2tango_true"]],
            gt_translation=[float(item) for item in annotation["r_Vo2To_vbs_true"]],
            image_path=image_dir / annotation["filename"],
        )
        for annotation in annotations
    ]


def permute_and_trim_domains(
    data_root: Path,
    domains: Sequence[str],
    seed: int,
    max_per_domain: int | None,
) -> tuple[Dict[str, List[DomainSampleRef]], int]:
    rng = np.random.default_rng(seed)
    refs_by_domain: Dict[str, List[DomainSampleRef]] = {}
    for domain in domains:
        refs = load_domain_refs(data_root, domain)
        order = rng.permutation(len(refs))
        refs_by_domain[domain.lower()] = [refs[int(index)] for index in order]

    k = min(len(refs) for refs in refs_by_domain.values())
    if max_per_domain is not None and max_per_domain > 0:
        k = min(k, int(max_per_domain))
    trimmed = {domain: refs[:k] for domain, refs in refs_by_domain.items()}
    return trimmed, int(k)


def build_stream(samples_by_domain: Dict[str, List[DomainSampleRef]], domains: Sequence[str], stream_name: str) -> List[DomainSampleRef]:
    normalized_domains = [domain.lower() for domain in domains]
    if len(normalized_domains) != 3:
        raise ValueError(f"Expected exactly 3 domains, got {normalized_domains}")
    a, b, c = normalized_domains
    s_a = samples_by_domain[a]
    s_b = samples_by_domain[b]
    s_c = samples_by_domain[c]

    if stream_name == "forward":
        return [*s_a, *s_b, *s_c]
    if stream_name == "reverse":
        return [*s_c, *s_b, *s_a]
    if stream_name == "cyclic":
        return [item for triplet in zip(s_a, s_b, s_c) for item in triplet]
    if stream_name == "shifted":
        return [item for triplet in zip(s_b, s_c, s_a) for item in triplet]
    raise ValueError(f"Unsupported stream name: {stream_name}")


def prepare_sample(
    sample_ref: DomainSampleRef,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    input_size: int,
) -> Dict[str, object]:
    image = np.array(Image.open(sample_ref.image_path).convert("RGB"))
    gt_quaternion = np.asarray(sample_ref.gt_quaternion, dtype=np.float64)
    gt_translation = np.asarray(sample_ref.gt_translation, dtype=np.float64)

    gt_keypoints_2d = project_keypoints(
        SPEEDPLUS_3D_KEYPOINTS,
        gt_quaternion.astype(np.float32),
        gt_translation.astype(np.float32),
        camera_matrix,
        dist_coeffs,
    )
    visible = visible_keypoints_mask(gt_keypoints_2d, (image.shape[1], image.shape[0]))
    bbox = compute_expanded_bbox(gt_keypoints_2d, visible, (image.shape[1], image.shape[0]), expand_ratio=1.25)
    crop_image, _ = crop_and_resize(image, gt_keypoints_2d, bbox, output_size=input_size)

    return {
        "domain": sample_ref.domain,
        "filename": sample_ref.filename,
        "image_name": f"{sample_ref.domain}/{sample_ref.filename}",
        "image": normalize_image(crop_image).unsqueeze(0).float(),
        "bbox": tuple(float(item) for item in bbox),
        "gt_quaternion": gt_quaternion,
        "gt_translation": gt_translation,
    }


def compute_failure_metrics(gt_translation: np.ndarray) -> Dict[str, float]:
    gt_t_norm = float(max(np.linalg.norm(gt_translation), 1e-12))
    et = gt_t_norm
    eq = float(np.pi)
    eq_deg = 180.0
    e_t_bar = 1.0
    ep = eq + e_t_bar
    return {
        "et": et,
        "eq": eq,
        "eq_deg": eq_deg,
        "e_t_bar": e_t_bar,
        "ep": ep,
        "e_star_t": et,
        "e_star_t_bar": e_t_bar,
        "e_star_q": eq,
        "e_star_p": ep,
    }


def build_step_record(
    step: int,
    sample: Dict[str, object],
    diagnosis: Dict[str, object],
    metrics: Dict[str, float],
    triggered: bool,
    adapted: bool,
    trigger_score: float,
    gate_weight: float,
    memory_size: int,
    collapse_threshold: float,
    extra: Dict[str, object] | None = None,
) -> Dict[str, object]:
    record = {
        "step": int(step),
        "domain": str(sample["domain"]),
        "image_name": str(sample["image_name"]),
        "triggered": bool(triggered),
        "adapted": bool(adapted),
        "trigger_score": float(trigger_score),
        "gate_weight": float(gate_weight),
        "memory_size": int(memory_size),
        "mean_confidence": finite_float(diagnosis.get("mean_confidence", 0.0)),
        "num_ransac_inliers": int(diagnosis.get("num_ransac_inliers", 0)),
        "inlier_ratio": finite_float(diagnosis.get("inlier_ratio", 0.0)),
        "mean_reprojection_error": finite_float(diagnosis.get("mean_reprojection_error", 0.0)),
        "max_reprojection_error": finite_float(diagnosis.get("max_reprojection_error", diagnosis.get("mean_reprojection_error", 0.0))),
        "used_fallback_epnp": bool(diagnosis.get("used_fallback_epnp", False)),
        "pose_failed": bool(diagnosis.get("rvec") is None or diagnosis.get("tvec") is None),
        "trigger_reasons": [str(item) for item in diagnosis.get("trigger_reasons", [])],
        "e_star_t_bar": float(metrics["e_star_t_bar"]),
        "e_star_q": float(metrics["e_star_q"]),
        "e_star_q_deg": float(metrics["e_star_q"] * 180.0 / np.pi),
        "e_star_p": float(metrics["e_star_p"]),
        "collapse": bool(float(metrics["e_star_p"]) > float(collapse_threshold)),
    }
    if extra:
        record.update(extra)
    return record


def summarize_stream(
    records: Sequence[Dict[str, object]],
    collapse_threshold: float,
) -> Dict[str, object]:
    if not records:
        raise ValueError("Cannot summarize an empty stream.")

    e_star_t_bar = np.asarray([float(item["e_star_t_bar"]) for item in records], dtype=np.float64)
    e_star_q = np.asarray([float(item["e_star_q"]) for item in records], dtype=np.float64)
    e_star_p = np.asarray([float(item["e_star_p"]) for item in records], dtype=np.float64)
    triggered = np.asarray([1.0 if item["triggered"] else 0.0 for item in records], dtype=np.float64)
    adapted = np.asarray([1.0 if item["adapted"] else 0.0 for item in records], dtype=np.float64)
    collapsed = np.asarray([1.0 if item["collapse"] else 0.0 for item in records], dtype=np.float64)
    inliers = np.asarray([float(item.get("num_ransac_inliers", 0.0)) for item in records], dtype=np.float64)
    fallback = np.asarray([1.0 if bool(item.get("used_fallback_epnp", False)) else 0.0 for item in records], dtype=np.float64)
    mean_conf = np.asarray([float(item.get("mean_confidence", 0.0)) for item in records], dtype=np.float64)
    mean_reproj = np.asarray([float(item.get("mean_reprojection_error", 0.0)) for item in records], dtype=np.float64)
    max_reproj = np.asarray([float(item.get("max_reprojection_error", item.get("mean_reprojection_error", 0.0))) for item in records], dtype=np.float64)

    return {
        "num_samples": int(len(records)),
        "collapse_threshold": float(collapse_threshold),
        "avg_e_star_t_bar": float(e_star_t_bar.mean()),
        "avg_e_star_q_deg": float(e_star_q.mean() * 180.0 / np.pi),
        "avg_e_star_p": float(e_star_p.mean()),
        "p95_e_star_p": float(np.percentile(e_star_p, 95)),
        "max_e_star_p": float(np.max(e_star_p)),
        "num_collapsed": int(collapsed.sum()),
        "collapse_rate": float(collapsed.mean()),
        "num_triggered": int(triggered.sum()),
        "trigger_ratio": float(triggered.mean()),
        "num_adapted": int(adapted.sum()),
        "adapt_ratio": float(adapted.mean()),
        "avg_num_ransac_inliers": float(inliers.mean()),
        "fallback_epnp_ratio": float(fallback.mean()),
        "avg_mean_confidence": float(mean_conf.mean()),
        "avg_mean_reprojection_error": float(mean_reproj.mean()),
        "avg_max_reprojection_error": float(max_reproj.mean()),
    }


def make_source_loader(args, device: torch.device) -> DataLoader:
    dataset = SpeedPlusDinoHeatmapDataset(
        data_root=args.data_root,
        split="train",
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        heatmap_sigma=args.heatmap_sigma,
        use_source_augmentation=False,
    )
    max_samples = getattr(args, "prototype_max_samples", None)
    if max_samples is not None and int(max_samples) > 0 and int(max_samples) < len(dataset):
        dataset = Subset(dataset, list(range(int(max_samples))))
    return DataLoader(
        dataset,
        batch_size=args.prototype_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )


def configure_update_scope(model: DinoHeatmapPoseModel, update_scope: str) -> List[torch.nn.Parameter]:
    if update_scope == "full_model":
        for param in model.parameters():
            param.requires_grad = True
        return [param for param in model.parameters() if param.requires_grad]
    return configure_trainable_parameters(model, update_scope)


def infer_pose(
    model: DinoHeatmapPoseModel,
    sample: Dict[str, object],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    args,
    device: torch.device,
    return_features: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, Dict[str, object], Dict[str, float]]:
    image = sample["image"].to(device, non_blocking=True)
    model.eval()
    with torch.no_grad(), autocast(enabled=device.type == "cuda"):
        if return_features:
            heatmap, cls_token = model(image, return_features=True)
        else:
            heatmap = model(image)
            cls_token = None

    diagnosis = diagnose_prediction(
        heatmap=heatmap,
        bbox=sample["bbox"],
        input_size=args.input_size,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        args=args,
    )

    rvec = diagnosis.get("rvec")
    tvec = diagnosis.get("tvec")
    if rvec is None or tvec is None:
        metrics = compute_failure_metrics(sample["gt_translation"])
    else:
        pred_quaternion = rotation_vector_to_quaternion(np.asarray(rvec))
        pred_translation = np.asarray(tvec, dtype=np.float64).reshape(-1)
        metrics = compute_metrics(
            pred_quaternion,
            pred_translation,
            sample["gt_quaternion"],
            sample["gt_translation"],
        )
    return heatmap, cls_token, diagnosis, metrics


class BaseStreamMethod:
    def __init__(self, args, device: torch.device, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> None:
        self.args = args
        self.device = device
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def process_sample(self, sample: Dict[str, object], step: int) -> Dict[str, object]:
        raise NotImplementedError


class SourceOnlyMethod(BaseStreamMethod):
    def __init__(self, args, device: torch.device, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> None:
        super().__init__(args, device, camera_matrix, dist_coeffs)
        self.model = load_source_model(args.source_checkpoint, device)

    def process_sample(self, sample: Dict[str, object], step: int) -> Dict[str, object]:
        _, _, diagnosis, metrics = infer_pose(
            self.model,
            sample,
            self.camera_matrix,
            self.dist_coeffs,
            self.args,
            self.device,
            return_features=False,
        )
        return build_step_record(
            step=step,
            sample=sample,
            diagnosis=diagnosis,
            metrics=metrics,
            triggered=False,
            adapted=False,
            trigger_score=0.0,
            gate_weight=0.0,
            memory_size=0,
            collapse_threshold=self.args.collapse_threshold,
            extra={"method": "source_only"},
        )


class OriginalOTTAMethod(BaseStreamMethod):
    def __init__(self, args, device: torch.device, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> None:
        super().__init__(args, device, camera_matrix, dist_coeffs)
        self.student_model = load_source_model(args.source_checkpoint, device)
        self.teacher_model = copy.deepcopy(self.student_model)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        source_loader = make_source_loader(args, device)
        self.source_prototype = build_original_source_prototype(self.student_model, source_loader, device)
        self.memory_bank = DynamicMemoryBank(capacity=args.original_memory_capacity)
        initialize_memory_bank(source_loader, self.memory_bank, max_samples=args.original_memory_capacity)

        self.optimizer = AdamW(self.student_model.parameters(), lr=args.original_lr, weight_decay=0.0)
        self.scaler = GradScaler(enabled=device.type == "cuda")
        self.lambda_st = 10.0
        self.lambda_ca = 0.01

    def process_sample(self, sample: Dict[str, object], step: int) -> Dict[str, object]:
        image = sample["image"].to(self.device, non_blocking=True)
        heatmap, _, diagnosis, metrics = infer_pose(
            self.student_model,
            sample,
            self.camera_matrix,
            self.dist_coeffs,
            self.args,
            self.device,
            return_features=False,
        )

        with torch.no_grad():
            scores = heatmap_confidence_score(heatmap)
            best_index = int(scores.argmax().item())
            self.memory_bank.push(image[best_index], heatmap[best_index])

        x_memory, old_pseudo, mem_indices = sample_from_memory_bank(
            self.memory_bank,
            self.args.original_memory_sample_size,
            self.device,
        )
        masked_memory = random_discontinuous_mask(
            x_memory,
            mask_ratio=self.args.mask_ratio,
            patch_size=self.args.patch_size,
        )

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=self.device.type == "cuda"):
            h_student, cls_student = self.student_model(x_memory, return_features=True)
            with torch.no_grad():
                h_teacher, cls_teacher = self.teacher_model(masked_memory, return_features=True)
            loss_st = self_training_loss(h_student, old_pseudo)
            loss_mh = masked_heatmap_consistency_loss(h_student, h_teacher, tau=self.args.tau)
            loss_ca = class_awareness_consistency_loss(cls_student, cls_teacher, self.source_prototype.to(self.device))
            total_loss = total_target_loss(loss_st, loss_mh, loss_ca, lambda_st=self.lambda_st, lambda_ca=self.lambda_ca)

        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        with torch.no_grad():
            update_memory_pseudolabels(self.memory_bank, mem_indices, h_student, old_pseudo)
            update_ema_teacher(self.student_model, self.teacher_model, alpha=self.args.ema_alpha)

        return build_step_record(
            step=step,
            sample=sample,
            diagnosis=diagnosis,
            metrics=metrics,
            triggered=True,
            adapted=True,
            trigger_score=1.0,
            gate_weight=1.0,
            memory_size=len(self.memory_bank),
            collapse_threshold=self.args.collapse_threshold,
            extra={"method": "original_otta"},
        )


class TriggeredSingleMethod(BaseStreamMethod):
    def __init__(self, args, device: torch.device, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> None:
        super().__init__(args, device, camera_matrix, dist_coeffs)
        self.model = load_source_model(args.source_checkpoint, device)
        trainable_params = configure_update_scope(self.model, args.update_scope)
        self.optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        self.scaler = GradScaler(enabled=device.type == "cuda")
        self.memory_bank = QualityMemoryBank(capacity=args.memory_capacity)

    def process_sample(self, sample: Dict[str, object], step: int) -> Dict[str, object]:
        image = sample["image"].to(self.device, non_blocking=True)
        heatmap, _, diagnosis, metrics = infer_pose(
            self.model,
            sample,
            self.camera_matrix,
            self.dist_coeffs,
            self.args,
            self.device,
            return_features=False,
        )

        maybe_push_current_sample(
            memory_bank=self.memory_bank,
            image=image,
            heatmap=heatmap,
            image_name=str(sample["image_name"]),
            quality=float(diagnosis["quality"]),
            args=self.args,
        )

        triggered = bool(diagnosis["triggered"])
        adapted = False
        if triggered and len(self.memory_bank) >= self.args.min_memory_for_update:
            geometry_target = geometry_target_from_pose(
                diagnosis.get("rvec"),
                diagnosis.get("tvec"),
                bbox=sample["bbox"],
                input_size=self.args.input_size,
                heatmap_size=heatmap.shape[-1],
                heatmap_sigma=self.args.heatmap_sigma,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
                device=self.device,
            )
            losses = adapt_single_trigger(
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                memory_bank=self.memory_bank,
                current_image=image,
                current_pseudo=heatmap.detach(),
                geometry_target=geometry_target,
                args=self.args,
                device=self.device,
            )
            adapted = bool(losses.get("executed_step", False))

        return build_step_record(
            step=step,
            sample=sample,
            diagnosis=diagnosis,
            metrics=metrics,
            triggered=triggered,
            adapted=adapted,
            trigger_score=1.0 if triggered else 0.0,
            gate_weight=1.0 if triggered else 0.0,
            memory_size=len(self.memory_bank),
            collapse_threshold=self.args.collapse_threshold,
            extra={"method": "triggered_single"},
        )


class LearnableTriggerMethod(BaseStreamMethod):
    def __init__(self, args, device: torch.device, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> None:
        super().__init__(args, device, camera_matrix, dist_coeffs)
        self.model = load_source_model(args.source_checkpoint, device)
        trainable_params = configure_update_scope(self.model, args.update_scope)
        self.optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        self.scaler = GradScaler(enabled=device.type == "cuda")
        self.gate = make_gate(args, device)
        self.gate_optimizer = AdamW(self.gate.parameters(), lr=args.gate_lr, weight_decay=args.gate_weight_decay) if self.gate is not None else None
        self.memory_bank = FeatureQualityMemoryBank(capacity=args.memory_capacity)
        self.source_prototype = build_learnable_source_prototype(self.model, args, device)

    def process_sample(self, sample: Dict[str, object], step: int) -> Dict[str, object]:
        image = sample["image"].to(self.device, non_blocking=True)
        heatmap, cls_token, diagnosis, metrics = infer_pose(
            self.model,
            sample,
            self.camera_matrix,
            self.dist_coeffs,
            self.args,
            self.device,
            return_features=True,
        )

        geo_features = build_geo_features(diagnosis, self.args, self.device)
        feat_features = self.memory_bank.feature_distances(cls_token.squeeze(0), self.source_prototype).to(self.device).unsqueeze(0)
        risk_label = heuristic_risk_label(diagnosis, self.args, self.device)

        if self.gate is not None and self.gate_optimizer is not None and self.args.train_gate:
            self.gate.train()
            self.gate_optimizer.zero_grad(set_to_none=True)
            if self.args.trigger_mode == "mlp_geo":
                risk_logit = self.gate(geo_features.detach())
            else:
                risk_logit = self.gate(geo_features.detach(), feat_features.detach())
            gate_loss = F.binary_cross_entropy_with_logits(risk_logit, risk_label)
            gate_loss.backward()
            self.gate_optimizer.step()

        if self.gate is not None:
            self.gate.eval()
        with torch.no_grad():
            risk_prob = gate_probability(self.gate, geo_features, feat_features, self.args)

        should_adapt, gate_weight = choose_gate_decision(step, risk_label, risk_prob, self.args)
        triggered = bool(should_adapt)

        if float(diagnosis["quality"]) >= self.args.memory_min_quality:
            self.memory_bank.push(
                image=image.squeeze(0),
                pseudo_heatmap=heatmap.squeeze(0),
                feature=cls_token.squeeze(0),
                quality=float(diagnosis["quality"]),
                image_name=str(sample["image_name"]),
            )

        adapted = False
        if triggered and len(self.memory_bank) >= self.args.min_memory_for_update:
            geometry_target = geometry_target_from_pose(
                diagnosis.get("rvec"),
                diagnosis.get("tvec"),
                bbox=sample["bbox"],
                input_size=self.args.input_size,
                heatmap_size=heatmap.shape[-1],
                heatmap_sigma=self.args.heatmap_sigma,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
                device=self.device,
            )
            losses = adapt_with_gate(
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                memory_bank=self.memory_bank,
                current_image=image,
                current_pseudo=heatmap.detach(),
                geometry_target=geometry_target,
                gate_weight=float(gate_weight),
                args=self.args,
                device=self.device,
            )
            adapted = bool(losses.get("executed_step", False))

        return build_step_record(
            step=step,
            sample=sample,
            diagnosis=diagnosis,
            metrics=metrics,
            triggered=triggered,
            adapted=adapted,
            trigger_score=float(risk_prob.item()),
            gate_weight=float(gate_weight),
            memory_size=len(self.memory_bank),
            collapse_threshold=self.args.collapse_threshold,
            extra={
                "method": "learnable_trigger",
                "trigger_mode": self.args.trigger_mode,
                "gate_usage": self.args.gate_usage,
                "heuristic_triggered": bool(risk_label.item() >= 0.5),
            },
        )


def make_method(method_name: str, args, device: torch.device, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> BaseStreamMethod:
    normalized = method_name.lower()
    if normalized == "source_only":
        return SourceOnlyMethod(args, device, camera_matrix, dist_coeffs)
    if normalized == "original_otta":
        return OriginalOTTAMethod(args, device, camera_matrix, dist_coeffs)
    if normalized == "triggered_single":
        return TriggeredSingleMethod(args, device, camera_matrix, dist_coeffs)
    if normalized in {"learnable_trigger", "ours"}:
        return LearnableTriggerMethod(args, device, camera_matrix, dist_coeffs)
    raise ValueError(f"Unsupported method: {method_name}")


def run_stream(method_name: str, stream_name: str, stream_samples: Sequence[DomainSampleRef], args, device: torch.device) -> Dict[str, object]:
    camera_matrix, dist_coeffs = load_camera(Path(args.data_root))
    method = make_method(method_name, args, device, camera_matrix, dist_coeffs)
    records: List[Dict[str, object]] = []

    for step, sample_ref in enumerate(tqdm(stream_samples, desc=f"{method_name}_{stream_name}"), start=1):
        sample = prepare_sample(sample_ref, camera_matrix, dist_coeffs, args.input_size)
        record = method.process_sample(sample, step)
        records.append(record)

    summary = summarize_stream(records, collapse_threshold=args.collapse_threshold)
    summary.update(
        {
            "method": method_name,
            "stream_name": stream_name,
            "domains": [domain.lower() for domain in args.domains],
            "k_per_domain": int(args.k_per_domain),
            "seed": int(args.seed),
        }
    )
    return {"summary": summary, "records": records}


def save_stream_outputs(output_dir: Path, result: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result["summary"], handle, indent=2)
    with (output_dir / "step_history.json").open("w", encoding="utf-8") as handle:
        json.dump(result["records"], handle, indent=2)


def run_experiment(args) -> None:
    if len(args.domains) != 3:
        raise ValueError(f"Expected exactly 3 domains, got {args.domains}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    data_root = Path(args.data_root)
    output_root = Path(args.output_dir)

    samples_by_domain, k_per_domain = permute_and_trim_domains(
        data_root=data_root,
        domains=args.domains,
        seed=args.seed,
        max_per_domain=args.max_per_domain,
    )
    args.k_per_domain = int(k_per_domain)

    stream_results: Dict[str, Dict[str, object]] = {}
    for stream_name in args.streams:
        stream_samples = build_stream(samples_by_domain, args.domains, stream_name)
        result = run_stream(args.method, stream_name, stream_samples, args, device)
        stream_results[stream_name] = result["summary"]
        save_stream_outputs(output_root / args.method / stream_name, result)

    overall = {
        "method": args.method,
        "domains": [domain.lower() for domain in args.domains],
        "streams": {name: stream_results[name] for name in args.streams},
        "k_per_domain": int(k_per_domain),
        "seed": int(args.seed),
        "source_checkpoint": args.source_checkpoint,
        "collapse_threshold": float(args.collapse_threshold),
        "trigger_mode": args.trigger_mode if args.method in {"learnable_trigger", "ours"} else None,
        "gate_usage": args.gate_usage if args.method in {"learnable_trigger", "ours"} else None,
        "update_scope": args.update_scope if args.method != "source_only" else None,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / f"{args.method}_table9_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(overall, handle, indent=2)
    print(json.dumps(overall, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="speedplusv2")
    parser.add_argument("--source_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="mixed_domain_stream_results")
    parser.add_argument("--method", type=str, choices=["source_only", "original_otta", "triggered_single", "learnable_trigger", "ours"], default="ours")
    parser.add_argument("--domains", nargs=3, default=["sunlamp", "lightbox", "shirt"])
    parser.add_argument("--streams", nargs="+", choices=list(STREAM_NAMES), default=list(STREAM_NAMES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_per_domain", type=int, default=0)
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
    args.max_per_domain = None if args.max_per_domain <= 0 else int(args.max_per_domain)
    return args


if __name__ == "__main__":
    run_experiment(parse_args())


