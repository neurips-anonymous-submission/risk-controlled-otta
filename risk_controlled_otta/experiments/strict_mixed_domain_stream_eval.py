from __future__ import annotations

import argparse
import copy
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm

from risk_controlled_otta.eval.evaluate_dino_heatmap import compute_metrics, rotation_vector_to_quaternion
from risk_controlled_otta.experiments import hybrid_tta_lite_single_model_tta_speed_dino_heatmap as hybrid_mod
from risk_controlled_otta.experiments import ltta_strict_single_model_tta_speed_dino_heatmap as ltta_mod
from risk_controlled_otta.experiments import petta_strict_single_model_tta_speed_dino_heatmap as petta_mod
from risk_controlled_otta.experiments.mixed_domain_stream_eval import (
    BaseStreamMethod,
    DomainSampleRef,
    build_step_record,
    build_stream,
    compute_failure_metrics,
    load_domain_refs,
    save_stream_outputs,
    summarize_stream,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def prepare_raw_sample(sample_ref: DomainSampleRef) -> Dict[str, object]:
    image = np.asarray(Image.open(sample_ref.image_path).convert("RGB"))
    return {
        "domain": sample_ref.domain,
        "image_name": f"{sample_ref.domain}/{sample_ref.filename}",
        "image_np": image,
        "image_path": str(sample_ref.image_path),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "gt_quaternion": np.asarray(sample_ref.gt_quaternion, dtype=np.float64),
        "gt_translation": np.asarray(sample_ref.gt_translation, dtype=np.float64),
    }


def diagnosis_to_metrics(diagnosis: Dict[str, object], sample: Dict[str, object]) -> Dict[str, float]:
    rvec = diagnosis.get("rvec")
    tvec = diagnosis.get("tvec")
    if rvec is None or tvec is None or bool(diagnosis.get("pose_failed", False)):
        return compute_failure_metrics(sample["gt_translation"])
    pred_quaternion = rotation_vector_to_quaternion(np.asarray(rvec))
    pred_translation = np.asarray(tvec, dtype=np.float64).reshape(-1)
    return compute_metrics(pred_quaternion, pred_translation, sample["gt_quaternion"], sample["gt_translation"])


class StrictLTTAMethod(BaseStreamMethod):
    def __init__(self, args, device: torch.device, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> None:
        super().__init__(args, device, camera_matrix, dist_coeffs)
        self.model = ltta_mod.load_source_model(args.source_checkpoint, device)
        self.trainable_params = ltta_mod.configure_ltta_parameters(self.model, args.update_scope)
        self.optimizer = AdamW(self.trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        self.scaler = GradScaler(enabled=device.type == "cuda")

    def process_sample(self, sample: Dict[str, object], step: int) -> Dict[str, object]:
        image_np = sample["image_np"]
        pseudo_bbox, bbox_info = ltta_mod.predict_pseudo_bbox(self.model, image_np, self.args, self.device)
        image = ltta_mod.make_predicted_crop_tensor(image_np, pseudo_bbox, self.args.input_size, self.device)

        adapted = False
        loss_record = {
            "loss_ltta_total": 0.0,
            "loss_entropy": 0.0,
            "loss_confidence": 0.0,
            "loss_dwt": 0.0,
        }

        self.model.train()
        for _ in range(self.args.adapt_steps):
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.device.type == "cuda"):
                pred_heatmap = self.model(image)
                loss, loss_record = ltta_mod.ltta_objective(pred_heatmap, image, self.args)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.trainable_params, self.args.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            adapted = True

        self.model.eval()
        with torch.no_grad(), autocast(enabled=self.device.type == "cuda"):
            heatmap_after = self.model(image)

        diagnosis = ltta_mod.diagnose_prediction(
            heatmap=heatmap_after,
            bbox=pseudo_bbox,
            input_size=self.args.input_size,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            args=self.args,
        )
        metrics = diagnosis_to_metrics(diagnosis, sample)
        return build_step_record(
            step=step,
            sample=sample,
            diagnosis=diagnosis,
            metrics=metrics,
            triggered=True,
            adapted=adapted,
            trigger_score=1.0,
            gate_weight=1.0,
            memory_size=0,
            collapse_threshold=self.args.collapse_threshold,
            extra={"method": "strict_ltta", **bbox_info, **loss_record},
        )


class StrictPeTTAMethod(BaseStreamMethod):
    def __init__(self, args, device: torch.device, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> None:
        super().__init__(args, device, camera_matrix, dist_coeffs)
        self.model = petta_mod.load_source_model(args.source_checkpoint, device)
        self.source_model = copy.deepcopy(self.model).to(device)
        self.source_model.eval()
        for param in self.source_model.parameters():
            param.requires_grad_(False)

        self.normal_params = petta_mod.select_update_parameters(self.model, args.normal_update_scope)
        self.guarded_params = petta_mod.select_update_parameters(self.model, args.guarded_update_scope)
        self.normal_optimizer = AdamW(self.normal_params, lr=args.lr, weight_decay=args.weight_decay)
        self.guarded_optimizer = AdamW(self.guarded_params, lr=args.guarded_lr, weight_decay=args.guarded_weight_decay)
        self.scaler = GradScaler(enabled=device.type == "cuda")
        self.teacher_model = petta_mod.make_ema_teacher(self.model, device) if args.use_ema_teacher else None
        self.monitor = petta_mod.PersistentCollapseMonitor(
            window_size=args.petta_window_size,
            warmup_steps=args.petta_warmup_steps,
            threshold=args.petta_threshold,
        )
        self.last_stable_state = petta_mod.clone_state_dict_to_cpu(self.model)
        self.last_stable_quality = float("-inf")

    def process_sample(self, sample: Dict[str, object], step: int) -> Dict[str, object]:
        image_np = sample["image_np"]
        pseudo_bbox, bbox_info = petta_mod.predict_pseudo_bbox(self.model, image_np, self.args, self.device)
        image = petta_mod.make_predicted_crop_tensor(image_np, pseudo_bbox, self.args.input_size, self.device)

        self.model.eval()
        with torch.no_grad(), autocast(enabled=self.device.type == "cuda"):
            heatmap_before = self.model(image)
        pre_diagnosis = petta_mod.diagnose_prediction(
            heatmap=heatmap_before,
            bbox=pseudo_bbox,
            input_size=self.args.input_size,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            args=self.args,
        )
        decision = self.monitor.decide(pre_diagnosis, step=step, args=self.args)

        stage = decision["petta_stage"]
        if stage == "guarded_adapt":
            active_params = self.guarded_params
            optimizer = self.guarded_optimizer
            active_grad_clip = self.args.guarded_grad_clip_norm
            active_steps = self.args.guarded_adapt_steps
        elif stage == "freeze_or_reset":
            active_params = []
            optimizer = None
            active_grad_clip = 0.0
            active_steps = 0
        else:
            active_params = self.normal_params
            optimizer = self.normal_optimizer
            active_grad_clip = self.args.grad_clip_norm
            active_steps = self.args.adapt_steps

        adapted = False
        soft_reset_applied = False
        rollback_applied = False
        loss_record = {
            "loss_petta_total": 0.0,
            "loss_entropy": 0.0,
            "loss_confidence": 0.0,
            "loss_dwt": 0.0,
            "loss_mim": 0.0,
            "loss_anchor": 0.0,
        }

        if stage == "freeze_or_reset":
            if self.args.reset_to_last_stable:
                petta_mod.restore_state_dict(self.model, self.last_stable_state, self.device)
                rollback_applied = True
            if self.args.soft_reset_momentum > 0:
                petta_mod.blend_student_with_teacher(self.model, self.teacher_model, self.source_model, self.args.soft_reset_momentum)
                soft_reset_applied = True
            petta_mod.clear_optimizer_state(self.normal_optimizer)
            petta_mod.clear_optimizer_state(self.guarded_optimizer)
        else:
            petta_mod.set_active_trainable_params(self.model, active_params)
            self.model.train()
            for _ in range(active_steps):
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=self.device.type == "cuda"):
                    pred_heatmap = self.model(image)
                    loss, loss_record = petta_mod.petta_objective(
                        self.model,
                        self.teacher_model,
                        self.source_model,
                        active_params,
                        pred_heatmap,
                        image,
                        self.args,
                    )
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                if active_grad_clip > 0 and len(active_params) > 0:
                    torch.nn.utils.clip_grad_norm_(active_params, active_grad_clip)
                self.scaler.step(optimizer)
                self.scaler.update()
                adapted = True

        if self.teacher_model is not None:
            petta_mod.update_ema_teacher(self.teacher_model, self.model, momentum=self.args.teacher_momentum)

        self.model.eval()
        with torch.no_grad(), autocast(enabled=self.device.type == "cuda"):
            heatmap_after = self.model(image)

        diagnosis = petta_mod.diagnose_prediction(
            heatmap=heatmap_after,
            bbox=pseudo_bbox,
            input_size=self.args.input_size,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            args=self.args,
        )
        self.monitor.update(diagnosis)

        current_quality = float(diagnosis["quality"])
        if current_quality >= self.last_stable_quality and not bool(diagnosis.get("pose_failed", False)):
            self.last_stable_quality = current_quality
            self.last_stable_state = petta_mod.clone_state_dict_to_cpu(self.model)

        metrics = diagnosis_to_metrics(diagnosis, sample)
        return build_step_record(
            step=step,
            sample=sample,
            diagnosis=diagnosis,
            metrics=metrics,
            triggered=bool(decision["collapse_risk_detected"]),
            adapted=adapted,
            trigger_score=float(decision["collapse_risk_score"]),
            gate_weight=1.0 if adapted else 0.0,
            memory_size=0,
            collapse_threshold=self.args.collapse_threshold,
            extra={
                "method": "strict_petta",
                **bbox_info,
                **decision,
                "pre_quality": float(pre_diagnosis["quality"]),
                "pre_mean_confidence": float(pre_diagnosis["mean_confidence"]),
                "pre_entropy_for_petta": float(pre_diagnosis["entropy_for_petta"]),
                "soft_reset_applied": bool(soft_reset_applied),
                "rollback_applied": bool(rollback_applied),
                **loss_record,
            },
        )


class StrictHybridTTALiteMethod(BaseStreamMethod):
    def __init__(self, args, device: torch.device, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> None:
        super().__init__(args, device, camera_matrix, dist_coeffs)
        self.model = hybrid_mod.load_source_model(args.source_checkpoint, device)
        self.efficient_params, self.full_params = hybrid_mod.configure_hybrid_tta_parameters(
            self.model,
            efficient_scope=args.efficient_update_scope,
            full_scope=args.full_update_scope,
        )
        self.efficient_optimizer = AdamW(self.efficient_params, lr=args.lr, weight_decay=args.weight_decay)
        self.full_optimizer = AdamW(self.full_params, lr=args.full_lr, weight_decay=args.full_weight_decay)
        self.scaler = GradScaler(enabled=device.type == "cuda")
        self.teacher_model = hybrid_mod.make_ema_teacher(self.model, device) if args.use_ema_teacher else None
        self.detector = hybrid_mod.DynamicDomainShiftDetector(
            window_size=args.ddsd_window_size,
            warmup_steps=args.ddsd_warmup_steps,
            threshold=args.ddsd_threshold,
            cooldown_steps=args.ddsd_cooldown_steps,
        )

    def process_sample(self, sample: Dict[str, object], step: int) -> Dict[str, object]:
        image_np = sample["image_np"]
        pseudo_bbox, bbox_info = hybrid_mod.predict_pseudo_bbox(self.model, image_np, self.args, self.device)
        image = hybrid_mod.make_predicted_crop_tensor(image_np, pseudo_bbox, self.args.input_size, self.device)

        self.model.eval()
        with torch.no_grad(), autocast(enabled=self.device.type == "cuda"):
            heatmap_before = self.model(image)
        pre_diagnosis = hybrid_mod.diagnose_prediction(
            heatmap=heatmap_before,
            bbox=pseudo_bbox,
            input_size=self.args.input_size,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            args=self.args,
        )
        hybrid_decision = self.detector.decide(pre_diagnosis, step=step, args=self.args)

        if hybrid_decision["hybrid_mode"] == "full":
            active_params = self.full_params
            optimizer = self.full_optimizer
            active_grad_clip = self.args.full_grad_clip_norm
        else:
            active_params = self.efficient_params
            optimizer = self.efficient_optimizer
            active_grad_clip = self.args.grad_clip_norm

        adapted = False
        loss_record = {
            "loss_ltta_total": 0.0,
            "loss_entropy": 0.0,
            "loss_confidence": 0.0,
            "loss_dwt": 0.0,
            "loss_mim": 0.0,
        }

        hybrid_mod.set_active_trainable_params(self.model, active_params)
        self.model.train()
        for _ in range(self.args.adapt_steps):
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.device.type == "cuda"):
                pred_heatmap = self.model(image)
                loss, loss_record = hybrid_mod.hybrid_tta_objective(self.model, self.teacher_model, pred_heatmap, image, self.args)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            if active_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(active_params, active_grad_clip)
            self.scaler.step(optimizer)
            self.scaler.update()
            adapted = True

        if self.teacher_model is not None:
            hybrid_mod.update_ema_teacher(self.teacher_model, self.model, momentum=self.args.teacher_momentum)

        self.model.eval()
        with torch.no_grad(), autocast(enabled=self.device.type == "cuda"):
            heatmap_after = self.model(image)

        diagnosis = hybrid_mod.diagnose_prediction(
            heatmap=heatmap_after,
            bbox=pseudo_bbox,
            input_size=self.args.input_size,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            args=self.args,
        )
        self.detector.update(diagnosis)
        metrics = diagnosis_to_metrics(diagnosis, sample)
        return build_step_record(
            step=step,
            sample=sample,
            diagnosis=diagnosis,
            metrics=metrics,
            triggered=bool(hybrid_decision["domain_shift_detected"]),
            adapted=adapted,
            trigger_score=float(hybrid_decision.get("ddsd_score", 0.0)),
            gate_weight=1.0 if hybrid_decision["hybrid_mode"] == "full" else 0.0,
            memory_size=0,
            collapse_threshold=self.args.collapse_threshold,
            extra={
                "method": "strict_hybrid_tta_lite",
                **bbox_info,
                **hybrid_decision,
                "pre_quality": float(pre_diagnosis["quality"]),
                "pre_mean_confidence": float(pre_diagnosis["mean_confidence"]),
                "pre_entropy_for_ddsd": float(pre_diagnosis["entropy_for_ddsd"]),
                **loss_record,
            },
        )


def make_method(method_name: str, args, device: torch.device, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> BaseStreamMethod:
    normalized = method_name.lower()
    if normalized == "strict_ltta":
        return StrictLTTAMethod(args, device, camera_matrix, dist_coeffs)
    if normalized == "strict_petta":
        return StrictPeTTAMethod(args, device, camera_matrix, dist_coeffs)
    if normalized == "strict_hybrid_tta_lite":
        return StrictHybridTTALiteMethod(args, device, camera_matrix, dist_coeffs)
    raise ValueError(f"Unsupported method: {method_name}")


def run_stream(method_name: str, stream_name: str, stream_samples: Sequence[DomainSampleRef], args, device: torch.device) -> Dict[str, object]:
    camera_matrix, dist_coeffs = hybrid_mod.load_camera(Path(args.data_root))
    method = make_method(method_name, args, device, camera_matrix, dist_coeffs)
    records: List[Dict[str, object]] = []
    start_time = time.perf_counter()

    for step, sample_ref in enumerate(tqdm(stream_samples, desc=f"{method_name}_{stream_name}"), start=1):
        sample = prepare_raw_sample(sample_ref)
        records.append(method.process_sample(sample, step))

    runtime_minutes = (time.perf_counter() - start_time) / 60.0
    summary = summarize_stream(records, collapse_threshold=args.collapse_threshold)
    summary.update(
        {
            "method": method_name,
            "stream_name": stream_name,
            "domains": [domain.lower() for domain in args.domains],
            "k_per_domain": int(args.k_per_domain),
            "seed": int(args.seed),
            "runtime_minutes": float(runtime_minutes),
        }
    )
    return {"summary": summary, "records": records}


def run_experiment(args) -> None:
    if len(args.domains) != 3:
        raise ValueError(f"Expected exactly 3 domains, got {args.domains}")

    set_seed(args.seed)
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
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / f"{args.method}_table89_summary.json").open("w", encoding="utf-8") as handle:
        import json

        json.dump(overall, handle, indent=2)
    print(overall)


def parse_args():
    parser = argparse.ArgumentParser(description="Strict mixed-domain stream evaluation for LTTA / PeTTA / Hybrid-TTA-lite.")
    parser.add_argument("--data_root", type=str, default="speedplusv2")
    parser.add_argument("--source_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="mixed_domain_stream_results_strict")
    parser.add_argument("--method", type=str, choices=["strict_ltta", "strict_petta", "strict_hybrid_tta_lite"], required=True)
    parser.add_argument("--domains", nargs=3, default=["sunlamp", "lightbox", "shirt"])
    parser.add_argument("--streams", nargs="+", choices=["forward", "reverse", "cyclic", "shifted"], default=["forward", "reverse", "cyclic", "shifted"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_per_domain", type=int, default=0)
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

    parser.add_argument("--update_scope", type=str, choices=["stem", "stem_norm", "decoder"], default="stem")

    parser.add_argument("--use_ema_teacher", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--teacher_momentum", type=float, default=0.999)
    parser.add_argument("--lambda_mim", type=float, default=0.25)
    parser.add_argument("--lambda_anchor", type=float, default=0.01)
    parser.add_argument("--mim_mask_ratio", type=float, default=0.35)
    parser.add_argument("--mim_patch_size", type=int, default=32)

    parser.add_argument("--normal_update_scope", type=str, choices=["stem", "stem_norm", "decoder", "stem_decoder", "all"], default="stem")
    parser.add_argument("--guarded_update_scope", type=str, choices=["stem", "stem_norm", "decoder", "stem_decoder", "all"], default="decoder")
    parser.add_argument("--guarded_lr", type=float, default=1e-6)
    parser.add_argument("--guarded_weight_decay", type=float, default=0.0)
    parser.add_argument("--guarded_adapt_steps", type=int, default=1)
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
    parser.add_argument("--reset_to_last_stable", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--efficient_update_scope", type=str, choices=["stem", "stem_norm", "decoder", "stem_decoder", "all"], default="stem")
    parser.add_argument("--full_update_scope", type=str, choices=["stem", "stem_norm", "decoder", "stem_decoder", "all"], default="stem_decoder")
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
    args.max_per_domain = None if args.max_per_domain <= 0 else int(args.max_per_domain)
    return args


if __name__ == "__main__":
    run_experiment(parse_args())


