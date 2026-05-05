from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm

from risk_controlled_otta.experiments.mixed_domain_stream_eval import (
    STREAM_NAMES,
    build_step_record,
    build_stream,
    infer_pose,
    permute_and_trim_domains,
    prepare_sample,
    save_stream_outputs,
    summarize_stream,
)
from risk_controlled_otta.models.dino_pose_model import DinoHeatmapPoseModel
from risk_controlled_otta.adapt.triggered_single_model_tta_dino_heatmap import load_source_model
from data.crop_and_heatmap import load_camera


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def denormalize(images: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(images.device, images.dtype)
    std = IMAGENET_STD.to(images.device, images.dtype)
    return images * std + mean


def normalize(images: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(images.device, images.dtype)
    std = IMAGENET_STD.to(images.device, images.dtype)
    return (images - mean) / std


def photometric_augment(images: torch.Tensor, noise_std: float = 0.03) -> torch.Tensor:
    rgb = denormalize(images).clamp(0.0, 1.0)
    brightness = 1.0 + (torch.rand(images.shape[0], 1, 1, 1, device=images.device) - 0.5) * 0.4
    contrast = 1.0 + (torch.rand(images.shape[0], 1, 1, 1, device=images.device) - 0.5) * 0.4
    mean = rgb.mean(dim=(2, 3), keepdim=True)
    rgb = (rgb - mean) * contrast + mean
    rgb = rgb * brightness
    if noise_std > 0:
        rgb = rgb + torch.randn_like(rgb) * noise_std
    rgb = rgb.clamp(0.0, 1.0)
    return normalize(rgb)


def heatmap_spatial_probs(heatmap: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    b, k, h, w = heatmap.shape
    logits = heatmap.reshape(b, k, h * w) / max(temperature, 1e-6)
    return F.softmax(logits, dim=-1)


def mean_peak_confidence(heatmap: torch.Tensor) -> float:
    probs = heatmap_spatial_probs(heatmap)
    peaks = probs.max(dim=-1).values
    return float(peaks.mean().item())


def soft_cross_entropy_heatmap(student_heatmap: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
    b, k, h, w = student_heatmap.shape
    log_probs = F.log_softmax(student_heatmap.reshape(b, k, h * w), dim=-1)
    return -(teacher_probs.detach() * log_probs).sum(dim=-1).mean()


@torch.no_grad()
def update_ema(model: DinoHeatmapPoseModel, ema_model: DinoHeatmapPoseModel, alpha: float) -> None:
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1.0 - alpha)


@torch.no_grad()
def stochastic_restore(model: DinoHeatmapPoseModel, source_state: Dict[str, torch.Tensor], restore_prob: float) -> None:
    if restore_prob <= 0.0:
        return
    model_state = model.state_dict()
    for name, tensor in model_state.items():
        if not tensor.is_floating_point():
            continue
        source_tensor = source_state.get(name)
        if source_tensor is None or source_tensor.shape != tensor.shape:
            continue
        mask = (torch.rand_like(tensor, dtype=torch.float32) < restore_prob)
        tensor.copy_(torch.where(mask, source_tensor.to(tensor.device, tensor.dtype), tensor))


def configure_update_scope(model: DinoHeatmapPoseModel, update_scope: str) -> List[torch.nn.Parameter]:
    for param in model.parameters():
        param.requires_grad = False

    if update_scope == "full_model":
        for param in model.parameters():
            param.requires_grad = True
    else:
        for param in model.decoder.parameters():
            param.requires_grad = True
        if update_scope == "decoder_last_block":
            blocks = getattr(model.encoder, "blocks", None)
            if blocks is None or len(blocks) == 0:
                raise ValueError("update_scope=decoder_last_block requires encoder.blocks.")
            for param in blocks[-1].parameters():
                param.requires_grad = True
        elif update_scope != "decoder":
            raise ValueError(f"Unsupported update_scope: {update_scope}")

    return [param for param in model.parameters() if param.requires_grad]


class CoTTAHeatmapMethod:
    def __init__(self, args, device: torch.device, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> None:
        self.args = args
        self.device = device
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

        self.model = load_source_model(args.source_checkpoint, device)
        self.ema_model = copy.deepcopy(self.model).eval()
        self.anchor_model = copy.deepcopy(self.model).eval()
        for module in (self.ema_model, self.anchor_model):
            for param in module.parameters():
                param.requires_grad = False

        self.source_state = {name: tensor.detach().cpu().clone() for name, tensor in self.model.state_dict().items()}
        trainable_params = configure_update_scope(self.model, args.update_scope)
        self.optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        self.scaler = GradScaler(enabled=device.type == "cuda")

    def teacher_probs(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), autocast(enabled=self.device.type == "cuda"):
            anchor_heatmap = self.anchor_model(image)
            ema_heatmap = self.ema_model(image)
        anchor_conf = mean_peak_confidence(anchor_heatmap)
        if anchor_conf >= self.args.anchor_conf_threshold:
            return heatmap_spatial_probs(ema_heatmap, temperature=self.args.temperature)

        probs = []
        with torch.no_grad():
            for _ in range(self.args.aug_times):
                aug_image = photometric_augment(image, noise_std=self.args.aug_noise_std)
                with autocast(enabled=self.device.type == "cuda"):
                    aug_heatmap = self.ema_model(aug_image)
                probs.append(heatmap_spatial_probs(aug_heatmap, temperature=self.args.temperature))
        return torch.stack(probs, dim=0).mean(dim=0)

    def process_sample(self, sample: Dict[str, object], step: int) -> Dict[str, object]:
        image = sample["image"].to(self.device, non_blocking=True)
        _, _, diagnosis, metrics = infer_pose(
            self.model,
            sample,
            self.camera_matrix,
            self.dist_coeffs,
            self.args,
            self.device,
            return_features=False,
        )

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=self.device.type == "cuda"):
            student_heatmap = self.model(image)
            teacher_probs = self.teacher_probs(image)
            loss = soft_cross_entropy_heatmap(student_heatmap, teacher_probs)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        update_ema(self.model, self.ema_model, alpha=self.args.ema_decay)
        stochastic_restore(self.model, self.source_state, restore_prob=self.args.restore_prob)

        return build_step_record(
            step=step,
            sample=sample,
            diagnosis=diagnosis,
            metrics=metrics,
            triggered=True,
            adapted=True,
            trigger_score=1.0,
            gate_weight=1.0,
            memory_size=0,
            collapse_threshold=self.args.collapse_threshold,
            extra={
                "method": "cotta",
                "loss": float(loss.item()),
                "anchor_confidence": float(mean_peak_confidence(self.anchor_model(image))),
            },
        )


def run_experiment(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    data_root = Path(args.data_root)
    output_root = Path(args.output_dir)
    camera_matrix, dist_coeffs = load_camera(data_root)

    samples_by_domain, k_per_domain = permute_and_trim_domains(
        data_root=data_root,
        domains=args.domains,
        seed=args.seed,
        max_per_domain=args.max_per_domain,
    )

    stream_results = {}
    for stream_name in args.streams:
        method = CoTTAHeatmapMethod(args, device, camera_matrix, dist_coeffs)
        stream_samples = build_stream(samples_by_domain, args.domains, stream_name)
        records = []
        for step, sample_ref in enumerate(tqdm(stream_samples, desc=f"cotta_{stream_name}"), start=1):
            sample = prepare_sample(sample_ref, camera_matrix, dist_coeffs, args.input_size)
            records.append(method.process_sample(sample, step))
        summary = summarize_stream(records, collapse_threshold=args.collapse_threshold)
        summary.update(
            {
                "method": "cotta",
                "stream_name": stream_name,
                "domains": [domain.lower() for domain in args.domains],
                "k_per_domain": int(k_per_domain),
                "seed": int(args.seed),
                "update_scope": args.update_scope,
            }
        )
        stream_results[stream_name] = summary
        save_stream_outputs(output_root / "cotta" / stream_name, {"summary": summary, "records": records})

    overall = {
        "method": "cotta",
        "domains": [domain.lower() for domain in args.domains],
        "streams": stream_results,
        "k_per_domain": int(k_per_domain),
        "seed": int(args.seed),
        "source_checkpoint": args.source_checkpoint,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "cotta_table9_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(overall, handle, indent=2)
    print(json.dumps(overall, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="speedplusv2")
    parser.add_argument("--source_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="mixed_domain_stream_results_external")
    parser.add_argument("--domains", nargs=3, default=["sunlamp", "lightbox", "shirt"])
    parser.add_argument("--streams", nargs="+", choices=list(STREAM_NAMES), default=list(STREAM_NAMES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_per_domain", type=int, default=0)
    parser.add_argument("--collapse_threshold", type=float, default=0.1)
    parser.add_argument("--input_size", type=int, default=384)
    parser.add_argument("--heatmap_size", type=int, default=96)
    parser.add_argument("--heatmap_sigma", type=float, default=3.0)
    parser.add_argument("--update_scope", type=str, choices=["decoder", "decoder_last_block", "full_model"], default="decoder")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--restore_prob", type=float, default=0.001)
    parser.add_argument("--anchor_conf_threshold", type=float, default=0.1)
    parser.add_argument("--aug_times", type=int, default=8)
    parser.add_argument("--aug_noise_std", type=float, default=0.03)

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
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()
    args.max_per_domain = None if args.max_per_domain <= 0 else int(args.max_per_domain)
    return args


if __name__ == "__main__":
    run_experiment(parse_args())


