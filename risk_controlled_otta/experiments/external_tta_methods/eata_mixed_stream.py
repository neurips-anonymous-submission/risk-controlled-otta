from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm

from data.crop_and_heatmap import load_camera
from risk_controlled_otta.adapt.triggered_single_model_tta_dino_heatmap import load_source_model
from risk_controlled_otta.data.dino_heatmap_dataset import SpeedPlusDinoHeatmapDataset
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
from torch.utils.data import DataLoader


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


def heatmap_spatial_probs(heatmap: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    b, k, h, w = heatmap.shape
    logits = heatmap.reshape(b, k, h * w) / max(temperature, 1e-6)
    return F.softmax(logits, dim=-1)


def heatmap_entropy(heatmap: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    probs = heatmap_spatial_probs(heatmap, temperature=temperature)
    return -(probs * torch.log(probs.clamp_min(1e-6))).sum(dim=-1).mean(dim=-1)


def heatmap_peakiness(heatmap: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    probs = heatmap_spatial_probs(heatmap, temperature=temperature)
    return probs.max(dim=-1).values.mean(dim=-1)


def flatten_probs(heatmap: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    probs = heatmap_spatial_probs(heatmap, temperature=temperature)
    flat = probs.reshape(probs.shape[0], -1)
    return F.normalize(flat, dim=-1)


def collect_source_loader(args) -> DataLoader:
    dataset = SpeedPlusDinoHeatmapDataset(
        data_root=args.data_root,
        split="train",
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        heatmap_sigma=args.heatmap_sigma,
        use_source_augmentation=False,
    )
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=not args.no_cuda)


@torch.enable_grad()
def estimate_fisher(model: DinoHeatmapPoseModel, loader: DataLoader, device: torch.device, max_samples: int) -> Dict[str, tuple[torch.Tensor, torch.Tensor]]:
    fishers: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    count = 0
    model.train()
    for batch in tqdm(loader, desc="eata_fisher", leave=False):
        if count >= max_samples:
            break
        image = batch["image"].to(device, non_blocking=True)
        gt_heatmap = batch["heatmap"].to(device, non_blocking=True)
        model.zero_grad(set_to_none=True)
        pred = model(image)
        loss = F.mse_loss(pred, gt_heatmap)
        loss.backward()
        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            grad_sq = param.grad.detach().pow(2)
            if name in fishers:
                fishers[name] = (fishers[name][0] + grad_sq, fishers[name][1])
            else:
                fishers[name] = (grad_sq.clone(), param.detach().clone())
        count += 1

    if count == 0:
        return {}
    return {name: (stat[0] / float(count), stat[1]) for name, stat in fishers.items()}


class EATAHeatmapMethod:
    def __init__(self, args, device: torch.device, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> None:
        self.args = args
        self.device = device
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

        self.model = load_source_model(args.source_checkpoint, device)
        trainable_params = configure_update_scope(self.model, args.update_scope)
        self.optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        self.scaler = GradScaler(enabled=device.type == "cuda")
        self.prob_momentum: torch.Tensor | None = None

        fisher_loader = collect_source_loader(args)
        self.fishers = estimate_fisher(self.model, fisher_loader, device, max_samples=args.fisher_samples)

    def ewc_loss(self) -> torch.Tensor:
        if not self.fishers:
            return torch.tensor(0.0, device=self.device)
        loss = torch.tensor(0.0, device=self.device)
        named_params = dict(self.model.named_parameters())
        for name, (fisher, reference) in self.fishers.items():
            param = named_params.get(name)
            if param is None:
                continue
            loss = loss + (fisher.to(self.device) * (param - reference.to(self.device)).pow(2)).sum()
        return loss

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
        with autocast(enabled=self.device.type == "cuda"):
            current_heatmap = self.model(image)
            entropy = heatmap_entropy(current_heatmap, temperature=self.args.temperature)
            peakiness = heatmap_peakiness(current_heatmap, temperature=self.args.temperature)
            current_prob = flatten_probs(current_heatmap, temperature=self.args.temperature)

        mean_confidence = float(diagnosis.get("mean_confidence", 0.0))
        if self.args.selection_metric == "mean_confidence":
            selection_score = mean_confidence
        elif self.args.selection_metric == "peakiness":
            selection_score = float(peakiness.item())
        else:
            selection_score = max(mean_confidence, float(peakiness.item()))

        selected = bool(selection_score >= self.args.selection_margin)
        redundant = False
        cosine_sim = 0.0
        if selected and self.prob_momentum is not None:
            cosine_sim = float(F.cosine_similarity(current_prob, self.prob_momentum.unsqueeze(0), dim=1).item())
            redundant = bool(cosine_sim > self.args.d_margin)
        if selected and not redundant:
            self.prob_momentum = current_prob.detach().squeeze(0) if self.prob_momentum is None else (
                self.args.momentum * self.prob_momentum + (1.0 - self.args.momentum) * current_prob.detach().squeeze(0)
            )
            self.prob_momentum = F.normalize(self.prob_momentum, dim=0)

        adapted = False
        loss_value = 0.0
        if selected and not redundant:
            denom = max(1.0 - self.args.selection_margin, 1e-6)
            coeff = float(np.clip((selection_score - self.args.selection_margin) / denom, 0.0, 1.0))
            coeff = max(coeff, self.args.min_selection_weight)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.device.type == "cuda"):
                heatmap = self.model(image)
                ent = heatmap_entropy(heatmap, temperature=self.args.temperature).mean()
                ewc = self.ewc_loss()
                loss = coeff * ent + self.args.fisher_alpha * ewc
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            adapted = True
            loss_value = float(loss.item())

        return build_step_record(
            step=step,
            sample=sample,
            diagnosis=diagnosis,
            metrics=metrics,
            triggered=selected and not redundant,
            adapted=adapted,
            trigger_score=float(selection_score),
            gate_weight=1.0 if adapted else 0.0,
            memory_size=0,
            collapse_threshold=self.args.collapse_threshold,
            extra={
                "method": "eata",
                "entropy": float(entropy.item()),
                "peakiness": float(peakiness.item()),
                "selection_score": float(selection_score),
                "mean_confidence_gate": float(mean_confidence),
                "cosine_similarity": float(cosine_sim),
                "redundant": bool(redundant),
                "loss": float(loss_value),
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
        method = EATAHeatmapMethod(args, device, camera_matrix, dist_coeffs)
        stream_samples = build_stream(samples_by_domain, args.domains, stream_name)
        records = []
        for step, sample_ref in enumerate(tqdm(stream_samples, desc=f"eata_{stream_name}"), start=1):
            sample = prepare_sample(sample_ref, camera_matrix, dist_coeffs, args.input_size)
            records.append(method.process_sample(sample, step))
        summary = summarize_stream(records, collapse_threshold=args.collapse_threshold)
        summary.update(
            {
                "method": "eata",
                "stream_name": stream_name,
                "domains": [domain.lower() for domain in args.domains],
                "k_per_domain": int(k_per_domain),
                "seed": int(args.seed),
                "update_scope": args.update_scope,
            }
        )
        stream_results[stream_name] = summary
        save_stream_outputs(output_root / "eata" / stream_name, {"summary": summary, "records": records})

    overall = {
        "method": "eata",
        "domains": [domain.lower() for domain in args.domains],
        "streams": stream_results,
        "k_per_domain": int(k_per_domain),
        "seed": int(args.seed),
        "source_checkpoint": args.source_checkpoint,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "eata_table9_summary.json").open("w", encoding="utf-8") as handle:
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
    parser.add_argument("--e_margin", type=float, default=3.5)
    parser.add_argument("--selection_metric", choices=["mean_confidence", "peakiness", "hybrid"], default="hybrid")
    parser.add_argument("--selection_margin", type=float, default=0.75)
    parser.add_argument("--min_selection_weight", type=float, default=0.05)
    parser.add_argument("--d_margin", type=float, default=0.95)
    parser.add_argument("--fisher_alpha", type=float, default=2000.0)
    parser.add_argument("--fisher_samples", type=int, default=64)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num_workers", type=int, default=4)

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


