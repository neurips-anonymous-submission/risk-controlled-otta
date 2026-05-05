# `risk_controlled_otta`

<p align="center">
  <img src="https://img.shields.io/badge/module-core%20package-blue" alt="Core Package">
  <img src="https://img.shields.io/badge/backbone-DINOv3-purple" alt="DINOv3">
  <img src="https://img.shields.io/badge/task-keypoint%20pose%20estimation-green" alt="Task">
  <img src="https://img.shields.io/badge/setting-online%20TTA-orange" alt="Online TTA">
</p>

This package contains the core implementation of **Risk-Controlled OTTA** for structured geometric prediction.

It includes the DINO-based heatmap pose-estimation pipeline, keypoint decoding, PnP-based pose recovery, task-level geometric risk estimation, quality-aware memory, and online adaptation logic used in the paper:

> **Risk-Controlled Online Test-Time Adaptation for Structured Geometric Prediction**

---

## Overview

Risk-Controlled OTTA treats online adaptation as a **risk-controlled intervention problem**.

Instead of adapting on every incoming target sample or relying on prediction confidence alone, the framework estimates **task-level geometric risk** from the keypoint-PnP pipeline and triggers adaptation only when intervention is likely to help.

The core pipeline is:

```text
Target image
    ↓
DINO heatmap pose model
    ↓
Heatmap decoding → 2D keypoints
    ↓
RANSAC-PnP pose recovery
    ↓
Geometry-aware risk estimation
    ↓
Selective adaptation with quality-aware memory
```

---

## Method Variants

The framework is instantiated with three trigger designs.

| Paper Name | Trigger Key | Description |
|---|---|---|
| `Risk-Controlled-OTTA (Threshold)` | `threshold` | Rule-based geometric trigger |
| `Risk-Controlled-OTTA (Learnable-Geo)` | `learnable_geo` | Learnable trigger based on task-level geometric risk |
| `Risk-Controlled-OTTA (Dual-Branch)` | `dual_branch` | Geometry-feature trigger with feature-distance context |

All three variants share the same task-level geometric risk backbone and differ only in how the adaptation decision is made.

---

## Package Structure

```text
risk_controlled_otta/
├── models/
│   └── dino_pose_model.py
├── data/
│   └── dino_heatmap_dataset.py
├── losses/
│   └── heatmap_loss.py
├── train/
│   └── train_dino_heatmap.py
├── eval/
│   ├── evaluate_dino_heatmap.py
│   └── sweep_pose_params.py
├── adapt/
│   ├── triggered_single_model_tta_dino_heatmap.py
│   └── learnable_trigger_single_model_tta_dino_heatmap.py
└── README.md
```

---

## Main Components

| File | Role |
|---|---|
| `models/dino_pose_model.py` | DINO-based heatmap keypoint pose model |
| `data/dino_heatmap_dataset.py` | SPEED+/SHIRT heatmap dataset loader |
| `losses/heatmap_loss.py` | Heatmap supervision losses |
| `train/train_dino_heatmap.py` | Source-domain heatmap model training |
| `eval/evaluate_dino_heatmap.py` | Source-only and adapted model evaluation |
| `eval/sweep_pose_params.py` | PnP / decoding parameter sweep |
| `adapt/triggered_single_model_tta_dino_heatmap.py` | Rule-based risk-triggered OTTA |
| `adapt/learnable_trigger_single_model_tta_dino_heatmap.py` | Learnable-Geo and Dual-Branch OTTA |

---

## Core Risk Signals

The adaptation trigger uses task-level geometric reliability cues from the keypoint-PnP pipeline.

Main cues include:

- mean heatmap peak score;
- RANSAC inlier count / inlier ratio;
- mean and maximum reprojection error;
- fallback EPnP indicator;
- PnP failure indicator;
- normalized translation statistics;
- optional feature-distance cues for the Dual-Branch trigger.

These cues estimate whether a target sample is risky at the downstream pose level, rather than only at the heatmap-confidence level.

---

## Default Configuration

The main experiments use the following default setting.

| Item | Value |
|---|---|
| Backbone | `vit_base_patch16_dinov3.lvd1689m` |
| Input size | `384` |
| Heatmap size | `96` |
| Heatmap sigma | `3.0` |
| Encoder learning rate | `5e-5` |
| Decoder learning rate | `5e-4` |
| Warmup steps | `1000` |
| Default update scope | Decoder only |
| Default online setting | Single-model adaptation, no teacher |

---

## Training Source Model

Example source-domain training command:

```bash
python -m risk_controlled_otta.train.train_dino_heatmap \
  --data_root speedplusv2 \
  --output_dir output/dinov3_heatmap_source \
  --model_name vit_base_patch16_dinov3.lvd1689m \
  --pretrained_path weights/dinov3/vit_base_patch16_dinov3/model.safetensors \
  --batch_size 32
```

---

## Evaluation

Example source-only evaluation command:

```bash
python -m risk_controlled_otta.eval.evaluate_dino_heatmap \
  --data_root speedplusv2 \
  --split sunlamp \
  --checkpoint output/dinov3_heatmap_source/best_source_dino_heatmap.pth \
  --output_dir output/eval_source_sunlamp
```

Use one of the following target split keys:

```text
sunlamp
lightbox
shirt
```

---

## Online Adaptation

### Threshold Trigger

```bash
python -m risk_controlled_otta.adapt.triggered_single_model_tta_dino_heatmap \
  --data_root speedplusv2 \
  --target_split sunlamp \
  --checkpoint output/dinov3_heatmap_source/best_source_dino_heatmap.pth \
  --trigger threshold \
  --output_dir output/otta_threshold_sunlamp
```

### Learnable-Geo Trigger

```bash
python -m risk_controlled_otta.adapt.learnable_trigger_single_model_tta_dino_heatmap \
  --data_root speedplusv2 \
  --target_split sunlamp \
  --checkpoint output/dinov3_heatmap_source/best_source_dino_heatmap.pth \
  --trigger learnable_geo \
  --output_dir output/otta_learnable_geo_sunlamp
```

### Dual-Branch Trigger

```bash
python -m risk_controlled_otta.adapt.learnable_trigger_single_model_tta_dino_heatmap \
  --data_root speedplusv2 \
  --target_split sunlamp \
  --checkpoint output/dinov3_heatmap_source/best_source_dino_heatmap.pth \
  --trigger dual_branch \
  --output_dir output/otta_dual_branch_sunlamp
```

---

## Naming Conventions

| Concept | Name |
|---|---|
| Package | `risk_controlled_otta` |
| Method key | `risk_controlled_otta` |
| Rule-based trigger | `threshold` |
| Learnable geometry-risk trigger | `learnable_geo` |
| Geometry-feature trigger | `dual_branch` |

Paper-style names:

```text
Risk-Controlled-OTTA (Threshold)
Risk-Controlled-OTTA (Learnable-Geo)
Risk-Controlled-OTTA (Dual-Branch)
```

---

## Notes

- This package preserves the experiment logic used in the paper.
- The implementation is organized for review and reproducibility, not fully rewritten as a general-purpose library.
- Dataset paths, checkpoint paths, and local environment settings may need to be adjusted before running.
- Some launch commands may also be available in the top-level `scripts/` directory.

---

## See Also

For repository-level instructions, dataset layout, and reproduction notes, see:

```text
../README.md
../docs/RUNBOOK.md
```
