# Risk-Controlled OTTA for Structured Geometric Prediction

<p align="center">
  <img src="https://img.shields.io/badge/NeurIPS-2026%20Submission-blue" alt="NeurIPS 2026">
  <img src="https://img.shields.io/badge/status-review%20release-orange" alt="Review Release">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c" alt="PyTorch">
  <img src="https://img.shields.io/badge/task-Structured%20Geometric%20Prediction-purple" alt="Task">
</p>

This repository contains the review-version implementation for:

> **Risk-Controlled Online Test-Time Adaptation for Structured Geometric Prediction**

We study online test-time adaptation (OTTA) for structured geometric prediction, where local predictions are coupled by downstream geometric solvers. Using keypoint-based spacecraft pose estimation as a representative task, this repository implements a risk-controlled selective adaptation framework that estimates task-level geometric risk and triggers online adaptation only when intervention is likely to help.

---

## Highlights

- **Risk-controlled OTTA formulation** for structured geometric prediction.
- **Geometry-aware adaptation risk estimation** based on keypoint-PnP reliability.
- **Three trigger instantiations** for different accuracy, update sparsity, and stability trade-offs.
- **Quality-aware memory** for reliable pseudo supervision during online adaptation.
- **Unified evaluation pipeline** for source-only models, continuous OTTA, selective TTA baselines, and proposed variants.

---

## Method Family

This release organizes the proposed method as one risk-controlled OTTA framework with three trigger designs.

| Paper Name | Trigger Key | Description |
|---|---|---|
| `Risk-Controlled-OTTA (Threshold)` | `threshold` | Rule-based geometric trigger |
| `Risk-Controlled-OTTA (Learnable-Geo)` | `learnable_geo` | Learnable trigger based on task-level geometric risk |
| `Risk-Controlled-OTTA (Dual-Branch)` | `dual_branch` | Geometry-feature trigger with feature-distance context |

All three variants share the same task-level geometric risk backbone and differ in how the adaptation decision is made.

---

## Repository Structure

After uploading the contents of this release folder as the repository root, the layout should be:

```text
repo_root/
├── risk_controlled_otta/              # Core model, OTTA logic, experiments, analysis, and figure code
├── table2_redo_external_baselines/    # Standalone CoTTA / EATA single-domain runners
├── data/                              # Shared preprocessing helpers for external baseline runners
├── losses/                            # Legacy top-level loss dependencies
├── memory/                            # Legacy memory dependencies
├── scripts/                           # Launch scripts
├── docs/                              # Run instructions and additional notes
├── tangoPoints.mat                    # Spacecraft 3D keypoint template
└── requirements.txt                   # Python dependencies
```

This release preserves the original experiment logic while exposing a cleaner top-level structure for review and reproducibility.

---

## Installation

Create a Python environment and install dependencies:

```bash
conda create -n risk_otta python=3.10 -y
conda activate risk_otta

pip install -r requirements.txt
```

For CUDA-enabled PyTorch, please install the PyTorch version matching your local CUDA toolkit before installing the remaining dependencies.

---

## Datasets

The experiments use publicly available datasets. The data are **not redistributed** in this repository.

Please download the datasets separately and place them under a local `speedplusv2/` directory.

Expected layout:

```text
speedplusv2/
├── synthetic/
│   ├── images/
│   ├── train.json
│   └── validation.json
├── sunlamp/
│   ├── images/
│   └── test.json
├── lightbox/
│   ├── images/
│   └── test.json
└── shirt/
    ├── images/
    └── test.json
```

In the paper, the target domains are referred to as **Sunlamp**, **Lightbox**, and **SHIRT**.  
In the code, these correspond to the split keys:

```text
sunlamp
lightbox
shirt
```

---

## Quick Start

Please start from the run guide:

```text
docs/RUNBOOK.md
```

Main launch scripts are provided in:

```text
scripts/
```

A typical workflow is:

```bash
# 1. Activate environment
conda activate risk_otta

# 2. Check dataset paths
# Edit paths in the corresponding config or launcher script.

# 3. Run experiments
# See docs/RUNBOOK.md for detailed commands.
```

For Windows / PowerShell users, use the PowerShell launchers provided in `scripts/`.

---

## Reproducing Paper Experiments

The release supports the main experiment groups reported in the paper.

| Experiment Group | Description |
|---|---|
| Cross-domain evaluation | Synthetic source to Sunlamp, Lightbox, and SHIRT target domains |
| Risk diagnostics | Prediction confidence vs. task-level geometric risk analysis |
| Difficulty-stratified analysis | Easy / Medium / Hard sample groups |
| Stability and tail-risk metrics | p95 pose error, collapse rate, adaptation rate |
| Component ablations | Trigger type, quality-aware memory, geometry-guided loss |
| Mixed-domain streams | Block-wise and alternating target-domain stream protocols |
| External baselines | CoTTA / EATA runners under the unified keypoint-PnP pipeline |

For detailed commands, please refer to:

```text
docs/RUNBOOK.md
```

---

## Naming Conventions

| Item | Name |
|---|---|
| Repository | `risk-controlled-otta` |
| Python package | `risk_controlled_otta` |
| Method key | `risk_controlled_otta` |
| Trigger keys | `threshold`, `learnable_geo`, `dual_branch` |

Paper-style names:

```text
Risk-Controlled-OTTA (Threshold)
Risk-Controlled-OTTA (Learnable-Geo)
Risk-Controlled-OTTA (Dual-Branch)
```

---

## Important Notes for Review

This is a review-version code release. The goal is to provide a self-contained workspace that keeps the current paper experiments runnable while preserving the original implementation logic.

- The released folder is an organized packaging layer, not a full rewrite.
- Dataset files are not included.
- Some legacy top-level dependencies are kept to ensure compatibility with external baseline runners.
- Paths may need to be updated according to the local dataset location.
- The public release will be further cleaned after review.

---

## Citation

The paper is currently under review. A BibTeX entry will be added after publication.

```bibtex
@inproceedings{anonymous2026riskcontrolledotta,
  title     = {Risk-Controlled Online Test-Time Adaptation for Structured Geometric Prediction},
  author    = {Anonymous},
  booktitle = {Submitted to NeurIPS},
  year      = {2026}
}
```

---

## License

The license will be specified in the final public release.
