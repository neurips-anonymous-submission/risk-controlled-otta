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

We study online test-time adaptation (OTTA) for structured geometric prediction, where local predictions are coupled by downstream geometric solvers. Using keypoint-based spacecraft pose estimation as a representative task, this code implements a risk-controlled selective adaptation framework that estimates task-level geometric risk and triggers online adaptation only when intervention is likely to help.

---

## Highlights

- **Risk-controlled OTTA formulation** for structured geometric prediction.
- **Task-level geometric risk cues** based on keypoint-PnP reliability.
- **Three trigger instantiations** for different accuracy, update sparsity, and stability trade-offs.
- **Quality-aware memory** for reliable pseudo supervision during online adaptation.
- **Unified evaluation pipeline** for source-only, continuous OTTA, selective TTA baselines, and proposed variants.

---

## Method Variants

This release organizes the proposed method family as one risk-controlled OTTA framework with three trigger designs.

| Paper Name | Trigger Key | Description |
|---|---|---|
| `Risk-Controlled-OTTA (Threshold)` | `threshold` | Rule-based geometric trigger |
| `Risk-Controlled-OTTA (Learnable-Geo)` | `learnable_geo` | Learnable trigger based on task-level geometric risk |
| `Risk-Controlled-OTTA (Dual-Branch)` | `dual_branch` | Geometry-feature trigger with additional feature-distance context |

All three variants share the same task-level geometric risk backbone and differ only in how the adaptation decision is made.

---

## Repository Structure

After uploading the contents of this release folder as the repository root, the layout should be:

```text
repo_root/
├── risk_controlled_otta/              # Core model, OTTA logic, experiments, analysis, and figure code
├── table2_redo_external_baselines/    # Standalone CoTTA / EATA single-domain runners
├── data/                              # Shared preprocessing helpers for external baselines
├── losses/                            # Legacy top-level loss dependencies
├── memory/                            # Legacy memory dependencies
├── scripts/                           # Clean launch scripts
├── docs/                              # Run instructions and additional notes
├── tangoPoints.mat                    # Spacecraft 3D keypoint template
└── requirements.txt                   # Python dependencies
