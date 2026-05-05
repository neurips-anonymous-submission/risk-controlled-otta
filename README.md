# Risk-Controlled-OTTA Release

This folder is a **self-contained release workspace** for the paper code.

It is designed so that you can upload **the contents of this folder as the
repository root**. After upload, the top-level layout should look like:

```text
repo_root/
|- risk_controlled_otta/
|- table2_redo_external_baselines/
|- data/
|- losses/
|- memory/
|- tangoPoints.mat
|- scripts/
|- docs/
`- requirements.txt
```

The goal here is practical:

- keep the original experiment logic unchanged
- keep the current paper code runnable
- expose a clean top-level structure for collaborators, reviewers, or release

## Canonical method naming

This release organizes the method family as one risk-controlled OTTA framework
with three instantiations:

- `Risk-Controlled-OTTA-Threshold`
  - rule-based geometric trigger
- `Risk-Controlled-OTTA-Learnable-Geo`
  - learnable geometry-risk trigger
- `Risk-Controlled-OTTA-Dual-Branch`
  - geometry-feature risk trigger

Recommended naming conventions:

- repository name: `risk-controlled-otta`
- Python package: `risk_controlled_otta`
- method key: `risk_controlled_otta`
- trigger key: `threshold`, `mlp_geo`, `dual_branch`

Paper-style method names:

- `Risk-Controlled-OTTA (Threshold)`
- `Risk-Controlled-OTTA (Learnable-Geo)`
- `Risk-Controlled-OTTA (Dual-Branch)`

## What is included

- `risk_controlled_otta/`
  - core model, adaptation logic, paper experiments, analyses, and figure code
- `table2_redo_external_baselines/`
  - standalone CoTTA / EATA single-domain runners used by the paper
- `data/`
  - shared preprocessing helpers required by the external baseline runners
- `losses/`, `memory/`, `tangoPoints.mat`
  - legacy top-level dependencies still imported by the external baseline code
- `scripts/`
  - clean PowerShell launchers
- `docs/`
  - run instructions
- `requirements.txt`
  - lightweight dependency list for this release

## Datasets

The experiments use **publicly available datasets** and the data are **not
redistributed in this repository**. Please download them separately and place
them under a local `speedplusv2/` directory.

The released code expects:

- **SPEED++**
  - `synthetic/` for source training and validation
  - `sunlamp/` for the Sunlamp target split
  - `lightbox/` for the Lightbox target split
- **SHIRT**
  - `shirt/` for the SHIRT target split

Expected layout:

```text
speedplusv2/
|- synthetic/
|  |- images/
|  |- train.json
|  `- validation.json
|- sunlamp/
|  |- images/
|  `- test.json
|- lightbox/
|  |- images/
|  `- test.json
`- shirt/
   |- images/
   `- test.json
```

In the paper, we refer to the target domains as **Sunlamp**, **Lightbox**, and
**SHIRT**. In the code, these correspond to the split keys
`sunlamp`, `lightbox`, and `shirt`.

## Start here

- run guide: `docs/RUNBOOK.md`
- main launcher folder: `scripts/`

## Important note

This release folder intentionally preserves the existing implementation logic.
It is an organized packaging layer, not a rewrite.

