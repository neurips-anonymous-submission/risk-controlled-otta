# Runbook

This document explains how to run the released code from this folder after its
contents are uploaded as a repository root.

## 1. Expected layout

```text
repo_root/
|- risk_controlled_otta/
|- table2_redo_external_baselines/
|- data/
|- losses/
|- memory/
|- tangoPoints.mat
|- speedplusv2/                  # dataset, not necessarily versioned
|- output/                       # generated checkpoints / histories / tables
|- scripts/
|- docs/
`- requirements.txt
```

## 2. Environment setup

In PowerShell:

```powershell
Set-Location <repo_root>
$env:PYTHONPATH = (Get-Location).Path
pip install -r .\requirements.txt
```

## 2.1 Dataset setup

The repository does **not** bundle the datasets. Please download the public
datasets separately and place them under `speedplusv2/`.

This codebase uses:

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

Supported split keys in the released code are:

- `validation`
- `sunlamp`
- `lightbox`
- `shirt`

## 2.2 Naming used in this release

The paper-facing names are:

- `Risk-Controlled-OTTA (Threshold)`
- `Risk-Controlled-OTTA (Learnable-Geo)`
- `Risk-Controlled-OTTA (Dual-Branch)`

Their trigger semantics are:

- `Threshold`: rule-based geometric trigger
- `Learnable-Geo`: learnable geometry-risk trigger
- `Dual-Branch`: geometry-feature risk trigger

The internal trigger selectors remain:

- `threshold`
- `mlp_geo`
- `dual_branch`

## 3. Source training

```powershell
.\scripts\run_source_training.ps1
```

Optional example:

```powershell
.\scripts\run_source_training.ps1 `
  -DataRoot .\speedplusv2 `
  -OutputDir .\output\dinov3_heatmap_source_v2
```

## 4. Single-domain step histories

### Source-only

```powershell
.\scripts\run_step_history.ps1 -Method source_only -Domain sunlamp
```

### Original OTTA

```powershell
.\scripts\run_step_history.ps1 -Method original_otta -Domain lightbox
```

### Risk-Controlled-OTTA variants

```powershell
.\scripts\run_step_history.ps1 -Method risk_controlled_threshold -Domain sunlamp
.\scripts\run_step_history.ps1 -Method risk_controlled_learnable_geo -Domain sunlamp
.\scripts\run_step_history.ps1 -Method risk_controlled_dual_branch -Domain sunlamp
```

### Strict baselines

```powershell
.\scripts\run_step_history.ps1 -Method strict_ltta -Domain sunlamp
.\scripts\run_step_history.ps1 -Method strict_petta -Domain sunlamp
.\scripts\run_step_history.ps1 -Method strict_hybrid -Domain sunlamp
```

## 5. External single-domain baselines

These are the standalone CoTTA / EATA runners used by the paper.

```powershell
.\scripts\run_external_baseline_single_domain.ps1 -Method cotta -Domain sunlamp
.\scripts\run_external_baseline_single_domain.ps1 -Method eata  -Domain sunlamp
```

Expected outputs:

- `output/table2_redo_external/cotta_<domain>/`
- `output/table2_redo_external/eata_<domain>/`

## 6. Mixed-domain streams

```powershell
.\scripts\run_mixed_domain.ps1 -Method strict_ltta
.\scripts\run_mixed_domain.ps1 -Method strict_petta
.\scripts\run_mixed_domain.ps1 -Method strict_hybrid
```

Defaults:

- domains: `sunlamp lightbox shirt`
- streams: `forward reverse cyclic shifted`

## 7. Table 1 risk analysis

Current default launcher:

```powershell
.\scripts\run_table1_risk_analysis.ps1
```

This runs the beneficial-update variant used in the current draft.

## 8. Table 3 group analysis

```powershell
.\scripts\run_table3_group_analysis.ps1
```

This expects the required step histories and external baseline histories to
already exist.

## 9. Motivation figure

```powershell
.\scripts\run_motivation_curve.ps1
```

This reproduces the current Sunlamp motivation plot using:

- source-only
- geometric risk-aware OTTA (ours)
- recent selective method
- lightweight method

## 10. Figure 3

```powershell
.\scripts\run_figure3.ps1
```

## 11. Typical output locations

- source checkpoints: `output/dinov3_heatmap_source_v2/`
- step histories: `output/step_history_*`
- strict baselines: `output/strict_tta_baselines/`
- mixed-domain summaries: `output/mixed_domain_stream_results*`
- analysis tables: `output/analysis_table*`
- figures: `visualization_results_dinov3_heatmap/`

## 12. Recommended execution order

If you want to rebuild the paper artifacts from scratch:

1. source training or checkpoint preparation
2. single-domain step histories
3. external CoTTA / EATA runs
4. strict baseline single-domain runs
5. mixed-domain runs
6. Table 1 / Table 3 analyses
7. motivation and figure plotting

