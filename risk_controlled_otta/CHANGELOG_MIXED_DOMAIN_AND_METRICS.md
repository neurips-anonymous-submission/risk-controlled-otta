# Mixed-Domain + Metrics Update Notes

This file records the code changes added for:

- metrics alignment with the paper tables
- mixed-domain target-stream evaluation (Table 9 style)
- external TTA baselines adapted to the DINOv3 heatmap pose setting

It is intended to help with code review and manual merging.


## 1. Files Modified

### `risk_controlled_otta/eval/evaluate_dino_heatmap.py`

Updated the evaluation summary to include additional paper-aligned statistics.

Added summary fields:

- `collapse_threshold`
- `num_collapsed`
- `collapse_rate`
- `max_eq_deg`
- `p95_ep`
- `max_ep`
- `p95_e_star_p`
- `max_e_star_p`
- `median_num_selected_points`
- `p95_num_selected_points`
- `median_num_ransac_inliers`
- `p95_num_ransac_inliers`
- `avg_inlier_ratio`
- `median_inlier_ratio`
- `p95_inlier_ratio`
- `median_mean_selected_confidence`
- `p95_mean_selected_confidence`
- `avg_max_reprojection_error`
- `median_max_reprojection_error`
- `p95_max_reprojection_error`
- `median_tvec_norm`

Added parser argument:

- `--collapse_threshold`

Purpose:

- directly support table fields such as `p95 E_p^*`, `max E_p^*`, and `collapse rate`
- reduce manual post-processing after evaluation


### `risk_controlled_otta/adapt/triggered_single_model_tta_dino_heatmap.py`

Updated trigger/adaptation accounting to explicitly distinguish:

- `triggered`
- `adapted`
- real optimizer execution

Added/updated logic:

- `executed_step` returned from the adaptation function
- `optimizer_step_executed` stored in history
- `adapted` now reflects actual optimizer step execution
- `trigger_ratio` added to summary
- `adapt_ratio` added to summary

Purpose:

- align `num_triggered` with final trigger decisions
- align `num_adapted` with real online updates
- support paper definitions for `trigger_ratio` and `adapt_ratio`


### `risk_controlled_otta/adapt/learnable_trigger_single_model_tta_dino_heatmap.py`

Updated trigger/adaptation accounting for the learnable-trigger method.

Added/updated logic:

- explicit `triggered = should_adapt`
- `executed_step` returned from the adaptation function
- `optimizer_step_executed` stored in history
- `num_triggered` added to summary
- `trigger_ratio` added to summary
- `adapt_ratio` added to summary

Purpose:

- keep warm-up heuristic trigger decisions and learned trigger decisions under one unified trigger count
- ensure `adapted` means actual optimizer update


## 2. Files Added

### `risk_controlled_otta/experiments/__init__.py`

Package marker for experiment scripts.


### `risk_controlled_otta/experiments/mixed_domain_stream_eval.py`

Standalone mixed-domain stream evaluation runner for Table 9 style experiments.

Supports:

- unified sample count across domains
- fixed per-domain permutation by seed
- four stream protocols:
  - `forward`
  - `reverse`
  - `cyclic`
  - `shifted`
- source-only evaluation
- original OTTA style online adaptation
- triggered single-model OTTA
- learnable-trigger OTTA

Outputs:

- per-stream `summary.json`
- per-stream `step_history.json`
- global `*_table9_summary.json`

Recorded per-step fields include:

- `domain`
- `triggered`
- `adapted`
- `trigger_score`
- `gate_weight`
- `memory_size`
- `mean_confidence`
- `num_ransac_inliers`
- `inlier_ratio`
- `mean_reprojection_error`
- `max_reprojection_error`
- `used_fallback_epnp`
- `pose_failed`
- `trigger_reasons`
- `e_star_t_bar`
- `e_star_q`
- `e_star_q_deg`
- `e_star_p`
- `collapse`

Purpose:

- run mixed-domain block-wise and alternating protocols without modifying existing train/eval entrypoints


### `risk_controlled_otta/experiments/external_tta_methods/__init__.py`

Package marker for external TTA baselines.


### `risk_controlled_otta/experiments/external_tta_methods/cotta_mixed_stream.py`

CoTTA-style baseline adapted for the DINOv3 heatmap pose setting.

Adaptation notes:

- uses EMA teacher
- uses augmentation averaging
- uses stochastic source-weight restoration
- replaces classification-output consistency with heatmap spatial consistency

Purpose:

- provide a CoTTA-style continual adaptation baseline for mixed-domain stream evaluation


### `risk_controlled_otta/experiments/external_tta_methods/eata_mixed_stream.py`

EATA-style baseline adapted for the DINOv3 heatmap pose setting.

Adaptation notes:

- uses heatmap entropy instead of classification entropy
- uses redundancy filtering on flattened heatmap distributions
- estimates Fisher regularization from source-domain heatmap regression

Purpose:

- provide an EATA-style efficient test-time adaptation baseline for mixed-domain stream evaluation


## 3. Merge-Sensitive Notes

### Metrics logic

Paper-aligned summary fields were added in:

- `risk_controlled_otta/eval/evaluate_dino_heatmap.py`

If another branch edited the evaluation summary, please merge carefully around the `summary = {...}` block and the parser arguments.


### Trigger/adaptation accounting

Paper-aligned trigger/update counting was added in:

- `risk_controlled_otta/adapt/triggered_single_model_tta_dino_heatmap.py`
- `risk_controlled_otta/adapt/learnable_trigger_single_model_tta_dino_heatmap.py`

Important semantics:

- `triggered`: final trigger says the sample should be adapted
- `adapted`: optimizer step was actually executed
- `trigger_ratio = num_triggered / num_samples`
- `adapt_ratio = num_adapted / num_samples`


### New experiment entrypoints

The mixed-domain and external-baseline scripts were added as new files only.
They do not replace the original train/eval/adapt entrypoints.


## 4. Recommended Sync List

If sharing only the relevant updates with collaborators, sync at least:

- `risk_controlled_otta/eval/evaluate_dino_heatmap.py`
- `risk_controlled_otta/adapt/triggered_single_model_tta_dino_heatmap.py`
- `risk_controlled_otta/adapt/learnable_trigger_single_model_tta_dino_heatmap.py`
- `risk_controlled_otta/experiments/mixed_domain_stream_eval.py`
- `risk_controlled_otta/experiments/external_tta_methods/cotta_mixed_stream.py`
- `risk_controlled_otta/experiments/external_tta_methods/eata_mixed_stream.py`
- `risk_controlled_otta/CHANGELOG_MIXED_DOMAIN_AND_METRICS.md`


