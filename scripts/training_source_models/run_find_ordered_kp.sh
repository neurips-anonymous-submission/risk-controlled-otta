#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

python find_ordered_kp_case.py \
  --data_root speedplusv2 \
  --split sunlamp \
  --source_model output/dinov3_heatmap_source_v2/best_source_dino_heatmap.pth \
  --otta_model output/dinov3_heatmap_tta_sunlamp_v3/tta_final.pth \
  --dinov3_model output/dinov3_heatmap_learnable_dual_sunlamp_v1/tta_final.pth \
  --output_dir ordered_case_vis_sunlamp \
  --min_source_minus_otta_eq 0.3 \
  --min_otta_minus_dinov3_eq 0.1 \
  --min_source_minus_otta_kp 0.5 \
  --min_otta_minus_dinov3_kp 0.2