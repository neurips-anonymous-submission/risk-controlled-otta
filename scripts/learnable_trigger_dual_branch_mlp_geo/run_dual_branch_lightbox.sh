#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

export PYTHONPATH="$(pwd):$(pwd)/output:${PYTHONPATH:-}"

python risk-controlled-otta/adapt/learnable_trigger_single_model_tta_dino_heatmap_dual_branch.py \
  --data_root speedplusv2 \
  --source_checkpoint output/dinov3_heatmap_source_v2/best_source_dino_heatmap.pth \
  --target_split lightbox \
  --output_dir output/dinov3_heatmap_dual_branch_lightbox_v1 \
  --trigger_mode dual_branch \
  --gate_usage soft_loss \
  --update_scope decoder_last_block \
  --lr 2e-6 \
  --gate_lr 5e-4 \
  --gate_warmup_steps 512 \
  --memory_capacity 64 \
  --memory_sample_size 8 \
  --min_memory_for_update 8 \
  --memory_min_quality 0.10 \
  --memory_min_confidence 0.35 \
  --memory_min_inliers 6 \
  --memory_max_reproj_error 10.0 \
  --lambda_self_training 0.5 \
  --lambda_geo 0.2 \
  --lambda_reg 0.05



python risk-controlled-otta/eval/evaluate_dino_heatmap.py \
  --data_root speedplusv2 \
  --model_path output/dinov3_heatmap_dual_branch_lightbox_v1/tta_final.pth \
  --split lightbox \
  --output_dir evaluation_results_dinov3_heatmap/dual_branch_lightbox_v1
