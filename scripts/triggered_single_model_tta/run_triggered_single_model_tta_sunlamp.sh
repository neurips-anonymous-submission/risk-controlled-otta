#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$(pwd):$(pwd)/output:${PYTHONPATH:-}"

# 1. 运行 Triggered Single-Model TTA (SPEED+ 数据集)
python risk-controlled-otta/adapt/triggered_single_model_tta_dino_heatmap.py \
  --data_root speedplusv2 \
  --source_checkpoint output/dinov3_heatmap_source_v2/best_source_dino_heatmap.pth \
  --target_split sunlamp \
  --output_dir output/dinov3_heatmap_triggered_sunlamp_v1 \
  --update_scope decoder \
  --lr 5e-6 \
  --adapt_steps 1 \
  --memory_capacity 64 \
  --memory_sample_size 8 \
  --min_memory_for_update 6 \
  --memory_min_quality 0.1 \
  --lambda_self_training 1.0 \
  --lambda_geo 0.1 \
  --lambda_reg 0.05 \
  --tau 0.7 \
  --grad_clip_norm 1.0 \
  --trigger_confidence 0.20 \
  --trigger_min_inliers 6 \
  --trigger_reprojection_error 6.5 \
  --quality_reprojection_cap 50.0 \
  --num_workers 4

# 2. 运行对齐最新评估矩阵的评估脚本
python risk-controlled-otta/eval/evaluate_dino_heatmap.py \
  --data_root speedplusv2 \
  --model_path output/dinov3_heatmap_triggered_sunlamp_v1/tta_final.pth \
  --split sunlamp \
  --output_dir evaluation_results_dinov3_heatmap/triggered_sunlamp_v1 \
  --num_vis 20 \
  --nms_kernel 3 \
  --subpixel_radius 2 \
  --min_confidence 0.05 \
  --top_k 8 \
  --min_points 6 \
  --ransac_reproj_error 6.0 \
  --ransac_iterations 100 \
  --ransac_confidence 0.999 \
  --collapse_threshold 0.1
