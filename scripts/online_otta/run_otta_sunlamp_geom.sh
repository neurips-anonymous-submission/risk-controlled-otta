#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$(pwd):$(pwd)/output:${PYTHONPATH:-}"

# 1. 运行在线 TTA (Geometry-Gated)
python risk-controlled-otta/adapt/online_tta_dino_heatmap_geom.py \
  --data_root speedplusv2 \
  --source_checkpoint output/dinov3_heatmap_source_v2/best_source_dino_heatmap.pth \
  --target_split sunlamp \
  --output_dir output/dinov3_heatmap_tta_sunlamp_geom_v1 \
  --adapt_modules decoder \
  --lr 1e-6 \
  --lambda_st 1.0 \
  --lambda_ca 0.01 \
  --ema_alpha 0.999 \
  --memory_capacity 16 \
  --memory_sample_size 16 \
  --bank_push_conf_thresh 0.20 \
  --memory_conf_thresh 0.20 \
  --min_reliable_samples 4 \
  --gate_min_confidence_mean 0.75 \
  --gate_min_inliers 6 \
  --gate_max_reproj_error 12.0 \
  --mask_ratio 0.8 \
  --tau 0.7

# 2. 运行对齐最新评估矩阵的评估脚本
python risk-controlled-otta/eval/evaluate_dino_heatmap.py \
  --data_root speedplusv2 \
  --model_path output/dinov3_heatmap_tta_sunlamp_geom_v1/tta_final.pth \
  --split sunlamp \
  --output_dir evaluation_results_dinov3_heatmap/otta_sunlamp_geom_v1 \
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
