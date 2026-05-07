#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$(pwd):$(pwd)/output:${PYTHONPATH:-}"

python risk-controlled-otta/adapt/learnable_trigger_single_model_tta_dino_heatmap_dual_branch_v2.py \
  --data_root speedplusv2 \
  --source_checkpoint output/dinov3_heatmap_source_v2/best_source_dino_heatmap.pth \
  --target_split lightbox \
  --output_dir output/dinov3_heatmap_dual_branch_lightbox_v2 \
  --update_scope decoder_last_block \
  --gate_usage soft_lr \
  --lr 2e-6 \
  --gate_lr 5e-4 \
  --gate_warmup_steps 512 \
  --memory_capacity 64 \
  --memory_sample_size 8 \
  --min_memory_for_update 8 \
  --memory_min_quality 0.10 \
  --memory_min_confidence 0.40 \
  --memory_min_inliers 6 \
  --memory_max_reproj_error 8.0 \
  --memory_min_tvec_norm 4.5 \
  --memory_max_tvec_norm 8.5 \
  --lambda_self_training 0.25 \
  --lambda_geo 0.35 \
  --lambda_reg 0.03 \
  --geo_boost_min_inliers 7 \
  --geo_boost_max_reproj 5.5 \
  --trigger_min_tvec_norm 4.5 \
  --trigger_max_tvec_norm 8.5

python risk-controlled-otta/eval/evaluate_dino_heatmap.py \
  --data_root speedplusv2 \
  --model_path output/dinov3_heatmap_dual_branch_lightbox_v2/tta_final.pth \
  --split lightbox \
  --output_dir evaluation_results_dinov3_heatmap/dual_branch_lightbox_v2 \
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
