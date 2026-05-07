#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$(pwd):$(pwd)/output:${PYTHONPATH:-}"

DATA_ROOT="SHIRT_Dataset"
SOURCE_CHECKPOINT="output/dinov3_heatmap_source_v2/best_source_dino_heatmap.pth"
TTA_OUTPUT_DIR="output/dinov3_heatmap_shirt_triggered_single_model_tta"

mkdir -p "${TTA_OUTPUT_DIR}"

# 1/2 Running SHIRT triggered single-model TTA
python risk-controlled-otta/adapt/triggered_single_model_tta_shirt_dino_heatmap.py \
  --data_root "${DATA_ROOT}" \
  --source_checkpoint "${SOURCE_CHECKPOINT}" \
  --roe all \
  --domain lightbox \
  --target_split shirt \
  --val_ratio 0.1 \
  --seed 42 \
  --output_dir "${TTA_OUTPUT_DIR}" \
  --model_name vit_base_patch16_dinov3.lvd1689m \
  --expand_ratio 1.25 \
  --update_scope decoder \
  --lr 5e-6 \
  --weight_decay 0.0 \
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
  --nms_kernel 3 \
  --subpixel_radius 2 \
  --min_confidence 0.05 \
  --top_k 8 \
  --min_points 6 \
  --ransac_reproj_error 6.0 \
  --ransac_iterations 100 \
  --ransac_confidence 0.999 \
  --num_workers 4

# 2/2 Evaluating SHIRT TTA-adapted model
python risk-controlled-otta/eval/evaluate_shirt_dino_heatmap.py \
  --data_root "${DATA_ROOT}" \
  --model_path "${TTA_OUTPUT_DIR}/shirt_tta_final.pth" \
  --roe all \
  --domain lightbox \
  --split val \
  --val_ratio 0.1 \
  --seed 42 \
  --output_dir "${TTA_OUTPUT_DIR}/eval_after_tta" \
  --expand_ratio 1.25 \
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
