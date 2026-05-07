#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$(pwd):$(pwd)/output:${PYTHONPATH:-}"

DATA_ROOT="SHIRT_Dataset"
SOURCE_CHECKPOINT="output/dinov2_heatmap_speedplusv2_source/best_source_dino_heatmap.pth"
DOMAIN="lightbox"
TTA_OUTPUT_DIR="output/dinov2_heatmap_online_tta_geom_${DOMAIN}"

mkdir -p "${TTA_OUTPUT_DIR}"

python risk-controlled-otta/adapt/online_tta_shirt_dinov2_heatmap_geom.py \
  --data_root "${DATA_ROOT}" \
  --source_checkpoint "${SOURCE_CHECKPOINT}" \
  --roe all \
  --domain "${DOMAIN}" \
  --output_dir "${TTA_OUTPUT_DIR}" \
  --lr 1e-5 \
  --memory_capacity 32 \
  --memory_sample_size 16 \
  --min_reliable_samples 4 \
  --ema_alpha 0.999 \
  --lambda_st 1.0 \
  --lambda_ca 0.01 \
  --pnp_min_confidence 0.05 \
  --top_k 8 \
  --min_points 6 \
  --ransac_reproj_error 6.0 \
  --ransac_iterations 100 \
  --ransac_confidence 0.999 \
  --gate_min_confidence_mean 0.75 \
  --gate_min_inliers 6 \
  --gate_max_reproj_error 12.0 \
  --num_workers 4




DATA_ROOT="SHIRT_Dataset"
DOMAIN="lightbox"
TTA_OUTPUT_DIR="output/dinov2_heatmap_online_tta_geom_${DOMAIN}"
MODEL_PATH="${TTA_OUTPUT_DIR}/tta_final.pth"
EVAL_OUTPUT_DIR="evaluation_results_shirt_online_tta_geom/${DOMAIN}_val"

mkdir -p "${EVAL_OUTPUT_DIR}"

python risk-controlled-otta/eval/evaluate_shirt_dinov2_heatmap.py \
  --data_root "${DATA_ROOT}" \
  --model_path "${MODEL_PATH}" \
  --roe all \
  --domain "${DOMAIN}" \
  --split val \
  --val_ratio 0.1 \
  --seed 42 \
  --output_dir "${EVAL_OUTPUT_DIR}" \
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
