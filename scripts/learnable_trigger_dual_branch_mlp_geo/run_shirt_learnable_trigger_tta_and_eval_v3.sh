#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$(pwd):$(pwd)/output:${PYTHONPATH:-}"

DATA_ROOT="SHIRT_Dataset"
SOURCE_CHECKPOINT="output/dinov3_heatmap_source_v2/best_source_dino_heatmap.pth"

TTA_OUTPUT_DIR="output/dinov3_heatmap_shirt_dual_branch_tta_v3_safe_micro_tune1"
TTA_MODEL_PATH="${TTA_OUTPUT_DIR}/tta_final.pth"
EVAL_OUTPUT_DIR="${TTA_OUTPUT_DIR}/eval_after_tta"

echo "[1/2] Running SHIRT safer dual-branch OTTA..."
python risk-controlled-otta/adapt/learnable_trigger_single_model_tta_shirt_dual_branch_v3_safe.py \
  --data_root "${DATA_ROOT}" \
  --source_checkpoint "${SOURCE_CHECKPOINT}" \
  --roe all \
  --domain lightbox \
  --source_split train \
  --target_split shirt \
  --val_ratio 0.1 \
  --seed 42 \
  --output_dir "${TTA_OUTPUT_DIR}" \
  --model_name vit_base_patch16_dinov3.lvd1689m \
  --input_size 384 \
  --heatmap_size 96 \
  --heatmap_sigma 3.0 \
  --mid_channels 256 \
  --num_deconv_layers 2 \
  --num_keypoints 11 \
  --expand_ratio 1.25 \
  --update_scope decoder \
  --gate_usage hard \
  --gate_threshold 0.68 \
  --min_soft_gate_weight 0.10 \
  --min_lr_gate_scale 0.15 \
  --gate_hidden_dim 32 \
  --gate_dropout 0.10 \
  --gate_lr 5e-4 \
  --gate_weight_decay 1e-4 \
  --gate_warmup_steps 48 \
  --prototype_batch_size 32 \
  --prototype_max_samples 512 \
  --feature_reprojection_cap 50.0 \
  --feature_tvec_norm_cap 20.0 \
  --lr 3e-7 \
  --weight_decay 0.0 \
  --adapt_steps 1 \
  --teacher_momentum 0.999 \
  --save_teacher \
  --memory_capacity 64 \
  --memory_sample_size 8 \
  --min_memory_for_update 10 \
  --memory_min_quality 0.30 \
  --memory_min_confidence 0.55 \
  --memory_min_inliers 7 \
  --memory_max_reproj_error 5.0 \
  --memory_min_tvec_norm 4.5 \
  --memory_max_tvec_norm 8.5 \
  --memory_peak_conf_thresh 0.35 \
  --teacher_sharpen_temperature 0.85 \
  --lambda_self_training 0.14 \
  --lambda_geo 0.08 \
  --lambda_reg 0.05 \
  --tau 0.7 \
  --grad_clip_norm 1.0 \
  --geo_boost_min_inliers 7 \
  --geo_boost_max_reproj 5.0 \
  --trigger_confidence 0.20 \
  --trigger_min_inliers 6 \
  --trigger_reprojection_error 6.0 \
  --trigger_min_tvec_norm 4.5 \
  --trigger_max_tvec_norm 8.5 \
  --geometry_min_confidence 0.55 \
  --geometry_min_inliers 7 \
  --geometry_max_reproj_error 4.5 \
  --geometry_min_tvec_norm 4.5 \
  --geometry_max_tvec_norm 8.5 \
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

if [[ ! -f "${TTA_MODEL_PATH}" ]]; then
  echo "[ERROR] TTA checkpoint not found: ${TTA_MODEL_PATH}" >&2
  exit 1
fi

echo "[2/2] Evaluating TTA-adapted model..."
python risk-controlled-otta/eval/evaluate_shirt_dino_heatmap.py \
  --data_root "${DATA_ROOT}" \
  --model_path "${TTA_MODEL_PATH}" \
  --roe all \
  --domain lightbox \
  --split val \
  --val_ratio 0.1 \
  --seed 42 \
  --output_dir "${EVAL_OUTPUT_DIR}" \
  --input_size 384 \
  --heatmap_size 96 \
  --expand_ratio 1.25 \
  --num_vis 20 \
  --nms_kernel 3 \
  --subpixel_radius 2 \
  --min_confidence 0.05 \
  --top_k 8 \
  --min_points 6 \
  --ransac_reproj_error 6.0 \
  --ransac_iterations 100 \
  --ransac_confidence 0.999

echo "[OK] Done."
echo "TTA checkpoint: ${TTA_MODEL_PATH}"
echo "Evaluation summary: ${EVAL_OUTPUT_DIR}/val_results.json"
echo "Per-image results: ${EVAL_OUTPUT_DIR}/val_per_image_results.json"
