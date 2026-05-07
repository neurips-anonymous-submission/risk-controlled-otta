#!/usr/bin/env bash
set -euo pipefail

# SHIRT Learnable-Trigger TTA: Dual-Branch version

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONPATH="$(pwd):$(pwd)/output:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-SHIRT_Dataset}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-output/dinov3_heatmap_source_v2/best_source_dino_heatmap.pth}"
TTA_SCRIPT="${TTA_SCRIPT:-risk-controlled-otta/adapt/learnable_trigger_single_model_tta_shirt_dino_heatmap.py}"

ROE="${ROE:-all}"
DOMAIN="${DOMAIN:-lightbox}"
TARGET_SPLIT="${TARGET_SPLIT:-val}"
VAL_RATIO="${VAL_RATIO:-0.1}"
SEED="${SEED:-42}"

TTA_OUTPUT_DIR="${TTA_OUTPUT_DIR:-output/dinov3_heatmap_shirt_dual_branch_tta_v1}"
TTA_MODEL_PATH="${TTA_OUTPUT_DIR}/shirt_tta_final.pth"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${TTA_OUTPUT_DIR}/eval_after_tta}"

PROTOTYPE_DOMAIN="${PROTOTYPE_DOMAIN:-synthetic}"
PROTOTYPE_SPLIT="${PROTOTYPE_SPLIT:-train}"

mkdir -p "${TTA_OUTPUT_DIR}" "${EVAL_OUTPUT_DIR}"

echo "[1/2] Running SHIRT learnable-trigger TTA: Dual-Branch"
python "${TTA_SCRIPT}" \
  --data_root "${DATA_ROOT}" \
  --source_checkpoint "${SOURCE_CHECKPOINT}" \
  --roe "${ROE}" \
  --domain "${DOMAIN}" \
  --target_split "${TARGET_SPLIT}" \
  --val_ratio "${VAL_RATIO}" \
  --seed "${SEED}" \
  --output_dir "${TTA_OUTPUT_DIR}" \
  --model_name vit_base_patch16_dinov3.lvd1689m \
  --expand_ratio 1.25 \
  --update_scope decoder \
  --trigger_mode dual_branch \
  --gate_usage soft_loss \
  --gate_threshold 0.5 \
  --min_soft_gate_weight 0.1 \
  --gate_hidden_dim 16 \
  --gate_lr 1e-4 \
  --gate_weight_decay 1e-4 \
  --gate_warmup_steps 10 \
  --train_gate \
  --prototype_domain "${PROTOTYPE_DOMAIN}" \
  --prototype_split "${PROTOTYPE_SPLIT}" \
  --prototype_batch_size 32 \
  --prototype_max_samples 1000 \
  --feature_reprojection_cap 50.0 \
  --feature_tvec_norm_cap 20.0 \
  --lr 5e-6 \
  --weight_decay 0.0 \
  --adapt_steps 1 \
  --memory_capacity 32 \
  --memory_sample_size 8 \
  --min_memory_for_update 4 \
  --memory_min_quality 0.1 \
  --lambda_self_training 1.0 \
  --lambda_geo 0.1 \
  --lambda_reg 0.05 \
  --tau 0.7 \
  --grad_clip_norm 1.0 \
  --trigger_confidence 0.15 \
  --trigger_min_inliers 5 \
  --trigger_reprojection_error 8.0 \
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

echo "[2/2] Evaluating Dual-Branch TTA-adapted model"
python risk-controlled-otta/eval/evaluate_shirt_dino_heatmap.py \
  --data_root "${DATA_ROOT}" \
  --model_path "${TTA_MODEL_PATH}" \
  --roe "${ROE}" \
  --domain "${DOMAIN}" \
  --split "${TARGET_SPLIT}" \
  --val_ratio "${VAL_RATIO}" \
  --seed "${SEED}" \
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

echo "[OK] Dual-Branch TTA done."
echo "TTA checkpoint: ${TTA_MODEL_PATH}"
echo "TTA history: ${TTA_OUTPUT_DIR}/shirt_learnable_trigger_history.json"
echo "Evaluation summary: ${EVAL_OUTPUT_DIR}/${TARGET_SPLIT}_results.json"
echo "Per-image results: ${EVAL_OUTPUT_DIR}/${TARGET_SPLIT}_per_image_results.json"
