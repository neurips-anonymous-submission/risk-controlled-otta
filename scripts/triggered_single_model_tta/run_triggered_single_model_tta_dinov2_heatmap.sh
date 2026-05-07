#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONPATH="$(pwd):$(pwd)/output:${PYTHONPATH:-}"

# =========================
# Paths
# =========================
DATA_ROOT="${DATA_ROOT:-speedplusv2}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-output/dinov2_heatmap_speedplusv2_source/best_source_dino_heatmap.pth}"

# TTA target split: sunlamp / lightbox / validation
TARGET_SPLIT="${TARGET_SPLIT:-sunlamp}"

# TTA output
TTA_OUTPUT_DIR="${TTA_OUTPUT_DIR:-output/dinov2_heatmap_triggered_single_model_tta_${TARGET_SPLIT}}"

# Final adapted checkpoint
TTA_CHECKPOINT="${TTA_CHECKPOINT:-${TTA_OUTPUT_DIR}/tta_final.pth}"

# Eval settings
EVAL_SPLIT="${EVAL_SPLIT:-${TARGET_SPLIT}}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${TTA_OUTPUT_DIR}/eval_${EVAL_SPLIT}}"

# Collapse threshold for evaluation
COLLAPSE_THRESHOLD="${COLLAPSE_THRESHOLD:-0.1}"

mkdir -p "${TTA_OUTPUT_DIR}"
mkdir -p "${EVAL_OUTPUT_DIR}"

echo "=================================================="
echo "[1/2] Running triggered single-model TTA"
echo "DATA_ROOT=${DATA_ROOT}"
echo "SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT}"
echo "TARGET_SPLIT=${TARGET_SPLIT}"
echo "TTA_OUTPUT_DIR=${TTA_OUTPUT_DIR}"
echo "=================================================="

python risk-controlled-otta/tta/triggered_single_model_tta_dinov2_heatmap.py \
  --data_root "${DATA_ROOT}" \
  --source_checkpoint "${SOURCE_CHECKPOINT}" \
  --target_split "${TARGET_SPLIT}" \
  --output_dir "${TTA_OUTPUT_DIR}" \
  --update_scope decoder \
  --lr 1e-5 \
  --weight_decay 0.0 \
  --adapt_steps 1 \
  --memory_capacity 32 \
  --memory_sample_size 8 \
  --min_memory_for_update 4 \
  --memory_min_quality 0.01 \
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

if [ ! -f "${TTA_CHECKPOINT}" ]; then
  echo "[ERROR] TTA checkpoint not found: ${TTA_CHECKPOINT}"
  exit 1
fi

echo "=================================================="
echo "[2/2] Running evaluation on adapted checkpoint"
echo "TTA_CHECKPOINT=${TTA_CHECKPOINT}"
echo "EVAL_SPLIT=${EVAL_SPLIT}"
echo "EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR}"
echo "=================================================="

python risk-controlled-otta/eval/evaluate_dinov2_heatmap.py \
  --data_root "${DATA_ROOT}" \
  --model_path "${TTA_CHECKPOINT}" \
  --split "${EVAL_SPLIT}" \
  --output_dir "${EVAL_OUTPUT_DIR}" \
  --num_vis 20 \
  --nms_kernel 3 \
  --subpixel_radius 2 \
  --min_confidence 0.05 \
  --top_k 8 \
  --min_points 6 \
  --ransac_reproj_error 6.0 \
  --ransac_iterations 100 \
  --ransac_confidence 0.999 \
  --collapse_threshold "${COLLAPSE_THRESHOLD}"

echo "=================================================="
echo "[DONE]"
echo "TTA checkpoint: ${TTA_CHECKPOINT}"
echo "Eval summary: ${EVAL_OUTPUT_DIR}/${EVAL_SPLIT}_results.json"
echo "=================================================="
