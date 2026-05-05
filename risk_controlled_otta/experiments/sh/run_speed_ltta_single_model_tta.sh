#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONPATH="$(pwd):$(pwd)/output:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-speedplusv2}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-output/dinov3_heatmap_source_v2/best_source_dino_heatmap.pth}"
LTTA_OUTPUT_DIR="${LTTA_OUTPUT_DIR:-output/dinov3_heatmap_speed_ltta_single_model_tta}"
LTTA_SCRIPT="${LTTA_SCRIPT:-ltta_single_model_tta_speed_dino_heatmap.py}"

TARGET_SPLIT="${TARGET_SPLIT:-lightbox}"
EVAL_SPLIT="${EVAL_SPLIT:-${TARGET_SPLIT}}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${LTTA_OUTPUT_DIR}/eval_after_ltta}"
SEED="${SEED:-42}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
NUM_VIS="${NUM_VIS:-20}"
COLLAPSE_THRESHOLD="${COLLAPSE_THRESHOLD:-0.1}"

mkdir -p "${LTTA_OUTPUT_DIR}"
mkdir -p "${EVAL_OUTPUT_DIR}"

if [[ ! -f "${LTTA_SCRIPT}" ]]; then
  echo "[ERROR] L-TTA script not found: ${LTTA_SCRIPT}" >&2
  echo "        Copy ltta_single_model_tta_speed_dino_heatmap.py to output/risk_controlled_otta/experiments/" >&2
  echo "        or set LTTA_SCRIPT=/path/to/ltta_single_model_tta_speed_dino_heatmap.py" >&2
  exit 1
fi

if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
  echo "[ERROR] Source checkpoint not found: ${SOURCE_CHECKPOINT}" >&2
  echo "        Set SOURCE_CHECKPOINT=/path/to/best_source_dino_heatmap.pth" >&2
  exit 1
fi

MAX_ARGS=()
if [[ "${MAX_SAMPLES}" != "0" ]]; then
  MAX_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

echo "[1/2] Running SPEED L-TTA single-model adaptation"
python "${LTTA_SCRIPT}" \
  --data_root "${DATA_ROOT}" \
  --source_checkpoint "${SOURCE_CHECKPOINT}" \
  --target_split "${TARGET_SPLIT}" \
  --seed "${SEED}" \
  --output_dir "${LTTA_OUTPUT_DIR}" \
  --model_name vit_base_patch16_dinov3.lvd1689m \
  --input_size 384 \
  --heatmap_size 96 \
  --heatmap_sigma 3.0 \
  --num_keypoints 11 \
  --mid_channels 256 \
  --num_deconv_layers 2 \
  --update_scope stem \
  --lr 5e-6 \
  --weight_decay 0.0 \
  --adapt_steps 1 \
  --temperature 1.0 \
  --lambda_entropy 1.0 \
  --lambda_confidence 0.05 \
  --lambda_dwt 0.1 \
  --grad_clip_norm 1.0 \
  --quality_reprojection_cap 50.0 \
  --nms_kernel 3 \
  --subpixel_radius 2 \
  --min_confidence 0.05 \
  --top_k 8 \
  --min_points 6 \
  --ransac_reproj_error 6.0 \
  --ransac_iterations 100 \
  --ransac_confidence 0.999 \
  --num_workers 4 \
  "${MAX_ARGS[@]}"

if [[ ! -f "${LTTA_OUTPUT_DIR}/ltta_final.pth" ]]; then
  echo "[ERROR] L-TTA final checkpoint not found: ${LTTA_OUTPUT_DIR}/ltta_final.pth" >&2
  exit 1
fi

echo "[2/2] Evaluating SPEED L-TTA-adapted model"
python -m risk_controlled_otta.eval.evaluate_dino_heatmap \
  --data_root "${DATA_ROOT}" \
  --model_path "${LTTA_OUTPUT_DIR}/ltta_final.pth" \
  --split "${EVAL_SPLIT}" \
  --output_dir "${EVAL_OUTPUT_DIR}" \
  --input_size 384 \
  --heatmap_size 96 \
  --num_vis "${NUM_VIS}" \
  --nms_kernel 3 \
  --subpixel_radius 2 \
  --min_confidence 0.05 \
  --top_k 8 \
  --min_points 6 \
  --ransac_reproj_error 6.0 \
  --ransac_iterations 100 \
  --ransac_confidence 0.999 \
  --collapse_threshold "${COLLAPSE_THRESHOLD}"

echo "[OK] L-TTA adaptation and evaluation finished."
echo "History: ${LTTA_OUTPUT_DIR}/ltta_history.json"
echo "Final checkpoint: ${LTTA_OUTPUT_DIR}/ltta_final.pth"
echo "Eval summary: ${EVAL_OUTPUT_DIR}/${EVAL_SPLIT}_results.json"


