#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONPATH="$(pwd):$(pwd)/output:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-speedplusv2}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-output/dinov3_heatmap_source_v2/best_source_dino_heatmap.pth}"

PETTA_OUTPUT_DIR="${PETTA_OUTPUT_DIR:-output/dinov3_heatmap_speed_strict_petta_single_model_tta}"
PETTA_SCRIPT="${PETTA_SCRIPT:-petta_strict_single_model_tta_speed_dino_heatmap.py}"

TARGET_SPLIT="${TARGET_SPLIT:-lightbox}"
EVAL_SPLIT="${EVAL_SPLIT:-${TARGET_SPLIT}}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${PETTA_OUTPUT_DIR}/eval_after_strict_petta}"

SEED="${SEED:-42}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
NUM_VIS="${NUM_VIS:-20}"
COLLAPSE_THRESHOLD="${COLLAPSE_THRESHOLD:-0.1}"

MODEL_NAME="${MODEL_NAME:-vit_base_patch16_dinov3.lvd1689m}"
INPUT_SIZE="${INPUT_SIZE:-384}"
HEATMAP_SIZE="${HEATMAP_SIZE:-96}"
HEATMAP_SIGMA="${HEATMAP_SIGMA:-3.0}"
NUM_KEYPOINTS="${NUM_KEYPOINTS:-11}"
MID_CHANNELS="${MID_CHANNELS:-256}"
NUM_DECONV_LAYERS="${NUM_DECONV_LAYERS:-2}"

NORMAL_UPDATE_SCOPE="${NORMAL_UPDATE_SCOPE:-stem}"
GUARDED_UPDATE_SCOPE="${GUARDED_UPDATE_SCOPE:-decoder}"
LR="${LR:-5e-6}"
GUARDED_LR="${GUARDED_LR:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
GUARDED_WEIGHT_DECAY="${GUARDED_WEIGHT_DECAY:-0.0}"
ADAPT_STEPS="${ADAPT_STEPS:-1}"
GUARDED_ADAPT_STEPS="${GUARDED_ADAPT_STEPS:-1}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
GUARDED_GRAD_CLIP_NORM="${GUARDED_GRAD_CLIP_NORM:-0.5}"

TEMPERATURE="${TEMPERATURE:-1.0}"
LAMBDA_ENTROPY="${LAMBDA_ENTROPY:-1.0}"
LAMBDA_CONFIDENCE="${LAMBDA_CONFIDENCE:-0.05}"
LAMBDA_DWT="${LAMBDA_DWT:-0.1}"
LAMBDA_MIM="${LAMBDA_MIM:-0.25}"
LAMBDA_ANCHOR="${LAMBDA_ANCHOR:-0.01}"
MIM_MASK_RATIO="${MIM_MASK_RATIO:-0.35}"
MIM_PATCH_SIZE="${MIM_PATCH_SIZE:-32}"
USE_EMA_TEACHER="${USE_EMA_TEACHER:-1}"
TEACHER_MOMENTUM="${TEACHER_MOMENTUM:-0.999}"

PSEUDO_BBOX_MIN_CONFIDENCE="${PSEUDO_BBOX_MIN_CONFIDENCE:-0.05}"
PSEUDO_BBOX_EXPAND_RATIO="${PSEUDO_BBOX_EXPAND_RATIO:-1.50}"
PSEUDO_BBOX_MIN_SIZE="${PSEUDO_BBOX_MIN_SIZE:-96.0}"

PETTA_WINDOW_SIZE="${PETTA_WINDOW_SIZE:-32}"
PETTA_WARMUP_STEPS="${PETTA_WARMUP_STEPS:-8}"
PETTA_THRESHOLD="${PETTA_THRESHOLD:-0.75}"
PETTA_FREEZE_THRESHOLD="${PETTA_FREEZE_THRESHOLD:-1.25}"
PETTA_QUALITY_WEIGHT="${PETTA_QUALITY_WEIGHT:-1.00}"
PETTA_CONFIDENCE_WEIGHT="${PETTA_CONFIDENCE_WEIGHT:-0.50}"
PETTA_ENTROPY_WEIGHT="${PETTA_ENTROPY_WEIGHT:-0.25}"
PETTA_REPROJECTION_WEIGHT="${PETTA_REPROJECTION_WEIGHT:-0.50}"
PETTA_INLIER_WEIGHT="${PETTA_INLIER_WEIGHT:-0.50}"
PETTA_FALLBACK_WEIGHT="${PETTA_FALLBACK_WEIGHT:-0.50}"
PETTA_MIN_INLIER_RATIO="${PETTA_MIN_INLIER_RATIO:-0.50}"
SOFT_RESET_MOMENTUM="${SOFT_RESET_MOMENTUM:-0.25}"
RESET_TO_LAST_STABLE="${RESET_TO_LAST_STABLE:-1}"

QUALITY_REPROJECTION_CAP="${QUALITY_REPROJECTION_CAP:-50.0}"
NMS_KERNEL="${NMS_KERNEL:-3}"
SUBPIXEL_RADIUS="${SUBPIXEL_RADIUS:-2}"
MIN_CONFIDENCE="${MIN_CONFIDENCE:-0.05}"
TOP_K="${TOP_K:-8}"
MIN_POINTS="${MIN_POINTS:-6}"
RANSAC_REPROJ_ERROR="${RANSAC_REPROJ_ERROR:-6.0}"
RANSAC_ITERATIONS="${RANSAC_ITERATIONS:-100}"
RANSAC_CONFIDENCE="${RANSAC_CONFIDENCE:-0.999}"
NUM_WORKERS="${NUM_WORKERS:-4}"

PETTA_FINAL_CHECKPOINT="${PETTA_OUTPUT_DIR}/strict_petta_final.pth"
PETTA_HISTORY="${PETTA_OUTPUT_DIR}/strict_petta_history.json"

mkdir -p "${PETTA_OUTPUT_DIR}" "${EVAL_OUTPUT_DIR}"

if [[ ! -f "${PETTA_SCRIPT}" ]]; then
  echo "[ERROR] PeTTA script not found: ${PETTA_SCRIPT}" >&2
  echo "        Put petta_strict_single_model_tta_speed_dino_heatmap.py in the current directory" >&2
  echo "        or set PETTA_SCRIPT=/path/to/petta_strict_single_model_tta_speed_dino_heatmap.py" >&2
  exit 1
fi

if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
  echo "[ERROR] Source checkpoint not found: ${SOURCE_CHECKPOINT}" >&2
  exit 1
fi

MAX_ARGS=()
if [[ "${MAX_SAMPLES}" != "0" ]]; then
  MAX_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

EMA_ARGS=()
if [[ "${USE_EMA_TEACHER}" == "1" || "${USE_EMA_TEACHER}" == "true" || "${USE_EMA_TEACHER}" == "TRUE" ]]; then
  EMA_ARGS+=(--use_ema_teacher)
fi

RESET_ARGS=()
if [[ "${RESET_TO_LAST_STABLE}" == "1" || "${RESET_TO_LAST_STABLE}" == "true" || "${RESET_TO_LAST_STABLE}" == "TRUE" ]]; then
  RESET_ARGS+=(--reset_to_last_stable)
fi

echo "[1/2] Running STRICT SPEED PeTTA single-model adaptation"
echo "      Strict mode: target GT pose/keypoints/heatmap/GT-derived bbox are not used during adaptation."
echo "      Script: ${PETTA_SCRIPT}"
echo "      Output dir: ${PETTA_OUTPUT_DIR}"
echo "      Normal scope: ${NORMAL_UPDATE_SCOPE}; guarded scope: ${GUARDED_UPDATE_SCOPE}"
echo "      Risk thresholds: guard=${PETTA_THRESHOLD}, freeze=${PETTA_FREEZE_THRESHOLD}"

python "${PETTA_SCRIPT}" \
  --data_root "${DATA_ROOT}" \
  --source_checkpoint "${SOURCE_CHECKPOINT}" \
  --target_split "${TARGET_SPLIT}" \
  --seed "${SEED}" \
  --output_dir "${PETTA_OUTPUT_DIR}" \
  --model_name "${MODEL_NAME}" \
  --input_size "${INPUT_SIZE}" \
  --heatmap_size "${HEATMAP_SIZE}" \
  --heatmap_sigma "${HEATMAP_SIGMA}" \
  --num_keypoints "${NUM_KEYPOINTS}" \
  --mid_channels "${MID_CHANNELS}" \
  --num_deconv_layers "${NUM_DECONV_LAYERS}" \
  --normal_update_scope "${NORMAL_UPDATE_SCOPE}" \
  --guarded_update_scope "${GUARDED_UPDATE_SCOPE}" \
  --lr "${LR}" \
  --guarded_lr "${GUARDED_LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --guarded_weight_decay "${GUARDED_WEIGHT_DECAY}" \
  --adapt_steps "${ADAPT_STEPS}" \
  --guarded_adapt_steps "${GUARDED_ADAPT_STEPS}" \
  --temperature "${TEMPERATURE}" \
  --lambda_entropy "${LAMBDA_ENTROPY}" \
  --lambda_confidence "${LAMBDA_CONFIDENCE}" \
  --lambda_dwt "${LAMBDA_DWT}" \
  --lambda_mim "${LAMBDA_MIM}" \
  --lambda_anchor "${LAMBDA_ANCHOR}" \
  --mim_mask_ratio "${MIM_MASK_RATIO}" \
  --mim_patch_size "${MIM_PATCH_SIZE}" \
  --teacher_momentum "${TEACHER_MOMENTUM}" \
  --grad_clip_norm "${GRAD_CLIP_NORM}" \
  --guarded_grad_clip_norm "${GUARDED_GRAD_CLIP_NORM}" \
  --pseudo_bbox_min_confidence "${PSEUDO_BBOX_MIN_CONFIDENCE}" \
  --pseudo_bbox_expand_ratio "${PSEUDO_BBOX_EXPAND_RATIO}" \
  --pseudo_bbox_min_size "${PSEUDO_BBOX_MIN_SIZE}" \
  --petta_window_size "${PETTA_WINDOW_SIZE}" \
  --petta_warmup_steps "${PETTA_WARMUP_STEPS}" \
  --petta_threshold "${PETTA_THRESHOLD}" \
  --petta_freeze_threshold "${PETTA_FREEZE_THRESHOLD}" \
  --petta_quality_weight "${PETTA_QUALITY_WEIGHT}" \
  --petta_confidence_weight "${PETTA_CONFIDENCE_WEIGHT}" \
  --petta_entropy_weight "${PETTA_ENTROPY_WEIGHT}" \
  --petta_reprojection_weight "${PETTA_REPROJECTION_WEIGHT}" \
  --petta_inlier_weight "${PETTA_INLIER_WEIGHT}" \
  --petta_fallback_weight "${PETTA_FALLBACK_WEIGHT}" \
  --petta_min_inlier_ratio "${PETTA_MIN_INLIER_RATIO}" \
  --soft_reset_momentum "${SOFT_RESET_MOMENTUM}" \
  --quality_reprojection_cap "${QUALITY_REPROJECTION_CAP}" \
  --nms_kernel "${NMS_KERNEL}" \
  --subpixel_radius "${SUBPIXEL_RADIUS}" \
  --min_confidence "${MIN_CONFIDENCE}" \
  --top_k "${TOP_K}" \
  --min_points "${MIN_POINTS}" \
  --ransac_reproj_error "${RANSAC_REPROJ_ERROR}" \
  --ransac_iterations "${RANSAC_ITERATIONS}" \
  --ransac_confidence "${RANSAC_CONFIDENCE}" \
  --num_workers "${NUM_WORKERS}" \
  "${EMA_ARGS[@]}" \
  "${RESET_ARGS[@]}" \
  "${MAX_ARGS[@]}"

if [[ ! -f "${PETTA_FINAL_CHECKPOINT}" ]]; then
  echo "[ERROR] Expected PeTTA checkpoint was not created: ${PETTA_FINAL_CHECKPOINT}" >&2
  exit 1
fi

echo "[2/2] Evaluating STRICT SPEED PeTTA-adapted model"
python -m risk_controlled_otta.eval.evaluate_dino_heatmap \
  --data_root "${DATA_ROOT}" \
  --model_path "${PETTA_FINAL_CHECKPOINT}" \
  --split "${EVAL_SPLIT}" \
  --output_dir "${EVAL_OUTPUT_DIR}" \
  --input_size "${INPUT_SIZE}" \
  --heatmap_size "${HEATMAP_SIZE}" \
  --num_vis "${NUM_VIS}" \
  --nms_kernel "${NMS_KERNEL}" \
  --subpixel_radius "${SUBPIXEL_RADIUS}" \
  --min_confidence "${MIN_CONFIDENCE}" \
  --top_k "${TOP_K}" \
  --min_points "${MIN_POINTS}" \
  --ransac_reproj_error "${RANSAC_REPROJ_ERROR}" \
  --ransac_iterations "${RANSAC_ITERATIONS}" \
  --ransac_confidence "${RANSAC_CONFIDENCE}" \
  --collapse_threshold "${COLLAPSE_THRESHOLD}"

echo "[OK] Strict PeTTA adaptation and evaluation finished."
echo "History: ${PETTA_HISTORY}"
echo "Final checkpoint: ${PETTA_FINAL_CHECKPOINT}"
echo "Eval results: ${EVAL_OUTPUT_DIR}/${EVAL_SPLIT}_results.json"

