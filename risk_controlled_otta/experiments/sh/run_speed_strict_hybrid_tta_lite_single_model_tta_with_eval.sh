#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Strict Hybrid-TTA-lite adaptation + evaluation for SPEED DINO heatmap pose.
#
# Strict mode:
#   Adaptation uses raw target images and predicted pseudo bboxes only.
#   It does NOT use target GT pose, GT keypoints, GT heatmaps, or GT-derived crops.
#
# Expected outputs:
#   ${HYBRID_TTA_OUTPUT_DIR}/strict_hybrid_tta_lite_history.json
#   ${HYBRID_TTA_OUTPUT_DIR}/strict_hybrid_tta_lite_final.pth
# -----------------------------------------------------------------------------

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONPATH="$(pwd):$(pwd)/output:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-speedplusv2}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-output/dinov3_heatmap_source_v2/best_source_dino_heatmap.pth}"

HYBRID_TTA_OUTPUT_DIR="${HYBRID_TTA_OUTPUT_DIR:-output/dinov3_heatmap_speed_strict_hybrid_tta_lite_single_model_tta}"
HYBRID_TTA_SCRIPT="${HYBRID_TTA_SCRIPT:-hybrid_tta_lite_single_model_tta_speed_dino_heatmap.py}"

TARGET_SPLIT="${TARGET_SPLIT:-lightbox}"
EVAL_SPLIT="${EVAL_SPLIT:-${TARGET_SPLIT}}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${HYBRID_TTA_OUTPUT_DIR}/eval_after_strict_hybrid_tta_lite}"

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

# Hybrid-TTA-lite tuning branches.
EFFICIENT_UPDATE_SCOPE="${EFFICIENT_UPDATE_SCOPE:-stem}"
FULL_UPDATE_SCOPE="${FULL_UPDATE_SCOPE:-stem_decoder}"

LR="${LR:-5e-6}"
FULL_LR="${FULL_LR:-2e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
FULL_WEIGHT_DECAY="${FULL_WEIGHT_DECAY:-0.0}"
ADAPT_STEPS="${ADAPT_STEPS:-1}"

GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
FULL_GRAD_CLIP_NORM="${FULL_GRAD_CLIP_NORM:-0.5}"

# Hybrid-TTA-lite losses.
TEMPERATURE="${TEMPERATURE:-1.0}"
LAMBDA_ENTROPY="${LAMBDA_ENTROPY:-1.0}"
LAMBDA_CONFIDENCE="${LAMBDA_CONFIDENCE:-0.05}"
LAMBDA_DWT="${LAMBDA_DWT:-0.1}"
LAMBDA_MIM="${LAMBDA_MIM:-0.25}"
MIM_MASK_RATIO="${MIM_MASK_RATIO:-0.35}"
MIM_PATCH_SIZE="${MIM_PATCH_SIZE:-32}"

USE_EMA_TEACHER="${USE_EMA_TEACHER:-1}"
TEACHER_MOMENTUM="${TEACHER_MOMENTUM:-0.999}"

# Pseudo bbox.
PSEUDO_BBOX_MIN_CONFIDENCE="${PSEUDO_BBOX_MIN_CONFIDENCE:-0.05}"
PSEUDO_BBOX_EXPAND_RATIO="${PSEUDO_BBOX_EXPAND_RATIO:-1.50}"
PSEUDO_BBOX_MIN_SIZE="${PSEUDO_BBOX_MIN_SIZE:-96.0}"

# Dynamic Domain Shift Detection.
DDSD_WINDOW_SIZE="${DDSD_WINDOW_SIZE:-32}"
DDSD_WARMUP_STEPS="${DDSD_WARMUP_STEPS:-8}"
DDSD_THRESHOLD="${DDSD_THRESHOLD:-0.75}"
DDSD_COOLDOWN_STEPS="${DDSD_COOLDOWN_STEPS:-0}"

DDSD_QUALITY_WEIGHT="${DDSD_QUALITY_WEIGHT:-1.00}"
DDSD_CONFIDENCE_WEIGHT="${DDSD_CONFIDENCE_WEIGHT:-0.50}"
DDSD_ENTROPY_WEIGHT="${DDSD_ENTROPY_WEIGHT:-0.25}"
DDSD_REPROJECTION_WEIGHT="${DDSD_REPROJECTION_WEIGHT:-0.50}"
DDSD_INLIER_WEIGHT="${DDSD_INLIER_WEIGHT:-0.50}"
DDSD_MIN_INLIER_RATIO="${DDSD_MIN_INLIER_RATIO:-0.50}"

# Pose / heatmap decoding.
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

HYBRID_TTA_HISTORY="${HYBRID_TTA_OUTPUT_DIR}/strict_hybrid_tta_lite_history.json"
HYBRID_TTA_FINAL_CHECKPOINT="${HYBRID_TTA_OUTPUT_DIR}/strict_hybrid_tta_lite_final.pth"

mkdir -p "${HYBRID_TTA_OUTPUT_DIR}" "${EVAL_OUTPUT_DIR}"

if [[ ! -f "${HYBRID_TTA_SCRIPT}" ]]; then
  echo "[ERROR] Hybrid-TTA-lite script not found: ${HYBRID_TTA_SCRIPT}" >&2
  echo "        Put hybrid_tta_lite_single_model_tta_speed_dino_heatmap.py in the current directory" >&2
  echo "        or set HYBRID_TTA_SCRIPT=/path/to/hybrid_tta_lite_single_model_tta_speed_dino_heatmap.py" >&2
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

EMA_ARGS=()
if [[ "${USE_EMA_TEACHER}" == "1" || "${USE_EMA_TEACHER}" == "true" || "${USE_EMA_TEACHER}" == "TRUE" ]]; then
  EMA_ARGS+=(--use_ema_teacher)
fi

echo "[1/2] Running STRICT SPEED Hybrid-TTA-lite single-model adaptation"
echo "      Strict mode: target GT pose/keypoints/heatmap/GT-derived bbox are not used during adaptation."
echo "      Script: ${HYBRID_TTA_SCRIPT}"
echo "      Output dir: ${HYBRID_TTA_OUTPUT_DIR}"
echo "      Efficient branch: ${EFFICIENT_UPDATE_SCOPE}"
echo "      Full branch: ${FULL_UPDATE_SCOPE}"
echo "      DDSD threshold: ${DDSD_THRESHOLD}"

python "${HYBRID_TTA_SCRIPT}" \
  --data_root "${DATA_ROOT}" \
  --source_checkpoint "${SOURCE_CHECKPOINT}" \
  --target_split "${TARGET_SPLIT}" \
  --seed "${SEED}" \
  --output_dir "${HYBRID_TTA_OUTPUT_DIR}" \
  --model_name "${MODEL_NAME}" \
  --input_size "${INPUT_SIZE}" \
  --heatmap_size "${HEATMAP_SIZE}" \
  --heatmap_sigma "${HEATMAP_SIGMA}" \
  --num_keypoints "${NUM_KEYPOINTS}" \
  --mid_channels "${MID_CHANNELS}" \
  --num_deconv_layers "${NUM_DECONV_LAYERS}" \
  --efficient_update_scope "${EFFICIENT_UPDATE_SCOPE}" \
  --full_update_scope "${FULL_UPDATE_SCOPE}" \
  --lr "${LR}" \
  --full_lr "${FULL_LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --full_weight_decay "${FULL_WEIGHT_DECAY}" \
  --adapt_steps "${ADAPT_STEPS}" \
  --temperature "${TEMPERATURE}" \
  --lambda_entropy "${LAMBDA_ENTROPY}" \
  --lambda_confidence "${LAMBDA_CONFIDENCE}" \
  --lambda_dwt "${LAMBDA_DWT}" \
  --lambda_mim "${LAMBDA_MIM}" \
  --mim_mask_ratio "${MIM_MASK_RATIO}" \
  --mim_patch_size "${MIM_PATCH_SIZE}" \
  --teacher_momentum "${TEACHER_MOMENTUM}" \
  --grad_clip_norm "${GRAD_CLIP_NORM}" \
  --full_grad_clip_norm "${FULL_GRAD_CLIP_NORM}" \
  --pseudo_bbox_min_confidence "${PSEUDO_BBOX_MIN_CONFIDENCE}" \
  --pseudo_bbox_expand_ratio "${PSEUDO_BBOX_EXPAND_RATIO}" \
  --pseudo_bbox_min_size "${PSEUDO_BBOX_MIN_SIZE}" \
  --ddsd_window_size "${DDSD_WINDOW_SIZE}" \
  --ddsd_warmup_steps "${DDSD_WARMUP_STEPS}" \
  --ddsd_threshold "${DDSD_THRESHOLD}" \
  --ddsd_cooldown_steps "${DDSD_COOLDOWN_STEPS}" \
  --ddsd_quality_weight "${DDSD_QUALITY_WEIGHT}" \
  --ddsd_confidence_weight "${DDSD_CONFIDENCE_WEIGHT}" \
  --ddsd_entropy_weight "${DDSD_ENTROPY_WEIGHT}" \
  --ddsd_reprojection_weight "${DDSD_REPROJECTION_WEIGHT}" \
  --ddsd_inlier_weight "${DDSD_INLIER_WEIGHT}" \
  --ddsd_min_inlier_ratio "${DDSD_MIN_INLIER_RATIO}" \
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
  "${MAX_ARGS[@]}"

if [[ ! -f "${HYBRID_TTA_FINAL_CHECKPOINT}" ]]; then
  echo "[ERROR] Expected Hybrid-TTA-lite checkpoint was not created: ${HYBRID_TTA_FINAL_CHECKPOINT}" >&2
  exit 1
fi

if [[ ! -f "${HYBRID_TTA_HISTORY}" ]]; then
  echo "[WARN] Expected Hybrid-TTA-lite history file was not found: ${HYBRID_TTA_HISTORY}" >&2
fi

echo "[2/2] Evaluating STRICT SPEED Hybrid-TTA-lite-adapted model"
echo "      Model path: ${HYBRID_TTA_FINAL_CHECKPOINT}"
echo "      Eval split: ${EVAL_SPLIT}"
echo "      Eval output dir: ${EVAL_OUTPUT_DIR}"

python -m risk_controlled_otta.eval.evaluate_dino_heatmap \
  --data_root "${DATA_ROOT}" \
  --model_path "${HYBRID_TTA_FINAL_CHECKPOINT}" \
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

echo "[OK] Strict Hybrid-TTA-lite adaptation and evaluation finished."
echo "History: ${HYBRID_TTA_HISTORY}"
echo "Final checkpoint: ${HYBRID_TTA_FINAL_CHECKPOINT}"
echo "Eval results: ${EVAL_OUTPUT_DIR}/${EVAL_SPLIT}_results.json"

