#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

export PYTHONPATH="$(pwd):$(pwd)/output:${PYTHONPATH:-}"

DATA_ROOT="speedplusv2"
SOURCE_CKPT="output/dinov3_heatmap_source_v2/best_source_dino_heatmap.pth"
OTTA_OUT="output/dinov3_heatmap_tta_lightbox_v4"
EVAL_SOURCE_OUT="evaluation_results_dinov3_heatmap/source_lightbox"
EVAL_OTTA_OUT="evaluation_results_dinov3_heatmap/otta_lightbox_v4"



echo "========== 2. Run OTTA on lightbox =========="
python risk-controlled-otta/adapt/online_tta_dino_heatmap.py \
  --data_root "${DATA_ROOT}" \
  --source_checkpoint "${SOURCE_CKPT}" \
  --target_split lightbox \
  --output_dir "${OTTA_OUT}"

echo "========== 3. Evaluate OTTA model on lightbox =========="
python risk-controlled-otta/eval/evaluate_dino_heatmap.py \
  --data_root "${DATA_ROOT}" \
  --model_path "${OTTA_OUT}/tta_final.pth" \
  --split lightbox \
  --output_dir "${EVAL_OTTA_OUT}"

echo "========== Done =========="
