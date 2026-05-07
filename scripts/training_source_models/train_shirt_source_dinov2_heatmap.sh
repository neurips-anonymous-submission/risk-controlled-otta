#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$(pwd):$(pwd)/output:${PYTHONPATH:-}"

DATA_ROOT="SHIRT_Dataset"
OUTPUT_DIR="output/dinov2_heatmap_shirt_source_v1"
PRETRAINED_PATH="pretrained/dinov2_base"

python risk-controlled-otta/train/train_shirt_dinov2_heatmap.py \
  --data_root "${DATA_ROOT}" \
  --roe all \
  --output_dir "${OUTPUT_DIR}" \
  --pretrained_path "${PRETRAINED_PATH}" \
  --input_size 384 \
  --heatmap_size 96 \
  --heatmap_sigma 3.0 \
  --num_keypoints 11 \
  --mid_channels 256 \
  --num_deconv_layers 2 \
  --batch_size 16 \
  --epochs 30 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --val_ratio 0.1
