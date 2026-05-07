#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$(pwd):$(pwd)/output:${PYTHONPATH:-}"

DATA_ROOT="speedplusv2"
OUTPUT_DIR="output/dinov2_heatmap_speedplusv2_source"
PRETRAINED_PATH="pretrained/dinov2_base"

mkdir -p "${OUTPUT_DIR}"

python risk-controlled-otta/train/train_dinov2_heatmap.py \
  --data_root "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --pretrained_path "${PRETRAINED_PATH}" \
  --input_size 384 \
  --heatmap_size 96 \
  --heatmap_sigma 3.0 \
  --mid_channels 256 \
  --num_deconv_layers 2 \
  --batch_size 16 \
  --epochs 30 \
  --encoder_lr 5e-5 \
  --decoder_lr 5e-4 \
  --weight_decay 0.1 \
  --warmup_steps 1000 \
  --freeze_encoder_epochs 0 \
  --positive_weight 4.0 \
  --positive_threshold 0.01 \
  --num_workers 4
