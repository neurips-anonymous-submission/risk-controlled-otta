#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

export PYTHONPATH="$(pwd):$(pwd)/output:${PYTHONPATH:-}"

python risk-controlled-otta/adapt/online_tta_dino_heatmap_stable.py \
  --data_root speedplusv2 \
  --source_checkpoint output/dinov3_heatmap_source_v2/best_source_dino_heatmap.pth \
  --target_split lightbox \
  --output_dir output/dinov3_heatmap_tta_lightbox_stable_v1 \
  --adapt_modules decoder \
  --lr 2e-6 \
  --ema_alpha 0.999 \
  --memory_capacity 16 \
  --memory_sample_size 16 \
  --bank_push_conf_thresh 0.20 \
  --memory_conf_thresh 0.20 \
  --min_reliable_samples 4 \
  --mask_ratio 0.8 \
  --tau 0.7



python risk-controlled-otta/eval/evaluate_dino_heatmap.py \
  --data_root speedplusv2 \
  --model_path output/dinov3_heatmap_tta_lightbox_stable_v1/tta_final.pth \
  --split lightbox \
  --output_dir evaluation_results_dinov3_heatmap/otta_lightbox_stable_v1
