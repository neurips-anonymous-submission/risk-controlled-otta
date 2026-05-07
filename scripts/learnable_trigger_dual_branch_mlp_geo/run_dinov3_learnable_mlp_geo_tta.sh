#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$(pwd):$(pwd)/output:${PYTHONPATH:-}"

# ==========================================
# 1. 路径与基础配置
# ==========================================
DATA_ROOT="speedplusv2"
SOURCE_CHECKPOINT="output/dinov3_heatmap_source_v2/best_source_dino_heatmap.pth"
TTA_OUTPUT_DIR="output/dinov3_heatmap_speed_mlp_geo_tta_sunlamp"
EVAL_OUTPUT_DIR="${TTA_OUTPUT_DIR}/eval_results"

mkdir -p "${TTA_OUTPUT_DIR}"
mkdir -p "${EVAL_OUTPUT_DIR}"

# ==========================================
# 2. 运行 MLP-Geo Learnable TTA
# ==========================================
echo ">>> [1/2] 开始运行 DINOv3 MLP-Geo Learnable TTA..."
python risk-controlled-otta/adapt/learnable_trigger_single_model_tta_dino_heatmap.py \
  --data_root "${DATA_ROOT}" \
  --source_checkpoint "${SOURCE_CHECKPOINT}" \
  --target_split sunlamp \
  --output_dir "${TTA_OUTPUT_DIR}" \
  --model_name vit_base_patch16_dinov3.lvd1689m \
  --update_scope decoder \
  --trigger_mode mlp_geo \
  --gate_usage soft_loss \
  --gate_threshold 0.5 \
  --min_soft_gate_weight 0.1 \
  --gate_hidden_dim 16 \
  --gate_lr 1e-4 \
  --gate_warmup_steps 10 \
  --feature_reprojection_cap 50.0 \
  --feature_tvec_norm_cap 20.0 \
  --lr 5e-6 \
  --adapt_steps 1 \
  --memory_capacity 64 \
  --memory_sample_size 8 \
  --min_memory_for_update 6 \
  --memory_min_quality 0.1 \
  --lambda_self_training 1.0 \
  --lambda_geo 0.1 \
  --lambda_reg 0.05 \
  --trigger_confidence 0.20 \
  --trigger_min_inliers 6 \
  --trigger_reprojection_error 6.5 \
  --num_workers 4

# ==========================================
# 3. 运行评估
# ==========================================
echo ">>> [2/2] 评估 MLP-Geo TTA 适应后的模型..."
python risk-controlled-otta/eval/evaluate_dino_heatmap.py \
  --data_root "${DATA_ROOT}" \
  --model_path "${TTA_OUTPUT_DIR}/tta_final.pth" \
  --split sunlamp \
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
  --collapse_threshold 0.1

echo ">>> [完成] MLP-Geo TTA 跨域评估结束！"
