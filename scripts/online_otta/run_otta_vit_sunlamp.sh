#!/bin/bash

# 指定只使用 1 号显卡（GPU 索引从 0 开始，1 表示第二张显卡）
export CUDA_VISIBLE_DEVICES=1
unset LD_LIBRARY_PATH

set -e

# 更新了输出文件夹的命名，以区分这是 sunlamp 域的对齐版本实验结果
OUT_DIR=./output/tta_vitpose_aligned_sunlamp
EVAL_DIR=./output/eval_tta_vitpose_aligned_sunlamp

echo "======================================"
echo "1. 启动 ViTPose 对齐版 OTTA 训练 (Sunlamp域)..."
echo "======================================"

python online_tta_vitpose_aligned.py \
  --data_root ../TROTT/speedplusv2 \
  --source_checkpoint output/source_paper_aligned/best_source.pth \
  --target_split sunlamp \
  --output_dir ${OUT_DIR} \
  --lr 1e-6 \
  --ema_alpha 0.9999 \
  --memory_sample_size 4 \
  --target_batch_size 1 \
  --enable_target_insertion \
  --bank_warmup_steps 300
  
echo "======================================"
echo "2. OTTA 训练完成，开始执行对齐版综合评估 (Sunlamp域)..."
echo "======================================"

python evaluate_vitpose_aligned.py \
  --model_path ${OUT_DIR}/tta_final.pth \
  --split sunlamp \
  --data_root ../TROTT/speedplusv2 \
  --output_dir ${EVAL_DIR} \
  --num_vis 20 \
  --collapse_threshold 0.1

echo "======================================"
echo "全部任务执行完毕！"
echo "训练权重及训练日志: ${OUT_DIR}/tta_final.pth 及 otta_history_aligned.json"
echo "评估结果及可视化图: ${EVAL_DIR}"
echo "======================================"
