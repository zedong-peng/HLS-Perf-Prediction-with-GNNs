#!/bin/bash
set -e

cd "$(dirname "$0")/"


# 定义清理函数，用于捕获中断信号
cleanup() {
    echo "接收到中断信号，正在终止所有后台进程..."
    # 杀死所有子进程
    pkill -P $$
    exit 1
}

# 捕获中断信号（Ctrl+C）
trap cleanup SIGINT SIGTERM

# features=("dsp" "lut" "ff")
features=("dsp")
# differentials=("true" "false")
differentials=("false")
# gnn=("gin" "gcn" "rgcn" "fast_rgcn")
gnn=("gin")
# hierarchical=("on" "off")
hierarchical=("on")

# # 预热缓存（on/off 各一次，epochs=0 仅构建缓存与配对）
# for h in on off; do
#   python train_e2e.py \
#     --kernel_base_dir /home/user/zedongpeng/workspace/Huggingface/forgehls_kernels \
#     --design_base_dir /home/user/zedongpeng/workspace/Huggingface/forgehls_lite_10designs \
#     --ood_design_base_dir /home/user/zedongpeng/workspace/Huggingface/forgehls_benchmark \
#     --output_dir ./output \
#     --cache_root ./graph_cache \
#     --gnn_type gin \
#     --epochs 0 \
#     --batch_size 1 \
#     --target_metric dsp \
#     --hierarchical $h \
#     --max_workers 100 \
#     --differential false &
# done
# wait

for differential in "${differentials[@]}"; do
  for feature in "${features[@]}"; do
    for gnn_type in "${gnn[@]}"; do
      for h in "${hierarchical[@]}"; do
        echo "Running: $differential | $feature | $gnn_type | $h"
        python train_e2e.py \
        --kernel_base_dir /home/user/zedongpeng/workspace/Huggingface/forgehls_kernels \
        --design_base_dir /home/user/zedongpeng/workspace/Huggingface/forgehls_lite_10designs \
        --ood_design_base_dir /home/user/zedongpeng/workspace/Huggingface/forgehls_benchmark \
        --output_dir ./output \
        --cache_root ./graph_cache \
        --gnn_type $gnn_type \
        --epochs 300 \
        --batch_size 32 \
        --hidden_dim 64 \
        --num_layers 2 \
        --dropout 0.1 \
        --lr 1e-3 \
        --grad_accum_steps 1 \
        --warmup_epochs 5 \
        --target_metric $feature \
        --hierarchical $h \
        --differential false &
      done
    done
  done
done

# 等待所有后台任务完成
wait

echo "所有训练任务已完成！"
