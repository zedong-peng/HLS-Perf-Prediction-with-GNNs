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


# 预热缓存（on/off 组合 各一次，epochs=0 仅构建缓存与配对）
# 注意：这里没有 target_metric，因为构建图时不需要考虑目标指标

for h in on off; do
  for r in on off; do
    python train_e2e.py \
      --kernel_base_dir /home/user/zedongpeng/workspace/Huggingface/forgehls_kernels \
      --design_base_dir /home/user/zedongpeng/workspace/Huggingface/forgehls_lite_100designs \
      --ood_design_base_dir /home/user/zedongpeng/workspace/Huggingface/forgehls_benchmark \
      --output_dir ./output \
      --cache_root ./graph_cache \
      --gnn_type gin \
      --epochs 0 \
      --batch_size 1 \
      --target_metric dsp \
      --hierarchical $h \
      --region $r \
      --max_workers 100 \
      --differential false &
  done
done
wait

# 等待所有后台任务完成
wait

echo "所有训练任务已完成！"
