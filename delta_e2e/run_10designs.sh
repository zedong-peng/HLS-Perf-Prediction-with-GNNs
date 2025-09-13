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

# 运行训练脚本
python train_e2e.py \
--kernel_base_dir /home/user/zedongpeng/workspace/Huggingface/forgehls_kernels \
--design_base_dir /home/user/zedongpeng/workspace/Huggingface/forgehls_lite_10designs \
--output_dir ./output \
--cache_root ./graph_cache \
--target_metric dsp &

python train_e2e.py \
--kernel_base_dir /home/user/zedongpeng/workspace/Huggingface/forgehls_kernels \
--design_base_dir /home/user/zedongpeng/workspace/Huggingface/forgehls_lite_10designs \
--output_dir ./output \
--cache_root ./graph_cache \
--target_metric lut &

python train_e2e.py \
--kernel_base_dir /home/user/zedongpeng/workspace/Huggingface/forgehls_kernels \
--design_base_dir /home/user/zedongpeng/workspace/Huggingface/forgehls_lite_10designs \
--output_dir ./output \
--cache_root ./graph_cache \
--target_metric ff &

wait

echo "所有训练任务已完成！"