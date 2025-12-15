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



features=("dsp" "ff" "lut")

# exp2: delta GNN
differentials=("true")
hierarchical=("off")
region=("on")
use_code_feature=("False")

# from fast to slow
# gnn=("gcn" "fast_rgcn" "rgcn" "gin" "pna")
gnn=("pna")


# design_base_dir
# design_base_dir="/home/user/zedongpeng/workspace/Huggingface/forgehls_lite_100designs"
# design_base_dir="/home/user/zedongpeng/workspace/Huggingface/forgehls_PolyBench_part_250designs"
# design_base_dir="/home/user/zedongpeng/workspace/Huggingface/new_polybench_without_attribution_pragma_50designs"
design_base_dir="/home/user/zedongpeng/workspace/Huggingface/forgehls_PolyBench_part_500designs"

# 串行运行，减少总内存占用；并配置更保守的内存参数
# 4090 当前配置下可稳定执行3个训练任务
# --ood_design_base_dir /home/user/zedongpeng/workspace/Huggingface/forgehls_benchmark \

for differential in "${differentials[@]}"; do
  for feature in "${features[@]}"; do
    for gnn_type in "${gnn[@]}"; do
      for h in "${hierarchical[@]}"; do
        for r in "${region[@]}"; do
          echo "Running exp2 (no code): $differential | $feature | $gnn_type | $h | $r"
          python train_e2e.py \
          --kernel_base_dir /home/user/zedongpeng/workspace/Huggingface/forgehls_kernels \
          --design_base_dir "$design_base_dir" \
          --output_dir ./output \
          --cache_root ./graph_cache \
          --gnn_type $gnn_type \
          --epochs 300 \
          --batch_size 16 \
          --hidden_dim 128 \
          --num_layers 2 \
          --dropout 0.02 \
          --lr 5e-4 \
          --target_metric $feature \
          --hierarchical $h \
          --region $r \
          --differential $differential \
          --kernel_baseline learned \
          --loss_type smoothl1
        done
      done
    done
  done
done

wait

echo "所有训练任务已完成！"
