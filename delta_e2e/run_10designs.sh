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



features=("lut" "dsp" "ff")
# features=("dsp")

# differentials=("true" "false")
differentials=("true")
# gnn=("gin" "gcn" "rgcn" "fast_rgcn")
gnn=("gin")
hierarchical=("off")
region=("on")


# 串行运行，减少总内存占用；并配置更保守的内存参数
for differential in "${differentials[@]}"; do
  for feature in "${features[@]}"; do
    for gnn_type in "${gnn[@]}"; do
      for h in "${hierarchical[@]}"; do
        for r in "${region[@]}"; do
          echo "Running: $differential | $feature | $gnn_type | $h | $r"
          python train_e2e.py \
          --kernel_base_dir /home/user/zedongpeng/workspace/Huggingface/forgehls_kernels \
          --design_base_dir /home/user/zedongpeng/workspace/Huggingface/forgehls_lite_10designs \
          --ood_design_base_dir /home/user/zedongpeng/workspace/Huggingface/forgehls_benchmark \
          --output_dir ./output \
          --cache_root ./graph_cache \
          --gnn_type $gnn_type \
          --epochs 100 \
          --batch_size 32 \
          --hidden_dim 64 \
          --num_layers 2 \
          --dropout 0.1 \
          --lr 1e-3 \
          --grad_accum_steps 1 \
          --warmup_epochs 5 \
          --target_metric $feature \
          --hierarchical $h \
          --region $r \
          --differential false \
          --kernel_baseline learned \
          --loader_workers 0 \
          --prefetch_factor 1 \
          --persistent_workers false \
          --pin_memory false
        done
      done
    done
  done
done
wait

echo "所有训练任务已完成！"
