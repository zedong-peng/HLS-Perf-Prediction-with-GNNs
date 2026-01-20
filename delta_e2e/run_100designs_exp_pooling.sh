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



features=("worst_latency")

# exp3: delta GNN with code feature
# code_model_path="/home/user/zedongpeng/workspace/GiT/zedong/Code-Verification/Qwen/Qwen2.5-Coder-1.5B-Instruct"
code_model_path="/home/user/zedongpeng/workspace/HLS-Perf-Prediction-with-GNNs/delta_e2e/model/Qwen2.5-Coder-1.5B"
code_pooling="last_token"
code_max_length=2048
code_batch_size=8

# from fast to slow
# gnn=("gcn" "fast_rgcn" "rgcn" "gin" "pna")
gnn=("graphsage")

# graph_pooling_type=("mean" "attention" "sum")
graph_pooling_type=("max")

# design_base_dir
# design_base_dir="/home/user/zedongpeng/workspace/Huggingface/forgehls_lite_100designs"
# design_base_dir="/home/user/zedongpeng/workspace/Huggingface/forgehls_PolyBench_part_250designs"
# design_base_dir="/home/user/zedongpeng/workspace/Huggingface/new_polybench_without_attribution_pragma_50designs"
#design_base_dir="/home/user/zedongpeng/workspace/Huggingface/forgehls_PolyBench_part_370designs"
design_base_dir="/home/user/zedongpeng/workspace/Huggingface/forgehls_PolyBench_part_500designs"
# design_base_dir="/home/user/zedongpeng/workspace/Huggingface/forgehls_PolyBench_part_1000designs"

# 串行运行，减少总内存占用；并配置更保守的内存参数
# 4090 当前配置下可稳定执行3个训练任务
# --ood_design_base_dir /home/user/zedongpeng/workspace/Huggingface/forgehls_benchmark \

for feature in "${features[@]}"; do
  for gnn_type in "${gnn[@]}"; do
    for graph_pooling in "${graph_pooling_type[@]}"; do
      echo "Running exp3 (code feature): true | $feature | $gnn_type | off | on | code"
      python train_e2e.py \
      --kernel_base_dir /home/user/zedongpeng/workspace/Huggingface/forgehls_kernels \
      --design_base_dir "$design_base_dir" \
      --output_dir ./output \
      --cache_root ./graph_cache \
      --gnn_type $gnn_type \
      --epochs 400 \
      --batch_size 16 \
      --hidden_dim 128 \
      --num_layers 2 \
      --dropout 0.02 \
      --lr 5e-4 \
      --target_metric $feature \
      --hierarchical off \
      --region on \
      --differential true \
      --kernel_baseline learned \
      --loss_type smoothl1 \
      --use_code_feature true \
      --code_model_path "$code_model_path" \
      --code_pooling "$code_pooling" \
      --code_max_length $code_max_length \
      --code_normalize false \
      --code_batch_size $code_batch_size \
      --graph_pooling $graph_pooling
    done
  done
done
wait

echo "所有训练任务已完成！"
