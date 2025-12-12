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



features=("ff" "dsp" "lut")
# features=("dsp" "ff")

# exp3: delta GNN with code feature
differentials=("true")
hierarchical=("off")
region=("on")
code_model_path="/home/user/zedongpeng/workspace/GiT/zedong/Code-Verification/Qwen/Qwen2.5-Coder-1.5B-Instruct"
code_pooling="last_token"
code_max_length=1024
code_batch_size=8

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

for feature in "${features[@]}"; do
  for gnn_type in "${gnn[@]}"; do
    echo "Running exp3 (code feature): true | $feature | $gnn_type | off | on | code"
    python train_e2e.py \
    --kernel_base_dir /home/user/zedongpeng/workspace/Huggingface/forgehls_kernels \
    --design_base_dir "$design_base_dir" \
    --output_dir ./output \
    --cache_root ./graph_cache \
    --gnn_type $gnn_type \
    --epochs 400 \
    --batch_size 24 \
    --hidden_dim 128 \
    --num_layers 2 \
    --dropout 0.05 \
    --lr 1e-3 \
    --grad_accum_steps 1 \
    --warmup_epochs 5 \
    --target_metric $feature \
    --hierarchical off \
    --region on \
    --differential true \
    --kernel_baseline learned \
    --loss_type mae \
    --loader_workers 0 \
    --prefetch_factor 1 \
    --persistent_workers false \
    --pin_memory false \
    --use_code_feature true \
    --code_model_path "$code_model_path" \
    --code_pooling "$code_pooling" \
    --code_max_length $code_max_length \
    --code_normalize true \
    --code_batch_size $code_batch_size
  done
done
wait

echo "所有训练任务已完成！"
