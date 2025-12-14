#!/bin/bash
set -e
cd "$(dirname "$0")"

targets=("dsp")
trials=40
max_pairs=4000
kernel_base_dir="/home/user/zedongpeng/workspace/Huggingface/forgehls_kernels"
design_base_dir="/home/user/zedongpeng/workspace/Huggingface/forgehls_PolyBench_part_370designs"
cache_root="./graph_cache"
output_dir="./output_optuna"
code_model_path="/home/user/zedongpeng/workspace/GiT/zedong/Code-Verification/Qwen/Qwen2.5-Coder-1.5B-Instruct"

for tgt in "${targets[@]}"; do
  echo "[Optuna] target=$tgt trials=$trials"
  python train_e2e_optuna.py \
    --target_metric "$tgt" \
    --n_trials "$trials" \
    --max_pairs "$max_pairs" \
    --kernel_base_dir "$kernel_base_dir" \
    --design_base_dir "$design_base_dir" \
    --cache_root "$cache_root" \
    --output_dir "$output_dir" \
    --use_code_feature true \
    --code_model_path "$code_model_path" \
    --code_pooling last_token \
    --code_max_length 1024 \
    --code_normalize true \
    --code_cache_root "$cache_root" \
    --code_batch_size 8 \
    --region on \
    --hierarchical off \
    --loss_fn smoothl1 \
    --apply_hard_filter true \
    --normalize_targets true \
    --study_name "optuna_exp1_${tgt}" \
    --seed 42
  echo "[Optuna] target=$tgt done"
done

echo "All Optuna runs finished"
