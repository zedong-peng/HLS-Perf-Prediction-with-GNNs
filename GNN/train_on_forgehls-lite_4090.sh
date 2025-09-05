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

# GPU:8*A800 80GB CPU:128 cores
#!/bin/bash

# gnns=("pna" "rgcn" "sage" "gat" "arma" "film" "ggnn" "pan" "sgn" "unet" "gin-virtual" "gcn-virtual" "gin" "gcn")
# gnns=("rgcn" "sage")
gnns=("sage")

datasets=("all_numerical_forgehls_10designs")
features=("dsp" "lut" "ff")

for gnn in "${gnns[@]}"; do
  for dataset in "${datasets[@]}"; do
    for feature in "${features[@]}"; do
      echo "Running: $gnn | $dataset | $feature"
      python src/train.py \
        --output_dir "saves/${gnn}/${feature}/${dataset}/default/" \
        --gnn "${gnn}" \
        --dataset "cdfg_${feature}_${dataset}" \
        --num_workers 3 \
        # --emb_dim 128 \
        --device 0
    done
  done
done
# 等待所有后台任务完成
wait

echo "所有训练任务已完成！"