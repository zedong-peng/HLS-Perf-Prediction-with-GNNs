#!/bin/bash
set -e

# this script is used to train the GNN models on forgehls
# the factor is set for 8*A800 80GB GPU, 128 cores CPU

# 创建日志目录
mkdir -p logs

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

python src/train.py --output_dir ./saves/pna/dsp/forgehls/default --gnn pna --dataset cdfg_dsp_all_numerical_forgehls --device 0 &
python src/train.py --output_dir ./saves/pna/ff/forgehls/default --gnn pna --dataset cdfg_ff_all_numerical_forgehls --device 1  &
python src/train.py --output_dir ./saves/pna/lut/forgehls/default --gnn pna --dataset cdfg_lut_all_numerical_forgehls --device 2 &

python src/train.py --output_dir ./saves/pna/dsp/wunan/default --gnn pna --dataset cdfg_dsp_all_numerical --device 3 &
python src/train.py --output_dir ./saves/pna/ff/wunan/default --gnn pna --dataset cdfg_ff_all_numerical --device 4  &
python src/train.py --output_dir ./saves/pna/lut/wunan/default --gnn pna --dataset cdfg_lut_all_numerical --device 5 &

# 等待所有后台任务完成
wait

python src/train.py --output_dir ./saves/pna/dsp/forgehls/drop_out_0 --gnn pna --dataset cdfg_dsp_all_numerical_forgehls --device 0 --drop_ratio 0 &
python src/train.py --output_dir ./saves/pna/ff/forgehls/drop_out_0 --gnn pna --dataset cdfg_ff_all_numerical_forgehls --device 1  --drop_ratio 0 &
python src/train.py --output_dir ./saves/pna/lut/forgehls/drop_out_0 --gnn pna --dataset cdfg_lut_all_numerical_forgehls --device 2 --drop_ratio 0 &

python src/train.py --output_dir ./saves/pna/dsp/wunan/drop_out_0 --gnn pna --dataset cdfg_dsp_all_numerical --device 3 --drop_ratio 0 &
python src/train.py --output_dir ./saves/pna/ff/wunan/drop_out_0 --gnn pna --dataset cdfg_ff_all_numerical --device 4  --drop_ratio 0 &
python src/train.py --output_dir ./saves/pna/lut/wunan/drop_out_0 --gnn pna --dataset cdfg_lut_all_numerical --device 5 --drop_ratio 0 &

wait

echo "所有训练任务已完成！"

