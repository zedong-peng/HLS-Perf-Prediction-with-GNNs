#!/bin/bash
set -e

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

# pna, rgcn, gat, sage, arma, film, ggnn, pan, sgn, unet, gin-virtual, gcn-virtual, gin, gcn
python src/train.py --output_dir saves/rgcn/gnn_test/dsp --gnn rgcn --dataset cdfg_dsp_all_numerical_gnn_test --batch_size=4 --num_workers 3 --device 0 --epochs 10 &
# python src/train.py --output_dir saves/rgcn/gnn_test/dsp --gnn rgcn --dataset cdfg_dsp_all_numerical_gnn_test --batch_size=4 --num_workers 3 --device 0 --epochs 10 &
# python src/train.py --output_dir saves/gat/gnn_test/dsp --gnn gat --dataset cdfg_dsp_all_numerical_gnn_test --batch_size=4 --num_workers 3 --device 0 --epochs 10 &
# python src/train.py --output_dir saves/sage/gnn_test/dsp --gnn sage --dataset cdfg_dsp_all_numerical_gnn_test --batch_size=4 --num_workers 3 --device 0 --epochs 10 &

# wait

# python src/train.py --output_dir saves/arma/gnn_test/dsp --gnn arma --dataset cdfg_dsp_all_numerical_gnn_test --batch_size=4 --num_workers 3 --device 0 --epochs 10 &
# python src/train.py --output_dir saves/film/gnn_test/dsp --gnn film --dataset cdfg_dsp_all_numerical_gnn_test --batch_size=4 --num_workers 3 --device 0 --epochs 10 &
# python src/train.py --output_dir saves/ggnn/gnn_test/dsp --gnn ggnn --dataset cdfg_dsp_all_numerical_gnn_test --batch_size=4 --num_workers 3 --device 0 --epochs 10 &
# python src/train.py --output_dir saves/pan/gnn_test/dsp --gnn pan --dataset cdfg_dsp_all_numerical_gnn_test --batch_size=4 --num_workers 3 --device 0 --epochs 10 &

# 等待所有后台任务完成
wait

echo "所有训练任务已完成！"

