#!/bin/bash
set -e

# this script is used to train the GNN models on PolyBench
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
# RGCN 模型 - 分配到GPU 0,1,2
python main.py --gnn rgcn --dataset cdfg_dsp_all_numerical_PolyBench --batch_size=4 --num_workers 3 --emb_dim 128 --device 0 > logs/rgcn_dsp.log 2>&1 &
python main.py --gnn rgcn --dataset cdfg_ff_all_numerical_PolyBench --batch_size=4 --num_workers 3 --emb_dim 128 --device 1 > logs/rgcn_ff.log 2>&1 &
python main.py --gnn rgcn --dataset cdfg_lut_all_numerical_PolyBench --batch_size=4 --num_workers 3 --emb_dim 128 --device 2 > logs/rgcn_lut.log 2>&1 &

# GAT 模型 - 分配到GPU 3,4,5
python main.py --gnn gat --dataset cdfg_dsp_all_numerical_PolyBench --batch_size=4 --num_workers 3 --emb_dim 128 --device 3 > logs/gat_dsp.log 2>&1 &
python main.py --gnn gat --dataset cdfg_ff_all_numerical_PolyBench --batch_size=4 --num_workers 3 --emb_dim 128 --device 4 > logs/gat_ff.log 2>&1 &
python main.py --gnn gat --dataset cdfg_lut_all_numerical_PolyBench --batch_size=4 --num_workers 3 --emb_dim 128 --device 5 > logs/gat_lut.log 2>&1 &

# SAGE 模型 - 分配到GPU 6,7,7
python main.py --gnn sage --dataset cdfg_dsp_all_numerical_PolyBench --batch_size=4 --num_workers 3 --emb_dim 128 --device 6 > logs/sage_dsp.log 2>&1 &
python main.py --gnn sage --dataset cdfg_ff_all_numerical_PolyBench --batch_size=4 --num_workers 3 --emb_dim 128 --device 7 > logs/sage_ff.log 2>&1 &
python main.py --gnn sage --dataset cdfg_lut_all_numerical_PolyBench --batch_size=4 --num_workers 3 --emb_dim 128 --device 7 > logs/sage_lut.log 2>&1 &

# 等待所有后台任务完成
wait

echo "所有训练任务已完成！"
echo "日志文件保存在 logs/ 目录下"

