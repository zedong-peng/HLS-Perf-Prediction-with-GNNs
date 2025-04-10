#!/bin/bash
set -e

# # for test
# python main.py --gnn sage --dataset cdfg_ff_all_numerical_gnn_test --epochs 20 --feature simple
# python main.py --gnn rgcn --dataset cdfg_ff_all_numerical_gnn_test --epochs 2 --feature simple
# python main.py --gnn gat --dataset cdfg_ff_all_numerical_gnn_test --epochs 2 --feature simple
# python main.py --gnn sage --dataset cdfg_ff_all_numerical_gnn_test --epochs 2 --feature simple

# for PolyBench --batch_size 4 --emb_dim 128 --num_workers 4 --device 0-7
# # pna
# python main.py --gnn pna --dataset cdfg_dsp_all_numerical_PolyBench --batch_size=4 --emb_dim 128 --num_workers 4 --device 0
# python main.py --gnn pna --dataset cdfg_ff_all_numerical_PolyBench --batch_size=4 --emb_dim 128 --num_workers 4 --device 0
# python main.py --gnn pna --dataset cdfg_lut_all_numerical_PolyBench --batch_size=4 --emb_dim 128 --num_workers 4 --device 0

# 创建日志目录
mkdir -p logs

# RGCN 模型 - 分配到GPU 0,1,2
python main.py --gnn rgcn --dataset cdfg_dsp_all_numerical_PolyBench --batch_size=4 --num_workers 4 --emb_dim 128 --device 0 > logs/rgcn_dsp.log 2>&1 &
python main.py --gnn rgcn --dataset cdfg_ff_all_numerical_PolyBench --batch_size=4 --num_workers 4 --emb_dim 128 --device 1 > logs/rgcn_ff.log 2>&1 &
python main.py --gnn rgcn --dataset cdfg_lut_all_numerical_PolyBench --batch_size=4 --num_workers 4 --emb_dim 128 --device 2 > logs/rgcn_lut.log 2>&1 &

# GAT 模型 - 分配到GPU 3,4,5
python main.py --gnn gat --dataset cdfg_dsp_all_numerical_PolyBench --batch_size=4 --num_workers 4 --emb_dim 128 --device 3 > logs/gat_dsp.log 2>&1 &
python main.py --gnn gat --dataset cdfg_ff_all_numerical_PolyBench --batch_size=4 --num_workers 4 --emb_dim 128 --device 4 > logs/gat_ff.log 2>&1 &
python main.py --gnn gat --dataset cdfg_lut_all_numerical_PolyBench --batch_size=4 --num_workers 4 --emb_dim 128 --device 5 > logs/gat_lut.log 2>&1 &

# SAGE 模型 - 分配到GPU 6,7,7
python main.py --gnn sage --dataset cdfg_dsp_all_numerical_PolyBench --batch_size=4 --num_workers 4 --emb_dim 128 --device 6 > logs/sage_dsp.log 2>&1 &
python main.py --gnn sage --dataset cdfg_ff_all_numerical_PolyBench --batch_size=4 --num_workers 4 --emb_dim 128 --device 7 > logs/sage_ff.log 2>&1 &
python main.py --gnn sage --dataset cdfg_lut_all_numerical_PolyBench --batch_size=4 --num_workers 4 --emb_dim 128 --device 7 > logs/sage_lut.log 2>&1 &

# 等待所有后台任务完成
wait

echo "所有训练任务已完成！"
echo "日志文件保存在 logs/ 目录下"

