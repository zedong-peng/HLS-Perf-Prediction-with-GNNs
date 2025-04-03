#!/bin/bash

# test
python main.py --gnn pna --dataset cdfg_cp_all_numerical --epochs 2 --feature simple

# pna
python main.py --gnn pna --dataset cdfg_cp_all_numerical_PolyBench
python main.py --gnn pna --dataset cdfg_dsp_all_numerical_PolyBench
python main.py --gnn pna --dataset cdfg_ff_all_numerical_PolyBench
python main.py --gnn pna --dataset cdfg_lut_all_numerical_PolyBench


# rgcn
python main.py --gnn rgcn --dataset cdfg_cp_all_numerical_PolyBench
python main.py --gnn rgcn --dataset cdfg_dsp_all_numerical_PolyBench
python main.py --gnn rgcn --dataset cdfg_ff_all_numerical_PolyBench
python main.py --gnn rgcn --dataset cdfg_lut_all_numerical_PolyBench

# gat
python main.py --gnn gat --dataset cdfg_cp_all_numerical_PolyBench
python main.py --gnn gat --dataset cdfg_dsp_all_numerical_PolyBench
python main.py --gnn gat --dataset cdfg_ff_all_numerical_PolyBench
python main.py --gnn gat --dataset cdfg_lut_all_numerical_PolyBench

# sage
python main.py --gnn sage --dataset cdfg_cp_all_numerical_PolyBench
python main.py --gnn sage --dataset cdfg_dsp_all_numerical_PolyBench
python main.py --gnn sage --dataset cdfg_ff_all_numerical_PolyBench
python main.py --gnn sage --dataset cdfg_lut_all_numerical_PolyBench
