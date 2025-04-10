#!/bin/bash

# for test
python main.py --gnn pna --dataset cdfg_ff_all_numerical_gnn_test --epochs 2 --batch_size=1 --lr=0.0001
# python main.py --gnn rgcn --dataset cdfg_dsp_all_numerical --epochs 10 --feature simple
# python main.py --gnn gat --dataset cdfg_dsp_all_numerical --epochs 10 --feature simple
# python main.py --gnn sage --dataset cdfg_dsp_all_numerical --epochs 10 --feature simple

# # for Polybench
# # pna
# python main.py --gnn pna --dataset cdfg_dsp_all_numerical_Polybench --batch_size=4
# python main.py --gnn pna --dataset cdfg_ff_all_numerical_Polybench --batch_size=4
# python main.py --gnn pna --dataset cdfg_lut_all_numerical_Polybench --batch_size=4


# # rgcn
# python main.py --gnn rgcn --dataset cdfg_dsp_all_numerical_Polybench --batch_size=4
# python main.py --gnn rgcn --dataset cdfg_ff_all_numerical_Polybench --batch_size=4
# python main.py --gnn rgcn --dataset cdfg_lut_all_numerical_Polybench --batch_size=4

# # gat
# python main.py --gnn gat --dataset cdfg_dsp_all_numerical_Polybench --batch_size=4
# python main.py --gnn gat --dataset cdfg_ff_all_numerical_Polybench --batch_size=4
# python main.py --gnn gat --dataset cdfg_lut_all_numerical_Polybench --batch_size=4

# # sage
# python main.py --gnn sage --dataset cdfg_dsp_all_numerical_Polybench --batch_size=4
# python main.py --gnn sage --dataset cdfg_ff_all_numerical_Polybench --batch_size=4
# python main.py --gnn sage --dataset cdfg_lut_all_numerical_Polybench --batch_size=4

