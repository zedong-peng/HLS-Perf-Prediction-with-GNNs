#! /bin/bash

# test the test_bench dataset on PolyBench.pt  models
# rgcn
python inference.py --gnn rgcn --batch_size=4 --emb_dim 128 --dataset cdfg_dsp_all_numerical_test_bench --model_path model/cdfg_dsp_all_numerical_PolyBench_rgcn_layer_5_model.pt --device 0 &
python inference.py --gnn rgcn --batch_size=4 --emb_dim 128 --dataset cdfg_ff_all_numerical_test_bench --model_path model/cdfg_ff_all_numerical_PolyBench_rgcn_layer_5_model.pt --device 1 &
python inference.py --gnn rgcn --batch_size=4 --emb_dim 128 --dataset cdfg_lut_all_numerical_test_bench --model_path model/cdfg_lut_all_numerical_PolyBench_rgcn_layer_5_model.pt --device 2 &

# sage
python inference.py --gnn sage --batch_size=4 --emb_dim 128 --dataset cdfg_dsp_all_numerical_test_bench --model_path model/cdfg_dsp_all_numerical_PolyBench_sage_layer_5_model.pt --device 3 &
python inference.py --gnn sage --batch_size=4 --emb_dim 128 --dataset cdfg_ff_all_numerical_test_bench --model_path model/cdfg_ff_all_numerical_PolyBench_sage_layer_5_model.pt --device 4 &
python inference.py --gnn sage --batch_size=4 --emb_dim 128 --dataset cdfg_lut_all_numerical_test_bench --model_path model/cdfg_lut_all_numerical_PolyBench_sage_layer_5_model.pt --device 5 &

# gat
python inference.py --gnn gat --batch_size=4 --emb_dim 128 --dataset cdfg_dsp_all_numerical_test_bench --model_path model/cdfg_dsp_all_numerical_PolyBench_gat_layer_5_model.pt --device 6 &
python inference.py --gnn gat --batch_size=4 --emb_dim 128 --dataset cdfg_ff_all_numerical_test_bench --model_path model/cdfg_ff_all_numerical_PolyBench_gat_layer_5_model.pt --device 7
python inference.py --gnn gat --batch_size=4 --emb_dim 128 --dataset cdfg_lut_all_numerical_test_bench --model_path model/cdfg_lut_all_numerical_PolyBench_gat_layer_5_model.pt --device 7



