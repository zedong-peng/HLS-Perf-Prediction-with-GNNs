#! /bin/bash

cd "$(dirname "$0")/"

# test the test_bench dataset on forgehls.pt  models
# rgcn
python src/inference.py --output_dir saves/rgcn/dsp/all_numerical_forgehls_kernels/dsp --gnn rgcn --batch_size=4 --emb_dim 128 --dataset cdfg_dsp_all_numerical_forgehls_kernels --model_path saves/rgcn/dsp/all_numerical_forgehls_kernels/default/best_model.pt --device 0 &
python src/inference.py --output_dir saves/rgcn/ff/all_numerical_forgehls_kernels/ff --gnn rgcn --batch_size=4 --emb_dim 128 --dataset cdfg_ff_all_numerical_forgehls_kernels --model_path saves/rgcn/ff/all_numerical_forgehls_kernels/default/best_model.pt --device 0 &
python src/inference.py --output_dir saves/rgcn/lut/all_numerical_forgehls_kernels/lut --gnn rgcn --batch_size=4 --emb_dim 128 --dataset cdfg_lut_all_numerical_forgehls_kernels --model_path saves/rgcn/lut/all_numerical_forgehls_kernels/default/best_model.pt --device 0 &

# sage
python src/inference.py --output_dir saves/sage/dsp/all_numerical_forgehls_kernels/dsp --gnn sage --batch_size=4 --emb_dim 128 --dataset cdfg_dsp_all_numerical_forgehls_kernels --model_path saves/sage/dsp/all_numerical_forgehls_kernels/default/best_model.pt --device 0 &
python src/inference.py --output_dir saves/sage/ff/all_numerical_forgehls_kernels/ff --gnn sage --batch_size=4 --emb_dim 128 --dataset cdfg_ff_all_numerical_forgehls_kernels --model_path saves/sage/ff/all_numerical_forgehls_kernels/default/best_model.pt --device 0 &
python src/inference.py --output_dir saves/sage/lut/all_numerical_forgehls_kernels/lut --gnn sage --batch_size=4 --emb_dim 128 --dataset cdfg_lut_all_numerical_forgehls_kernels --model_path saves/sage/lut/all_numerical_forgehls_kernels/default/best_model.pt --device 0 &

# gat
python src/inference.py --output_dir saves/gat/dsp/all_numerical_forgehls_kernels/dsp --gnn gat --batch_size=4 --emb_dim 128 --dataset cdfg_dsp_all_numerical_forgehls_kernels --model_path saves/gat/dsp/all_numerical_forgehls_kernels/default/best_model.pt --device 0 &
python src/inference.py --output_dir saves/gat/ff/all_numerical_forgehls_kernels/ff --gnn gat --batch_size=4 --emb_dim 128 --dataset cdfg_ff_all_numerical_forgehls_kernels --model_path saves/gat/ff/all_numerical_forgehls_kernels/default/best_model.pt --device 0 &
python src/inference.py --output_dir saves/gat/lut/all_numerical_forgehls_kernels/lut --gnn gat --batch_size=4 --emb_dim 128 --dataset cdfg_lut_all_numerical_forgehls_kernels --model_path saves/gat/lut/all_numerical_forgehls_kernels/default/best_model.pt --device 0