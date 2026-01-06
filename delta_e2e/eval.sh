#! /bin/bash

python eval.py \
--model_path /home/user/zedongpeng/workspace/HLS-Perf-Prediction-with-GNNs/delta_e2e/output/e2e_delta_dsp_20260106_181359/best_e2e_delta_dsp_model.pt \
--ood_design_base_dir /home/user/zedongpeng/workspace/Huggingface/forgehls_PolyBench_part_50designs \
--output_dir /home/user/zedongpeng/workspace/HLS-Perf-Prediction-with-GNNs/delta_e2e/eval_output/e2e_delta_dsp_20260106_181359/ \
--use_code_feature true \
--code_model_path /home/user/zedongpeng/workspace/HLS-Perf-Prediction-with-GNNs/delta_e2e/model/Qwen2.5-Coder-1.5B \
--code_cache_root ./graph_cache \
--code_batch_size 2
