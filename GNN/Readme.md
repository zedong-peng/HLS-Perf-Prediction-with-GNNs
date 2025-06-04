# Graph-Level Regression

### Datasets
* For the off-the-shelf approach:
  * DFG: dfg_*cp/dsp/ff/lut*
  * CDFG: cdfg_*cp/dsp/ff/lut*
  * CDFG+real-case benchmarks: cdfg_*cp/dsp/ff/lut*_all
* For the knowledge-rich approach:
  * DFG: dfg_*cp/dsp/ff/lut*_numerical
  * CDFG: cdfg_*cp/dsp/ff/lut*_numerical
  * CDFG+real-case benchmarks: cdfg_*cp/dsp/ff/lut*_all_numerical
* For the knowledge-infused approach:
  * DFG: dfg_*cp/dsp/ff/lut*_binary
  * CDFG: dfg_*cp/dsp/ff/lut*_binary
  * CDFG+real-case benchmarks: cdfg_*cp/dsp/ff/lut*_all_binary
* [More detailes of dataset format](https://github.com/lydiawunan/HLS-Perf-Prediction-with-GNNs/tree/main/GNN/dataset)

### GNN Models
* 14 models are included
* To switch among different approaches, the imported files should be changed accordingly in [node_encoder.py](https://github.com/lydiawunan/HLS-Perf-Prediction-with-GNNs/blob/main/GNN/node_encoder.py).
  * For example, for the knowledge-infused approach, features_numerical.py should be imported.
    ```sh
    from features_numerical import get_node_feature_dims, get_edge_feature_dims 
    ```
* How to run:
  ```sh 
  python src/check_dataset_valid.py --dataset_name cdfg_ff_all_numerical_gnn_test --max_samples 10000
  vim src/make_master_file.py
  # edit make_master_file.py. add dataset name.
  python src/make_master_file.py
  bash train.sh
  ```

  ### 
  for various training in forgehls, we do:
  1. fileter extremly large graph: nodes>5000, edges>10000
  2. when training PNA model, needs empty_cache() for each epochs.
    No empty_cache: batch_size=4,  80GB GPU memory still OOM.
    with empty_cache: batch_size=32, ~24GB GPU memory.
