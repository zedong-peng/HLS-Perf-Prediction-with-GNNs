# High-Level Synthesis Performance Prediction using GNNs: Benchmarking, Modeling, and Advancing

## Training on PolyBench
generate dataset:
```
cd Graphs
python process_real_case_graph_PolyBench.py
mv dataset ../GNN/dataset/
```

generate model:
```
cd GNN
python make_master_file.py
bash train_on_PolyBench.sh
```

test model on test_bench:
```
cd GNN
bash inference_on_PolyBench.sh
```

<!-- Prerequisites -->
## Prerequisites
* Program generation: if no new synthetic program is desired, there is **no need** to install [ldrgrn](https://github.com/gergo-/ldrgen).
* HLS and FPGA implementation: if no new data instance is desired, there is **no need** to install [Vivado Design Suite](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vivado-design-tools/2022-1.html).
* [Pytorch Geometric](https://github.com/pyg-team/pytorch_geometric) for graph-level regression tasks
* [OGB](https://github.com/snap-stanford/ogb/tree/e84a2ab93172433c58740d4f7727997727bbb52e) for node-level classification tasks

```
pip install torch-geometric torch-scatter scikit-learn
```