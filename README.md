# High-Level Synthesis Performance Prediction using GNNs: Benchmarking, Modeling, and Advancing


```mermaid
flowchart TB
    subgraph KernelBranch["Kernel Branch"]
        Kenc[Node Encoder\nLinear(d→h)]
        Kgcn[GCN Layers × num_layers]
        Kpool[Global Pool\nMean Pooling]
        Kenc --> Kgcn --> Kpool
    end

    subgraph DesignBranch["Design Branch"]
        Denc[Node Encoder\nLinear(d→h)]
        Dgcn[GCN Layers × num_layers]
        Dpool[Global Pool\nMean Pooling]
        Denc --> Dgcn --> Dpool
    end

    Kpool --> Krepr[Kernel Repr.\n[h]]
    Dpool --> Drepr[Design Repr.\n[h]]
    Krepr --> Merge
    Drepr --> Merge

    subgraph PredictionHeads["Prediction Heads"]
        Bhead[Baseline Head\nLinear(h→1)]
        Dhead[Delta Head\nLinear(2h→1)]
        Fhead[Final Prediction\nbaseline + delta]
    end
    Merge --> Bhead
    Merge --> Dhead
    Bhead --> Fhead
    Dhead --> Fhead

    subgraph Training["Enhanced Training Framework"]
        MTL[Multi-task Loss\n0.3×baseline\n0.5×delta\n0.2×final]
        Grad[Gradient Clip\nmax_norm=1.0]
        LR[LR Scheduler\nReduceLROnP\npatience=15]
    end

    Fhead --> Training

```

## ForgeHLS Dataset Assets
- Latest dataset snapshot lives in `/home/user/zedongpeng/workspace/Huggingface/forgehls`, mirrored from the public Hugging Face repo.
- Detailed design dumps are packaged under `/home/user/zedongpeng/workspace/Huggingface/forgehls/designs/design_package`, grouped by benchmark suite (`CHStone.tar.gz`, `MachSuite.tar.gz`, `PolyBench.tar.gz`, `hls_algorithms.tar.gz`, `leetcode_hls_algorithms.tar.gz`, `operators.tar.gz`, `rosetta.tar.gz`, `rtl_chip.tar.gz`, `rtl_ip.tar.gz`, `rtl_module.tar.gz`, `Vitis-HLS-Introductory-Examples-flatten.tar.gz`, `Vitis_Libraries.tar.gz`).
- Each archive expands back into the original directory hierarchy; extract with `tar -xzf <archive> -C <target_dir>` before pointing `--design_base_dir` to the unpacked root.
- Helper scripts in the same folder (`analyze_kernel_designs.py`, `analyze_cpp_tokens.py`, etc.) summarise pragma counts, token stats, and kernel/design pairings for quick sanity checks.

### Citation
Please cite the ForgeHLS release when using these assets:

```bibtex
@misc{peng2025forgehls,
    title={ForgeHLS: A Large-Scale, Open-Source Dataset for High-Level Synthesis},
    author={Zedong Peng and Zeju Li and Mingzhe Gao and Qiang Xu and Chen Zhang and Jieru Zhao},
    year={2025},
    eprint={2507.03255},
    archivePrefix={arXiv},
    primaryClass={cs.AR}
}
```

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

## results

### 主要模型性能比较 (与 GraphLLM 基线对比)

on ForgeHLS_kernels:

| Model | DSP_MAPE | FF_MAPE | LUT_MAPE | DSP_RMSE | FF_RMSE | LUT_RMSE |
|-------|----------|---------|----------|----------|---------|----------|
| GraphLLM | 54.56% | 126.75% | 51.88% | 40.32 | 5722.37 | 4823.72 |
| GraphLLM+Code | 55.23% | 131.28% | 47.38% | 38.89 | 5958.19 | 5048.97 |
| Graph_Text+Code | 3446.67% | 3370.63% | 90.08% | 559.69 | 3758.39 | 438.74 |
| Graph_Text | 64.67% | 84.96% | 98.29% | 8.05 | 256.89 | 251.66 |
| Code | 322308.17% | 6074.65% | 98.03% | 164312.98 | 331208.58 | 7800.54 |
| Code-Lora | 144.09% | 86.27% | 102.12% | 31.70 | 2320.31 | 6821.32 |
| GAT | 60.25% | 99.54% | 99.71% | 20.66 | 4368.80 | 4344.83 |
| RGCN | 205.72% | 1425.85% | 67.49% | 120.84 | 4051.98 | 1877.79 |
| SAGE | 134.16% | 1218.65% | 65.03% | 104.40 | 2282.19 | 2969.52 |

On ForgeHLS_10designs:

| Model | DSP_MAPE | FF_MAPE | LUT_MAPE | DSP_RMSE | DSP_Util-RMSE | FF_RMSE | FF_Util-RMSE | LUT_RMSE | LUT_Util-RMSE |
|-------|----------|---------|----------|----------|---------------|---------|--------------|----------|---------------|
| Differential E2E | 4.3666 | 1.0135 | 1.1745 | 45.25 | 0.0050 | 35124.91 | 0.0135 | 10477.59 | 0.0080 |
| SAGE | 1.3416 | 12.1865 | 0.6503 | 104.40 | 0.0116 | 2282.19 | 0.0009 | 2969.52 | 0.0023 |



### 关键发现

**向导师汇报要点：**
1. **DSP 预测**：GraphLLM 表现最佳 (54.56% MAPE)，GAT 次之 (60.25% MAPE)，明显优于 RGCN 和 SAGE
2. **FF 预测**：GAT 表现最佳 (99.54% MAPE)，大幅优于 GraphLLM (126.75% MAPE) 和其他模型
3. **LUT 预测**：GraphLLM 表现最佳 (51.88% MAPE)，但 SAGE 和 RGCN 的表现也相对较好
4. **整体评估**：GraphLLM 在 DSP 和 LUT 上表现优异，GAT 在 FF 预测上有突出表现

**实验配置：**

**硬件环境：**
- GPU: 8x A800 80GB
- CPU: 128 cores
- 实际测试设备：NVIDIA GeForce RTX 4090

**训练参数：**
- 学习率 (Learning Rate): 0.001
- 训练轮数 (Epochs): 300
- 批大小 (Batch Size): 32
- Dropout 比率: 0.5 (GAT), 0.0 (RGCN/SAGE)
- 优化器: Adam
- 学习率调度: ReduceLROnPlateau (factor=0.8, patience=10, min_lr=1e-05)
- 网络层数: 5
- 嵌入维度: 300
- 数据加载器工作进程: 3
- 最大节点数: 5000
- 最大边数: 10000

**数据集配置：**
- 数据集名称: all_numerical_forgehls_kernels
- 特征类型: DSP, FF, LUT (分别训练三个独立模型)
- 数据集格式: cdfg_[feature]_all_numerical_forgehls_kernels
- 特征提取: full features

**训练策略：**
- 早停机制: 基于验证集损失
- 并行训练: 不同特征类型(DSP/FF/LUT)同时训练
- 模型保存: 保存验证集上最佳性能的模型
- 设备分配: 单GPU训练 (device 0)

**支持的GNN模型：**
- 当前测试: GAT, RGCN, SAGE
- 支持的其他模型: PNA, ARMA, FILM, GGNN, PAN, SGN, UNet, GIN-Virtual, GCN-Virtual, GIN, GCN

**训练脚本：**
```bash
# 运行GNN训练
bash train_on_forgehls_kernels.sh
```

**自定义配置：**
如需测试其他GNN模型或调整参数，可修改 `train_on_forgehls_kernels.sh` 中的配置：
```bash
# 修改测试的GNN模型
gnns=("gat" "rgcn" "sage" "pna" "gin")  # 添加更多模型

# 修改训练参数
--drop_ratio 0.5    # Dropout比率
--lr 0.001         # 学习率
--epochs 300       # 训练轮数
--device 0         # GPU设备ID
```
