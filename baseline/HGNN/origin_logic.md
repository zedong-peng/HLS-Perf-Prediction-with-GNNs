# H-GNN 原始逻辑

## 数据集构建
- `GNNp/data/datasets/run.py` 负责整体流程：遍历 kernel，读取 `../raw/source_label.csv` 中的 pragma/latency 标签，把对应 C 循环插入指定 pragma 后，通过 Programl 输出更新后的 LLVM IR 图。
- `GNNp/data/datasets/graph.py` 对每个 NetworkX 图补充手工特征（资源、延迟、调用次数、入/出度等），并根据数组分区配置挂上虚拟内存节点与读写边，使带宽约束在图里显式化。
- 处理完成的图被序列化成 PyTorch Geometric `Data` 对象，包含节点特征、`edge_index` 以及 `latency/dsp/lut/ff/IL` 等 QoR 标签，统一写入 `GNNp/data/processed/pt/`。
- `GNNp/src/data.py` 中的 `HLSGraphDataset` 直接 glob 这些 `.pt` 文件并逐条 `torch.load`，无需额外 `process()`，训练阶段按索引即可取样。

## 训练流程
- `GNNp/src/main.py` 只是实例化 `HLSGraphDataset` 并调用 `train_main`，作为命令行入口。
- `GNNp/src/train.py` 用固定种子把数据集按 80/10/10 划分，构建 PyG `DataLoader`，并根据 `FLAGS.gnn_type` 配置 SAGE/GCN/GAT、带度数直方图的 PNA，或 Transformer 风格图层。
- 训练环节用 SmoothL1Loss 优化 `FLAGS.target` 指定的指标，每个 epoch 记录 train/val/test 的 MAPE，并在验证集最优时借助 `saver` 工具保存权重。
- `GNNp/src/model.py` 定义多层消息传递网络，结合全局加和池化与最大池化，再接一个 MLP 头回归单个 QoR 数值。

## 配置入口
- `GNNp/src/config.py` 集中声明命令行参数：数据集/基准名称、预测目标、GNN 类型、层数、宽度 `D`、训练轮数、batch size、设备、归一化策略及 `no_graph`/`only_pragma` 等开关，模块间通过全局 `FLAGS` 共享。
- `saver` 与 `utils`（本文未展开）负责记录元数据、建立实验目录，并提供 `MLP`、`MyGlobalAttention` 等通用组件。

## 层次化变体与基准脚本
- 仓库包含 `GNNp`、`GNNnp`、`GNNg` 三个同构子项目，分别对应不同层级的程序图，以落地论文中的层次化预测方法；每个子项目都拥有与 `GNNp` 同步的 dataset builder 与训练入口。循环的“内层/外层”在代码里通过 `loop_blocks`/`flatten_loops`（`GNNg/data/datasets/graph.py:16-75`）标注：每个 kernel 的嵌套循环被划分为两个分组（比如 `[['lp1','lp2'], ['lp3','lp4']]`），其中组内的 loop 视为“内层层次”（即第一阶段模型的目标），剩余逻辑归入“外层层次”（第二阶段）。
- `GNNp` 处理带 pipeline/pragma 的内层循环（`p = pipelined pragma`），`GNNnp` 则覆盖未 pipeline 的内层循环（`np = non-pipeline`）。两者都在 `create_graph()` 中抽取 loop 子图，训练完毕后需对全部 `(kernel, config)` 推理，生成资源与延迟 CSV。
- `GNNg` 依赖 `GNNp/GNNnp` 的离线预测：`GNNg/data/datasets/run.py:32-70` 在构图前调用 `get_subloop_config()` 读取 `GNNp_pred_resource.csv`、`GNNnp_pred_resource.csv`、`*_latency.csv`（`graph.py:642-685`）。随后 `get_supernode_feature()`（`graph.py:120-154`）会按 loop 分组挑出匹配的 array/loop 配置，把来自 GNNp/GNNnp 的 LUT/FF/DSP/latency 输送给 supernode，并在 `add_supernode_to_c()`（`graph.py:332-420`）里把原内层循环替换为 `pipeline_callX/no_pipeline_callX` 节点。`add_resource_feature()`、`add_latency_feature()`（`graph.py:156-232`）再把这些数值写入 supernode 节点特征，完成“p/np → g”的衔接。
- `benchmark/run_script.py` 自动化采集真实 QoR：批量触发 Vitis HLS Tcl 脚本，监控 `kernel4hls/<kernel>/<mode>s` 目录，并确保生成的 QoR CSV 与上述图数据一一对应。

本文档记录了 origin 版本的完整流水线，便于后续在此基础上做定制或对齐。
