
# ForgeHLS PolyBench（100 designs）上复现 HGNN 的步骤

数据已经在 `/home/user/zedongpeng/workspace/Huggingface/forgehls_lite_100designs_polybenchpart` 跑过 `vitis_hls -f *.tcl`，下面的流程说明如何把这些结果接入 `baseline/HGNN/origin`，并得到论文同款的 HGNN 训练 / 测试输出。实际去实现脚本之前，先确认每一步都可行。

## 0. 环境与路径约定
- 保持当前仓库目录为 `baseline/HGNN/origin/hierarchical-gnn-for-hls`，该目录下自带 `benchmark/`, `GNNp/`, `GNNnp/`, `GNNg/`。
- 设定数据根：`export DATA_ROOT=/home/user/zedongpeng/workspace/Huggingface/forgehls_lite_100designs_polybenchpart/PolyBench`
- HGNN 代码期待的原始标签文件放在 `GNN*/data/raw/source_label.csv`；若要区分不同数据版本，可在 `GNN*/data/raw/` 下额外存放 `source_label_polybench.csv` 再做符号链接。

## 1. 汇总 QoR 指标成 `source_label.csv`
1. 遍历 `$DATA_ROOT/<kernel>/design_*/project/solution1/syn/report/csynth.xml`，抽取：
   - `PerformanceEstimates/SummaryOfOverallLatency` 中的 `Best-caseLatency`、`Interval-min`（II）
   - `AreaEstimates/Resources` 中的 `LUT`, `FF`, `DSP`
   - 如果需要 `IL`，可用 `SummaryOfLoopLatency` 里核心 loop 的 `IterationLatency`（或将其和 II 统一到 1-cycle 粒度）。
2. 解析每个 design 的 pragma 组合，转成 HGNN 期望的 `config` 字符串（`<array_cfgs>__<loop_cfgs>`）：
   - `array_cfgs`: 从 `run_hls.tcl` 中的 `set_directive_array_partition` / `set_directive_array_reshape` 捕获 `(数组名, 维度, 因子)`；拼成 `A_1_4_B_2_2...`。
   - `loop_cfgs`: 从 `set_directive_pipeline` / `set_directive_unroll` / `set_directive_loop_flatten` 中读取 `(loop_name, 是否 pipeline, unroll 因子, 是否 flatten)`；保证最多 3 层，不足用 `None` 填充。
   - 可以直接复用 `origin/hierarchical-gnn-for-hls/GNNp/data/datasets/utils.py` 里的 `parse_*` 逻辑做逆运算，以确保字段顺序一致。
3. 把字段写成 CSV：`kernel,config,latency,lut,dsp,ff,II,IL`。建议写一个脚本 `python e2e/collect_forgehls_labels.py --data_root $DATA_ROOT --out_csv origin/.../GNNp/data/raw/source_label.csv` 来生成，并复用到 `GNNnp`, `GNNg`（软链接即可）。

## 2. 对齐 kernel 源码与 HGNN 构图脚本
1. HGNN 的图构建会读取 `benchmark/kernel4graph/<kernel>/<kernel>_<loop>.c`。如果 ForgeHLS 版本的源码与仓库自带版本一致，可直接沿用；否则请把 `$DATA_ROOT/<kernel>/design_*/<kernel>.c` 中的基准代码复制/链接到 `benchmark/kernel4graph/<kernel>/`，保证每个 loop 拥有独立文件。
2. 若生成 IR 前需要替换 pragma（`add_unroll_factors_to_c` 会往文件里插值），确保这些 C 文件是可写的临时副本，避免覆盖原始 dataset，可在 `benchmark/kernel4graph` 下放一份可摧毁 copy。

## 3. 生成 PyG 数据集
1. 进入对应目录，执行：
   ```bash
   cd origin/hierarchical-gnn-for-hls/GNNp/data/datasets
   python run.py        # 默认 kernel 列表在脚本底部，需要改成 ForgeHLS 覆盖的 PolyBench 列表
   ```
2. 同样流程跑 `GNNnp/data/datasets/run.py` 与 `GNNg/data/datasets/run.py`。运行前请确认 `init_directories()` 不会误删你已有的数据备份。
3. 构图完成后，检查 `GNN*/data/processed/pt/` 下的 `.pt` 文件数量与 `source_label.csv` 行数一致，再用 `python GNN/src/check_dataset_valid.py --dataset_name PolyBenchForgeHLS`（在更高一层的 GNN 项目里提供）核对统计信息。

## 4. 训练三层 HGNN
1. 对应命令示例（确保 `python -m pip install -r requirements.txt` 已完成，并在 `origin/hierarchical-gnn-for-hls` 下）：
   ```bash
   cd GNNp/src && python main.py --target dsp --gnn_type sage --epoch_num 250 --batch_size 16
   cd GNNnp/src && python main.py --target dsp --gnn_type sage --epoch_num 250 --batch_size 16
   cd GNNg/src && python main.py --target dsp --gnn_type sage --epoch_num 250 --batch_size 16
   ```
2. 需要其它 QoR（LUT/FF/Latency/II）时，重复训练并切换 `--target`。`saver/logs/` 里能找到 `val_model_state_dict_*.pth` 与训练日志。

## 5. 推理与指标汇总
1. 如果只看测试集 MAPE，可直接阅读 `logs/*/events.log` 中 `Test MAPE`；如需在 ForgeHLS 设计上重新跑推理，可写一个小脚本加载 `.pth`，对 `processed/pt` 做前向（`train.py:eval` 已提供参考）。
2. 论文使用层次化组合：可把 `GNNp/GNNnp` 的预测或 embedding 作为额外特征输入 `GNNg`，或者复现实验时以 3 个模型的单独指标报告。
3. 最终把指标与原论文表格对齐：记录 kernel 覆盖范围、平均/最大误差，并注明 ForgeHLS Lite 的实际 design 数量。

---

## 一些工程挑战 / 待确认事项

1. **Pragma 反向解析**  
   - ForgeHLS Lite 里给出的 `run_hls.tcl`/`design_xxx` 并不是逐 loop 拆开的 `kernel_<loop>.c`，而是整份源文件配合 Tcl 指令。要复现 HGNN，需要把 pragma 组合转换为 `<array_cfgs>__<loop_cfgs>`，字段顺序、补齐规则必须与 `GNN*/data/datasets/utils.py` 的 `parse_*` 完全对齐，否则构图脚本读不回来。  
   - 多层循环的命名需与 `loop_blocks` 一致（`lp1/lp2/...`），否则 `get_supernode_feature()` 匹配不到 supernode，样本会被丢弃。

2. **源码同步与可写副本**  
   - HGNN 构图过程中会改写 `benchmark/kernel4graph/<kernel>/<kernel>_<loop>.c`（插 pragma、插 supernode、删函数），如果我们直接指向 ForgeHLS 原文件，可能污染原始设计；最好先同步一份“可破坏”版本到 `benchmark/kernel4graph`，并保证命名/loop 顺序与 `loop_blocks` 要求一致。

3. **三阶段数据/模型依赖链**  
   - 训练顺序必须是 `GNNp → GNNnp → (推理生成 CSV) → GNNg`。`GNNg/data/datasets/run.py` 在第 32-70 行会访问 `GNNp_pred_resource.csv` 等文件，缺一不可。也就是说，在 ForgeHLS 数据集上想跑 GNNg，需要先写推理脚本，把前两层模型对所有设计的 LUT/FF/DSP/Latency 预测导出为 CSV。

4. **dataset/run.py 里的 kernel 列表**  
   - 默认只包含 8 个 kernel（`['atax','bicg',...]`）。需要结合 ForgeHLS Lite 的子集（如 26 个 PolyBench kernel）去扩充列表，并确认 `loop_blocks/flatten_loops` 覆盖所有嵌套结构；对新增 kernel 需手动补齐 loop 拆分配置。

5. **算子特征库/LLVM 版本**  
   - `graph.py` 使用 `clang` + Programl 生成 IR，并引用静态资源映射（`fadd`→DSP=2 等）。ForgeHLS Lite 若使用 double/int 以外的类型或有自定义 IP，需确认这些映射仍适用，否则需要扩展 `resource_mapping`/`latency_map`。

6. **验证与对齐**  
   - 在大规模 dataset 运行前，可选 1-2 个 kernel 做 sanity check：从 ForgeHLS 提取某个 config → 生成 `.pt` → 手工检查 `data.x` 中的 supernode 特征是否与 HLS report 接近，以避免全量数据跑完才发现字段错位。

> ✅ 以上步骤确认后，再逐条实现脚本（例如标签汇总、推理脚本、loop_blocks 配置扩展）即可完成 ForgeHLS PolyBench 数据集上的 HGNN 复现。
