# Qwen Coder QoR 回归流水线说明

本目录提供了一套以 **Qwen2.5-Coder** 为特征提取器、冻结大模型权重的 HLS QoR 预测流程，实现从数据构建、表征缓存、回归头训练到推理落地的全链路脚本。

## 核心技术思路

1. **描述式模板输入**
   - 针对每个设计样本渲染固定模板（`[HLS_QOR_SAMPLE] ... [/HLS_QOR_SAMPLE]`），严格包含 `device/tool/clock/context/code` 等字段，确保训练、验证、推理三阶段模板一致，避免 Prompt 漂移。
   - `context` 字段汇总 pragma、循环数量等结构化信息，代码正文保留关键循环与 pragma，以增大模型获取性能关键信息的概率。

2. **Coder 冻结特征提取**
   - 直接加载本地 Qwen2.5-Coder (~1.5B) 模型，完全冻结权重，仅做前向。
   - 对输入序列执行 `max_length=2048` 的截断与 padding，提取中后层 hidden states，并做 mask mean-pooling 得到句向量。
   - 通过 **层探针（layer probe）** 在验证集上对多层候选（默认 0.6/0.7/0.8/0.9 × L）训练小型岭回归探针，自动选择 MAE 最优的层作为主干层。

3. **向量缓存与复用**
   - `cache/embeddings/` 下缓存 `{层号 → numpy 向量}`，含模板版本、模型名、max_length 等元信息，避免重复推理，节省 4090 显存与时间。
   - 需要重建可加 `--rebuild_cache`，也可针对 OOD 数据单独缓存。

4. **特征与目标预处理**
   - 对句向量使用 `StandardScaler` 做 z-score；训练后保存到 `x_scaler.pkl`。
   - 四个 QoR 目标全部纳入：`[LUT, DSP, FF, latency_cycles]`；其中 `LUT/FF/latency` 先 `log1p` 再 z-score，`DSP` 直接 z-score，并在 `y_stats.json` 中记录均值、方差及 log 掩码，保证推理/评估时可正确逆变换。

5. **多任务 MLP 回归头**
   - 结构：`LayerNorm → Linear(emb→256) → ReLU → Dropout(0.1) → Linear(256→256) → ReLU → Linear(256→4)`。
   - 损失：Huber（`beta=1.0`），优化器：AdamW（`lr=1e-3`，`weight_decay=1e-2`），带 Cosine 调度与 warmup（`ratio=0.05`）。
   - 大批量（默认 1024）+ 20~40 epoch + `patience=8` 的 early stopping，记录验证集 MAE 最佳权重。

6. **评估与产物**
   - 自动生成 ID / OOD 数据拆分并在真实单位下报告 `MAE/RMSE/R²`（先逆标准化，再对 log 目标 `expm1`）。
   - 输出产物：
     - `model.pt`：训练好的 MLP 参数。
     - `x_scaler.pkl`：特征标准化器。
     - `y_stats.json`：目标均值、方差、log 掩码。
     - `config.json`：包含使用的 coder 模型、最佳层号、模板版本等。
     - `metrics.json`、`test_predictions.npz`、`ood_predictions.npz`：指标与预测详情。

7. **推理流程（部署）**
   1. 渲染相同模板文本。
   2. coder 前向 → 取指定层 hidden states → mean-pool（bf16 可提升速度/显存）。
   3. 使用 `x_scaler.pkl` 标准化向量。
   4. 调用 `model.pt` MLP 产生归一化预测，按 `y_stats.json` 反标准化并对 log 目标做 `expm1`。
   5. 输出 QoR 指标 JSON，例如 `{"LUT": 12345, "DSP": 12, "FF": 45678, "latency_cycles": 9876}`。

8. **4090 运行建议**
   - 嵌入阶段 batch size 可 8~16，默认 12；使用 bf16 前向，关闭梯度与缓存。
   - 若样本较长，可按需提升 `max_length` 至 3072/4096；推荐先确认关键信息在 2048 token 内。
   - 向量缓存生成一次即可，后续调参只需读取 cached embedding 加速迭代。

9. **可选扩展（Δ 任务）**
   - 进一步可将 kernel/design 分别编码构造差分特征，叠加 kernel/delta 头与一致性损失；暂未在当前脚本中实现，留待后续迭代。

## 模型架构详解（`train_llm_e2e.py`）

- **文本编码主干**（冻结）：`AutoTokenizer` 对模板文本做截断/填充后交给 `AutoModel`（Qwen2.5-Coder）。仅执行前向，获取所有层的 hidden states，并按 attention mask 做均值池化得到句向量。
- **层探针与候选挑选**：对预设的多层候选向量训练轻量岭回归探针（验证集 MAE 最低者入选），最终只保留最佳层的 mean-pooled 向量作为下游输入。
- **特征标准化**：`StandardScaler` 对选定层的句向量做 z-score 变换；推理时复用同一 scaler。
- **RegressionHead**（`LLM_as_predictor/train_llm_e2e.py:81`）：
  - `LayerNorm(in_dim)`
  - `Linear(in_dim → 256)` + `ReLU`
  - `Dropout(p=0.1)`
  - `Linear(256 → 256)` + `ReLU`
  - `Linear(256 → 4)`，同时输出 `LUT/DSP/FF/latency_cycles`
- **损失与优化**：Huber 损失（对四个目标同时回传），`AdamW(lr=1e-3, weight_decay=1e-2)`，配合余弦退火调度和 warmup。
- **输出反变换**：利用保存的 `y_stats.json` 恢复标准化与 `log1p`，得到真实量纲预测值。

## 主要脚本

| 脚本 | 作用 |
| --- | --- |
| `train_llm_e2e.py` | 主训练入口：构建数据、缓存 coder 向量、层探针、训练回归头、评估。 |
| `run_10designs.sh` | ForgeHLS 10-design 子集默认跑法，封装常用参数。 |

执行示例：

```bash
bash run_10designs.sh --rebuild_cache
```

首次运行会生成嵌入缓存，后续如不变更模板/模型即可直接复用，显著缩短训练耗时。

## 目录结构概览

```
LLM_as_predictor/
├── dataset_csv.py          # 原有设计数据解析与源码嵌入工具
├── train_llm_e2e.py        # 新版 coder QoR 回归主脚本
├── run_10designs.sh        # 样例跑法
├── cache/                  # 嵌入缓存（运行时生成）
└── outputs/                # 训练日志与模型产物（运行时生成）
```

欢迎基于上述流程继续扩展，比如更细粒度的 OOD 划分、特征解释或多任务 Delta 方案等。
