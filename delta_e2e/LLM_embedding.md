LLM 设计代码表征流程
- 输入：设计代码字符串（单条或批量 list[str]）。
- Tokenizer：Qwen2.5-Coder-1.5B-Instruct 自带 tokenizer，对齐模型词表；自动 padding/truncation（默认 max_length=2048）。
- 模型前向：AutoModelForCausalLM（同目录加载，trust_remote_code=True）；输出最后一层 hidden states，形状 (batch, seq_len, hidden_size)。
- Pooling：
  - 默认 last_token：按 attention_mask 取每条序列的最后非 padding token。
  - 可选 mean：masked mean pooling。
- 归一化：可选 L2 normalize（默认开启）。
- 输出：张量 (batch, 1536)，1536 为模型 hidden_size。
- 设备/精度：自动选 GPU/CPU；CUDA 时默认 fp16，否则 fp32。
- 入口：`LLM_embedding.py` 内的 `LLMEmbedder.encode`（torch.Tensor）或 `encode_to_numpy`。

# easy to understand
## embedding
code -> tokenizer -> AutoModelForCausalLM -> hidden states (batch, seq_len, hidden_size) -> last pooling -> (optional L2) -> 1536-d embedding

## inference (对比用)
code -> tokenizer -> AutoModelForCausalLM -> hidden states -> lm_head (linear layer) -> logits -> tokenizer.decode -> text


##
/home/user/zedongpeng/workspace/HLSBatchProcessor/src/dataset_csv.py：get_source_code可以得到design目录下的所有source code 我这里特征只需要.c和.cpp

## 计划：在 train_e2e.py 引入 design code 特征（只改设计侧）
- 开关设计：新增 CLI `--use_code_feature`（默认 False）；开启时才读取/编码 design 代码，其他路径保持兼容。
- 代码收集：沿用 `get_source_code` 逻辑，只取 design 目录下的 `.c/.cpp`，按文件名排序后串接为单条字符串；一般只有一个文件，直接拼接即可。
- LLM 编码：复用 `LLMEmbedder`（配置通过 CLI 传递模型路径、pooling、max_length）；输出 shape (1, hidden_size=1536)。
- 缓存策略：在 `graph_cache` 下新建 `code_embeddings/`（按 design base_path + 代码内容哈希 + 模型路径 + pooling + max_length + normalize 的哈希命名 `.pt`）；先查缓存再跑 LLM，源码变化会命中不同哈希，避免误用旧向量。
- Pair 存储：仅对 design 增加 `design_code_embedding`（Tensor 或 None）和 `code_hash` 元数据，写入 pair cache；kernel 不需要代码模态。
- DataLoader：`E2EDifferentialDataset.__getitem__` 返回 `design_code_embedding`；collate 将非空拼成 batch，缺失时返回 None，不影响旧缓存。
- 模型融合：`SimpleDifferentialGNN` 新增 design code 投影头（LayerNorm + Linear -> hidden_dim）；开启时在 `delta_input` 里 concat `design_repr` 与 `design_code_repr`（和 `kernel_repr`）；`delta_head` 输入维度随开关调整（关=2*hidden_dim，开=2*hidden_dim+hidden_dim_code）。
- 训练/日志：SwanLab 记录 `config/use_code_feature`、模型路径等，便于对比 code 模态影响；不开时行为与现有完全一致。

## 性能与推理策略
- 模型只加载一次：LLMEmbedder 懒加载，进程内单实例，避免多进程重复占显存。
- 预取与批处理：先收集所有未命中缓存的 design 源码，做内容哈希；按批次（默认 8–16，可配置）调用 `embedder.encode`，用 `torch.inference_mode()` + `autocast`，逐批写缓存。
- 不用多模型/多进程：4090 + 1.5B 模型，串行 batch 足够；多进程会爆显存，收益有限。
- vLLM 不采用：需求是拿 hidden states 做 pooling，vLLM 对这类 embedding 场景不友好，直接 HF 模型批前向更稳。
