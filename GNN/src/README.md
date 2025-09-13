目录概览与清理建议

本目录包含训练与模型实现的源码文件。为了保持 `src` 干净、专注于源码，训练产生的缓存与可视化结果已统一迁移到 `GNN/` 目录下。

主要结构
- 模型实现: `gat.py`, `rgcn.py`, `sage.py`, `pna.py`, `ggnn.py`, `arma.py`, `film.py`, `pan.py`, `sgn.py`, `unet.py`, `conv.py`
- 数据与特征: `dataset_pyg.py`, `features.py`, `features_numerical.py`, `features_binary.py`, `node_encoder.py`
- 训练与推理: `train.py`, `train_differential_e2e.py`, `inference.py`, `evaluate.py`, `check_dataset_valid.py`
  - 元信息读取已简化：不再依赖 `master.csv` 或 `meta.py`，所需默认值在代码中直接定义。
- 运行时产物(已搬迁):
  - 训练可视化输出: `GNN/differential_output_e2e/`
  - 图缓存: `GNN/graph_cache/`

如何运行
- 请从 `GNN/` 目录运行脚本，例如：
  - `python src/check_dataset_valid.py ...`
  - `python src/train_differential_e2e.py --target_metric dsp`

路径规范（避免把产物写进 src/）
- `train_differential_e2e.py` 的默认输出目录和缓存目录已改为相对于项目 `GNN/` 的固定路径：
  - 输出: `GNN/differential_output_e2e/`
  - 缓存: `GNN/graph_cache/`
  即使从 `src/` 启动脚本，产物也不会再落入 `src/`。

版本控制
- 已在本目录添加 `.gitignore`，忽略 `__pycache__/`、运行时缓存与结果文件，避免干扰代码视图。

提示
- 如果你仍然看到旧的 `src/differential_output_e2e/` 或 `src/graph_cache/`，它们已被移动到 `GNN/` 对应目录。
