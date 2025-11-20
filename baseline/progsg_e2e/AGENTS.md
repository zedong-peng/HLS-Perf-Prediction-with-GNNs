/home/user/zedongpeng/workspace/ProgSG_origin @ProgSG_origin/ 是 ProgSG 的 GitHub 代码。
/home/user/zedongpeng/workspace/ProgSG_origin/2406.09606v3-2.pdf 是 ProgSG 论文。

ProgSG 端到端复刻脚本位于 /home/user/zedongpeng/workspace/HLS-Perf-Prediction-with-GNNs/progsg_e2e。

## 训练入口

- ForgeHLS 精简 10-design：`./run_10designs.sh [extra-python-args]`
- ForgeHLS 精简 100-design：`./run_100designs.sh [extra-python-args]`

两份脚本默认参数对齐 `ProgSG_origin/train/ProgSG/src/config.py` 中的训配置：

- 批大小 `24`
- 训练轮数 `800`
- GNN 隐藏维度 `512`
- GNN 层数 `8`
- GNN 注意力头 `8`
- 优化器学习率 `1e-5`
- CodeT5 序列截断长度 `64`
- Code 模态 transformer 层/头：`4` 层、`8` 头
- 默认启用 `--code_node_interaction`

可通过附加参数覆盖，脚本会把附加参数拼接到 Python 命令末尾。

## 参考

- 原始实现：`/home/user/zedongpeng/workspace/ProgSG_origin/train/ProgSG/`
- ForgeHLS 数据：`/home/user/zedongpeng/workspace/Huggingface/`
