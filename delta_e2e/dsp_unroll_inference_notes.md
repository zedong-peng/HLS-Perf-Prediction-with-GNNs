## DSP 展开倍数推断思路（草案）

- 适用场景：模型训练/误差分析需要解释“csynth DSP > ADB DSP 求和”的样本，怀疑 pipeline/unroll 副本未在 ADB 中显式计数。
- 建议执行人：熟悉 Vitis HLS CDFG/ADB 结构的开发者（非轻量脚本工作），需能解析 loop 层级和节点归属。

### 最低可行实现（近似）
- 输入：设计目录；输出：每个 ADB 文件的 DSP 求和、与 csynth 差值、估计的“缺口倍数”（csynth_dsp / adb_sum）。
- 步骤：
  1. 继续用现有 `dp_component_resource` 统计 ADB DSP 总和，计算缺口。
  2. 在 report 中记录缺口倍数，作为“有效展开倍数”参考，不映射到具体 loop，仅供分析或过滤。
- 作用：快速提示缺口大小；不需要深入 CDFG 映射。

### 深度推断方案（需要额外工作）
- 目标：从 ADB 中解析 loop/pipeline/unroll 配置，推断算子所在 loop 的展开倍数，并将倍数乘以算子 DSP，得到更接近 csynth 的总数。
- 解析来源：
  - `.autopilot/db/kernel_*.adb` 的 `<loops>` 节点（`name`, `tripcount`, `unroll_factor`, `pipeline`, `II`）。
  - `VITIS_LOOP_*` 文件名/内容中的 loop 层级信息（部分 adb 会写 `<name>VITIS_LOOP_xxx</name>`）。
  - CDFG 节点与 loop 的层级关系（需要遍历节点所属的 `loop_id` 或父链）。
- 推断流程（示意）：
  1. 建立 loop 树：解析 `<loops>`，记录每个 loop 的 unroll_factor/pipeline/II。
  2. 建立节点归属：在 `cdfg` 中找到运算节点与 loop 的绑定（如果缺失，需要基于行号/作用域近似）。
  3. 对每个节点的 DSP：`dsp_node * effective_factor`，其中 factor = ∏(loop.unroll_factor 或 pipeline 并行度)。若 pipeline 仅影响时序不复制算子，可选择只用 unroll。
  4. 汇总估计 DSP，并与 csynth 对比，输出“补偿后总数”和映射明细。
- 复杂度/风险：
  - ADB 不总是显式标注节点-循环绑定，可能需要启发式匹配。
  - pipeline 的 II 与复制关系在 HLS 不总是一一对应，容易误判。
  - 需要较多 XML 结构实验和校验样例。

### 与 delta_e2e 训练的关系
- 训练标签使用 csynth DSP，特征来自 ADB 图。若缺口不补偿，特征可能低估少量样本的 DSP；影响较小但会引入噪声。
- 若实现“补偿后 DSP”：
  - 可作为额外特征/过滤依据（例如在 Processor 内存储 `adb_dsp_sum_adjusted`）。
  - 也可用于自动过滤“严重缺口”样本，以降低噪声。

### 后续建议
- 先上线“缺口倍数”近似，便于快速发现问题样本。
- 如需精确展开推断，再投入时间做 loop 解析验证；优先选 1~2 个不匹配案例（如 symm/syr2k/trmm）手工对比，验证启发式可行性后再工程化。
