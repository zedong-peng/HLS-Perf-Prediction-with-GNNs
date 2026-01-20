## DSP 资源缺口汇报

### 总览（要点汇总）
- 现有 ADB 仅给出优化后 CDFG 的基数资源，缺少 loop→节点映射、unroll/pipeline 并行度、接口/FSM 开销，无法重建展开倍率。
- 部分设计仅有 `kernel_*.adb`（无 loop 子图），部分设计有 `kernel_*_Pipeline_VITIS_LOOP_*`（管线快照，可能重复同一硬件），简单累加会“重算或漏算”。
- csynth.xml 的 DSP/LUT 是最终展开后的总量，常见现象：csynth > ADB（缺少展开/控制/接口开销）或 ADB > csynth（多个管线快照重复计）。
- 过滤策略：`--filter_resource_mismatch true` 可剔除 ADB 求和 ≠ csynth 的样本；否则只能人工核查。
- 相关脚本与输出：`dsp_gap_probe/analyze_gap.py`、`dsp_gap_probe/lut_analysis_top10.py`；报告位于 `output/train_split_error_analysis/dsp_analysis_top10/dsp_report.txt` 和 `output/train_split_error_analysis/lut_analysis_top10/lut_report.txt`（路径相对 `delta_e2e/`）。

### ADB 目录与结构（例：symm_design_1001）
- 目录：`design_1001/project/solution1/.autopilot/db/`
- 文件：仅有 `kernel_symm.adb`，无 `VITIS_LOOP_*.adb`
- 内容：`dp_component_resource` 列出算子资源（如 `dmul_*` DSP=8、`dadd_*` DSP=3），DSP 求和 82
- 缺失：无 `<loop>` 节点/`loop_id`/`unroll_factor` 等结构化字段，无法映射算子到 loop

### 不匹配示例（csynth DSP > ADB DSP 求和）
- symm_design_1001：csynth 94 vs ADB 82（缺口 12）
- syr2k_design_312：csynth 49 vs ADB 43（缺口 6）
- trmm_design_892：csynth 51 vs ADB 23（缺口 28）
上述设计的 ADB 均无 `<loop>` 结构，也无 `VITIS_LOOP_*.adb` 子图。

### 原因
- ADB 只给出算子基数资源，未提供 loop 层级/并行度（unroll/pipeline）映射，无法推断展开后的实例数。
- csynth.xml 的 DSP 是展开后的总实例数（含 unroll 副本、pipeline 并行、接口/尾部处理），因此出现 csynth > ADB 的缺口。
- HLS 源代码的 loop 配置无法直接对齐到 ADB 图，因为缺少 loop→节点的结构化标注。

### 现有对策
- `--filter_resource_mismatch true`：丢弃 ADB 求和 ≠ csynth 的样本，降低噪声。
- 分析阶段记录缺口倍数（csynth/ADB），便于人工排查。

### 深度方案（需额外投入）
- 调研 ADB 其他位置或 csynth loop 报告，重建 loop→节点映射，乘以 unroll/pipeline 并行度补偿 DSP。
- 目前 ADB 格式缺少必要字段，需更多解析工作或更完整的导出。现阶段无法自动补上缺口，只能过滤或人工分析。

### 额外探测（示例结论）
- 尝试解析不匹配案例的 `csynth.xml` 中 `<LoopPerformanceEstimates>`，结果 symm/syr2k/trmm 三例 loop entries 均为 0（报告层面无 loop 信息）。
- ADB 侧只有 `kernel_*.adb`，无 `VITIS_LOOP_*.adb`，且 `<loop>` 节点缺失（loop elements=0）。文件文本虽含 “loop_id” 字样，但非标准节点，无法建立节点→loop→unroll 映射。
- 因此在现有文件集合上无法补出展开倍数，也无法让 ADB DSP 与 csynth DSP 自动对齐；需要更完整的导出或手工/过滤处理。

### 追加探测（trmm_design_892 重新综合，Vitis HLS 2023.2）
- 路径：`PolyBench/trmm/design_892/project/solution1/.autopilot/db` 与 `.../syn/report` 新增 XML/ADB，但仍仅有 `kernel_trmm.adb`，无 `VITIS_LOOP_*.adb`。
- `csynth.xml` 的 `<LoopPerformanceEstimates>` 仍为空；ADB 中未出现标准 `<loop>` 节点，无法推断 unroll/pipeline。
- 结论：即便重新跑一次，缺口依旧，原因仍是缺少 loop→节点映射，自动补齐展开倍数不可行；暂以过滤或人工核查为主。

### LUT 差异的具体观察
- ADB 不是“最终展开后的净表”，而是 CDFG/管线快照的资源统计，既可能漏算也可能重复算。
- 例：`heat-3d/design_407` 有 `kernel_heat_3d.adb` (18117 LUT) + 多个 `kernel_heat_3d_Pipeline_VITIS_LOOP_*` (各 3660 LUT)，总和 32757，而 csynth.xml 只有 25955，说明多个管线视图被重复计入。
- 例：`doitgen/design_220` 的 kernel 1752 LUT，加 8 个 VITIS_LOOP_* 各 20 LUT，总 1912，而 csynth.xml 为 11825，说明 ADB 仅包含数据通路算子基数，缺失接口/控制/FSM 等大量 LUT。
- 例：`seidel-2d/design_115` 只有 `kernel_seidel_2d.adb` 13912 LUT，csynth.xml 为 31635，缺少所有 loop 级展开/接口逻辑。
- 结论：ADB 记录的是单份算子基数（有时按 loop 拆分），缺少实际共享/复用关系、接口/控制逻辑、loop 展开倍率；简单累加 ADB 会出现“重算”或“漏算”，与 csynth.xml 的总 LUT 不等。

### FSM/控制开销缺失（导致 ADB 求和 < 报告）
- 示例路径：`PolyBench/doitgen/design_220/project/solution1/.autopilot/db/kernel_doitgen.adb`。
- 文件中存在 `<fsm>` 段（307 个 states、333 条 transitions），但 state 仅含 `id` 和空的 `operations`，没有 loop_id/unroll，也没有资源计数。
- `dp_component_resource` 仅统计数据通路算子基数，FSM/控制/接口握手的 LUT/FF 开销未被计入，因此 csynth.xml 的总 LUT/FF 往往大于 ADB 求和。

### cdfg_regions 为空（无法用 region 重构）
- 示例路径同上：`PolyBench/doitgen/design_220/.../kernel_doitgen.adb`。
- `<cdfg_regions>` 存在但 `sub_regions` 为空，没有 region→节点分组，更没有 loop/unroll/parent 信息。
- 其他样例（heat-3d/syr2k/symm/trmm）同样无有效 region 数据，因此无法依赖 region 信息重建展开或校正资源。

### 关于 `kernel_*_Pipeline_VITIS_LOOP_*` 是否能用于“展开 loop”
- 这些文件也是 ADB/CDFG 视图，只是把内层管线或 loop 拆成独立快照，节点资源与 kernel 中的基数资源相同或部分重用。
- 现有文件缺少“节点→loop/阶段”的映射与展开倍数，因此无法可靠地把 kernel 图与每个 VITIS_LOOP_* 对齐，更无法据此推断 unroll/pipeline 并行度。
- 直接“把所有 ADB 图合并”会重复计数同一硬件（如 heat-3d 例子），不能得到真实展开后的资源。
- 若要尝试构建映射，需要额外的 loop 报告（`report_loop_tripcount`/pipeline/unroll 报告）或 VITIS_LOOP_* 中的 loop_id/parent 关系，但当前导出文件不存在这些字段。
- 结论：在现有 ADB 集合下无法自动通过 VITIS_LOOP_* 图重建展开关系，只能过滤不匹配样本或依赖额外导出/手工分析。

### 现有 ADB 能否重建展开关系？
- 不能。缺少节点→loop/阶段映射、unroll/pipeline 并行度、接口/FSM 资源等关键字段。
- `kernel_*_Pipeline_VITIS_LOOP_*` 与顶层 kernel 图没有可用的 parent/loop_id 关联，简单合并会重复或漏算。
- `csynth.xml` 的 `<LoopPerformanceEstimates>` 为空，无法补充并行度。
- 当前文件集合不足以完成该工作。

### 参考资料
- PDF：`delta_e2e/2025-7.4_adb_graph_analysis.pdf`（说明 ADB/CDFG 已体现部分优化后的节点，但仍缺 loop 映射与控制开销）。
- 脚本：`dsp_gap_probe/analyze_gap.py`（DSP/LUT 差异探测）、`dsp_gap_probe/lut_analysis_top10.py`（Top10 LUT 汇总）。
- 报告输出：`output/train_split_error_analysis/dsp_analysis_top10/dsp_report.txt`、`output/train_split_error_analysis/lut_analysis_top10/lut_report.txt`。
