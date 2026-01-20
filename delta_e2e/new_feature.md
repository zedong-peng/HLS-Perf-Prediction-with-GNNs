## 新特征方案：流水线/端口拆分/资源映射改进（尚未合入）
Design: PolyBench/2mm design_12（post‑HLS，多 ADB）  
Goal: 提升 DSP/FF/LUT 预测，直接利用综合后的可见信号（不反推 pragma，不引入图级特征）  

ADB 中可用的信息
- 主 ADB：`cdfg`（nodes/edges/blocks/ports 已用）；`cdfg_regions` 里大多 `mII/mDepth=-1`；无 interval/pipe_depth/ap_enable_reg_pp。
- 子 ADB（如 `kernel_2mm_Pipeline_VITIS_LOOP_72_6.adb`, `_72_61.adb`）：存在 `<interval>/<pipe_depth>`，rtlName/常量名含 `Pipeline`，DSP/FF/LUT 在 `dp_component_resource`。
- 端口已拆分（如 `A_0_0` 等），`direction/if_type/array_size` 可用，partition 需靠命名/数量推断。
- `dp_component_resource` 覆盖主要算子与 pipeline 组件；`dp_register_resource` 等提供 FF 旁证（暂不扩展资源类型）。

改动方案（保持无图级特征，聚焦节点/区域）
1) 流水/循环信息（逐 ADB 保留，拼图后仍在）
   - 解析每个 ADB 时：优先读取 `cdfg_regions` 的 `mII/mDepth/mIsDfPipe`；若存在 `<interval>/<pipe_depth>`（常见于子 pipeline ADB），填入 `region_ii/region_pipe_depth` 并置 `region_is_pipelined=1`。缺省则维持 0。
   - 区域标注：在 region 开启时把上述值写到对应 basic block 节点；hierarchical 开启时也写入 region 节点（保留 mTag/mType，用 mType 区分循环/普通）。
   - 名称含 `Pipeline` 的节点可兜底设 `region_is_pipelined=1`，防止缺字段漏标。

2) Partition/端口信号
   - 构造端口特征时解析端口名基名与 bank 索引（如 `A_0_1` -> base `A`, bank=1），对“同 base 端口数”做桶化特征；direction/if_type 也加一个小桶，直接表达拆分度对资源复制的影响。
   - 完全依赖端口拆分后的现状，无需恢复 pragma 因子。

3) 资源映射稳健性
   - 仅保留 DSP/FF/LUT。`dp_component_resource` 能映射到节点（rtlName）；未匹配到时节点资源维度保持 0，必要时按算子类型给一个轻量默认桶化值作为兜底（不新增图级属性）。

4) 缓存/开关
   - 推荐 `--region on`（写入流水三维，缺省为 0）；`--hierarchical` 视显存决定，开启可让 region 节点承载 mTag/mType/ii/depth。特征变更需 bump `feature_version` 并重建 `graph_cache`。
