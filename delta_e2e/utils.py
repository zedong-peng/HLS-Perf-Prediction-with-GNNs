#!/usr/bin/env python3
"""
HLS Performance Prediction Utilities
====================================

这个模块包含了从pre_process_forgehls.py提取的关键组件，
用于图处理和特征提取的核心功能。

Author: Zedong Peng
"""

import xml.etree.cElementTree as et
import networkx as nx
import re
from typing import Dict, List, Optional, Any


# ============================================================================
# 特征定义和配置
# ============================================================================

allowable_features = {
    'node_category': ['nodes', 'blocks', 'ports', 'regions', 'misc'],
    'bitwidth': list(range(0, 256)) + ['misc'],
    'opcode_category': ['terminator', 'binary_unary', 'bitwise', 'conversion', 'memory', 'aggregate', 'other', 'misc'],
    'possible_opcode_list': [
        'br', 'ret', 'switch',
        'add', 'dadd', 'fadd', 'sub', 'dsub', 'fsub', 'mul', 'dmul', 'fmul', 'udiv', 'ddiv', 'fdiv', 'sdiv', 'urem', 'srem', 'frem', 'dexp', 'dsqrt',
        'shl', 'lshr', 'ashr', 'and', 'xor', 'or',
        'uitofp', 'sitofp', 'uitodp', 'sitodp', 'bitconcatenate', 'bitcast', 'zext', 'sext', 'fpext', 'trunc', 'fptrunc',
        'extractvalue', 'insertvalue',
        'alloca', 'load', 'store', 'read', 'write', 'getelementptr',
        'phi', 'call', 'icmp', 'dcmp', 'fcmp', 'select', 'bitselect', 'partselect', 'mux', 'dacc',
        'bitset',   # 新增，便于更细粒度统计（可选）
        'misc'
    ],
    'possible_is_start_of_path': [0, 1, 'misc'],
    'possible_is_LCDnode': [0, 1, 'misc'],
    'possible_cluster_group_num': [-1] + list(range(0, 256)) + ['misc'],
    'LUT': list(range(0, 1000)) + ['misc'],
    'DSP': list(range(0, 11)) + ['misc'],
    'FF': list(range(0, 1000)) + ['misc'],
    'possible_edge_type_list': [1, 2, 3, 4, 'misc'],  # 新增 4: region_contains
    'possible_is_back_edge': [0, 1],
    # 新增：流水线相关的节点特征空间（离散桶化，含 'misc' 兜底）
    'possible_region_is_pipelined': [0, 1, 'misc'],
    'region_ii_bucket': list(range(0, 65)) + ['misc'],
    'region_pipe_depth_bucket': list(range(0, 65)) + ['misc'],
}


# ============================================================================
# 辅助函数
# ============================================================================

def safe_index(l: List, e: Any) -> int:
    """安全获取元素索引"""
    try:
        return l.index(e)
    except:
        return len(l) - 1


def opcode_type(opcode: str) -> str:
    o = (opcode or 'misc').lower()
    if o in {'br', 'ret', 'switch'}:
        return 'terminator'
    if o in {'add', 'dadd', 'fadd', 'sub', 'dsub', 'fsub', 'mul', 'dmul', 'fmul',
             'udiv', 'ddiv', 'fdiv', 'sdiv', 'urem', 'srem', 'frem', 'dexp', 'dsqrt'}:
        return 'binary_unary'
    if o in {'shl', 'lshr', 'ashr', 'and', 'xor', 'or'}:
        return 'bitwise'
    if o in {'uitofp', 'sitofp', 'uitodp', 'sitodp', 'bitconcatenate', 'bitcast',
             'zext', 'sext', 'fpext', 'trunc', 'fptrunc'}:
        return 'conversion'
    if o in {'alloca', 'load', 'store', 'read', 'write', 'getelementptr'}:
        return 'memory'
    if o in {'extractvalue', 'insertvalue'}:
        return 'aggregate'
    if o in {'phi', 'call', 'icmp', 'dcmp', 'fcmp', 'select', 'bitselect', 'partselect', 'mux', 'dacc', 'sparsemux'}:
        return 'other'
    # 规则兜底：包含“bit”归到 bitwise；常见字符串模式辅助归类；否则 other
    if 'bit' in o:
        return 'bitwise'
    if any(tok in o for tok in ['add','sub','mul','div','rem','sqrt','exp']):
        return 'binary_unary'
    if any(tok in o for tok in ['load','store','read','write','alloca','gep','getelementptr']):
        return 'memory'
    if any(tok in o for tok in ['zext','sext','trunc','cast','ext','tofp','todp']):
        return 'conversion'
    if any(tok in o for tok in ['extract','insert','aggregate']):
        return 'aggregate'
    return 'other'


def get_rtl_hash_table(root) -> Dict[str, Dict[str, str]]:
    """获取RTL资源哈希表"""
    res_considered = ['FF', 'LUT', 'DSP']
    all_rtl = root.findall('*/res/*/item')
    rtl_table = {}
    if_add = False
    rep = re.compile(' \(.*\)')
    for i in all_rtl:
        res_table = {}
        rtl_name = i.find('first').text
        rtl_res = i.find('second')
        if rtl_name not in rtl_table.keys():
            for res in rtl_res.iter('item'):
                try:
                    res_name = res.findall('first')[0].text
                    res_num = res.findall('second')[0].text
                except BaseException:
                    break
                else:
                    if res_name in res_considered:
                        res_table[res_name] = res_num
                        if_add = True
        if if_add:
            rtl_table[re.sub(rep, '', rtl_name)] = res_table
        if_add = False
    return rtl_table


# ============================================================================
# 特征提取函数
# ============================================================================

def node_to_feature_vector(node: Dict) -> List[int]:
    """将节点转换为特征向量"""
    # 完全空节点：返回全 misc/默认桶
    if node == {} or node is None:
        node_feature = [
            len(allowable_features['node_category']) - 1,
            len(allowable_features['bitwidth']) - 1,
            len(allowable_features['opcode_category']) - 1,
            len(allowable_features['possible_opcode_list']) - 1,
            len(allowable_features['possible_is_start_of_path']) - 1,
            len(allowable_features['possible_is_LCDnode']) - 1,
            len(allowable_features['possible_cluster_group_num']) - 1,
            len(allowable_features['LUT']) - 1,
            len(allowable_features['DSP']) - 1,
            len(allowable_features['FF']) - 1,
            # 新增三维：流水线特征
            len(allowable_features['possible_region_is_pipelined']) - 1,
            len(allowable_features['region_ii_bucket']) - 1,
            len(allowable_features['region_pipe_depth_bucket']) - 1,
        ]
        return node_feature

    # 统一兜底读取，避免 KeyError
    category = node.get('category', 'misc')
    try:
        bitwidth_val = int(node.get('bitwidth', 0))
    except Exception:
        bitwidth_val = 0
    opcode_val = node.get('opcode', 'misc')
    try:
        is_start_of_path_val = int(node.get('m_isStartOfPath', 0))
    except Exception:
        is_start_of_path_val = 0
    try:
        is_lcdnode_val = int(node.get('m_isLCDNode', 0))
    except Exception:
        is_lcdnode_val = 0
    try:
        cluster_group_num_val = int(node.get('m_clusterGroupNumber', -1))
    except Exception:
        cluster_group_num_val = -1
    try:
        lut_val = int(node.get('LUT', 0))
    except Exception:
        lut_val = 0
    try:
        dsp_val = int(node.get('DSP', 0))
    except Exception:
        dsp_val = 0
    try:
        ff_val = int(node.get('FF', 0))
    except Exception:
        ff_val = 0

    # 提取统一的流水线特征（若缺失则置为0）
    try:
        region_is_pipelined_val = int(node.get('region_is_pipelined', 0))
    except Exception:
        region_is_pipelined_val = 0
    try:
        region_ii_val = max(0, min(64, int(node.get('region_ii', 0))))
    except Exception:
        region_ii_val = 0
    try:
        region_pipe_depth_val = max(0, min(64, int(node.get('region_pipe_depth', 0))))
    except Exception:
        region_pipe_depth_val = 0

    if category == 'nodes':
        node_feature = [
            safe_index(allowable_features['node_category'], category),
            safe_index(allowable_features['bitwidth'], bitwidth_val),
            safe_index(allowable_features['opcode_category'], opcode_type(opcode_val)),
            safe_index(allowable_features['possible_opcode_list'], opcode_val),
            safe_index(allowable_features['possible_is_start_of_path'], is_start_of_path_val),
            safe_index(allowable_features['possible_is_LCDnode'], is_lcdnode_val),
            safe_index(allowable_features['possible_cluster_group_num'], cluster_group_num_val),
            safe_index(allowable_features['LUT'], lut_val),
            safe_index(allowable_features['DSP'], dsp_val),
            safe_index(allowable_features['FF'], ff_val),
            # 新增：流水线特征
            safe_index(allowable_features['possible_region_is_pipelined'], region_is_pipelined_val),
            safe_index(allowable_features['region_ii_bucket'], region_ii_val),
            safe_index(allowable_features['region_pipe_depth_bucket'], region_pipe_depth_val),
        ]
    elif category == 'ports':
        node_feature = [
            safe_index(allowable_features['node_category'], category),
            safe_index(allowable_features['bitwidth'], bitwidth_val),
            len(allowable_features['opcode_category']) - 1,
            len(allowable_features['possible_opcode_list']) - 1,
            len(allowable_features['possible_is_start_of_path']) - 1,
            len(allowable_features['possible_is_LCDnode']) - 1,
            len(allowable_features['possible_cluster_group_num']) - 1,
            len(allowable_features['LUT']) - 1,
            len(allowable_features['DSP']) - 1,
            len(allowable_features['FF']) - 1,
            # 新增：流水线特征（端口无区域，统一置0）
            safe_index(allowable_features['possible_region_is_pipelined'], region_is_pipelined_val),
            safe_index(allowable_features['region_ii_bucket'], region_ii_val),
            safe_index(allowable_features['region_pipe_depth_bucket'], region_pipe_depth_val),
        ]
    elif category == 'blocks':
        node_feature = [
            safe_index(allowable_features['node_category'], category),
            len(allowable_features['bitwidth']) - 1,
            len(allowable_features['opcode_category']) - 1,
            len(allowable_features['possible_opcode_list']) - 1,
            len(allowable_features['possible_is_start_of_path']) - 1,
            len(allowable_features['possible_is_LCDnode']) - 1,
            len(allowable_features['possible_cluster_group_num']) - 1,
            len(allowable_features['LUT']) - 1,
            len(allowable_features['DSP']) - 1,
            len(allowable_features['FF']) - 1,
            # 新增：流水线特征（块节点上优先体现）
            safe_index(allowable_features['possible_region_is_pipelined'], region_is_pipelined_val),
            safe_index(allowable_features['region_ii_bucket'], region_ii_val),
            safe_index(allowable_features['region_pipe_depth_bucket'], region_pipe_depth_val),
        ]
    elif category == 'regions':
        # 区域节点：无 opcode/bitwidth，承载流水线属性
        node_feature = [
            safe_index(allowable_features['node_category'], category),
            len(allowable_features['bitwidth']) - 1,
            len(allowable_features['opcode_category']) - 1,
            len(allowable_features['possible_opcode_list']) - 1,
            len(allowable_features['possible_is_start_of_path']) - 1,
            len(allowable_features['possible_is_LCDnode']) - 1,
            len(allowable_features['possible_cluster_group_num']) - 1,
            len(allowable_features['LUT']) - 1,
            len(allowable_features['DSP']) - 1,
            len(allowable_features['FF']) - 1,
            safe_index(allowable_features['possible_region_is_pipelined'], region_is_pipelined_val),
            safe_index(allowable_features['region_ii_bucket'], region_ii_val),
            safe_index(allowable_features['region_pipe_depth_bucket'], region_pipe_depth_val),
        ]
    else:
        # 未知类别：按普通 nodes 逻辑兜底
        node_feature = [
            safe_index(allowable_features['node_category'], 'misc'),
            safe_index(allowable_features['bitwidth'], bitwidth_val),
            safe_index(allowable_features['opcode_category'], opcode_type(opcode_val)),
            safe_index(allowable_features['possible_opcode_list'], opcode_val),
            safe_index(allowable_features['possible_is_start_of_path'], is_start_of_path_val),
            safe_index(allowable_features['possible_is_LCDnode'], is_lcdnode_val),
            safe_index(allowable_features['possible_cluster_group_num'], cluster_group_num_val),
            safe_index(allowable_features['LUT'], lut_val),
            safe_index(allowable_features['DSP'], dsp_val),
            safe_index(allowable_features['FF'], ff_val),
            safe_index(allowable_features['possible_region_is_pipelined'], region_is_pipelined_val),
            safe_index(allowable_features['region_ii_bucket'], region_ii_val),
            safe_index(allowable_features['region_pipe_depth_bucket'], region_pipe_depth_val),
        ]

    return node_feature


def get_node_feature_dims() -> List[int]:
    """获取节点特征维度"""
    return list(map(len, [
        allowable_features['node_category'],
        allowable_features['bitwidth'],
        allowable_features['opcode_category'],
        allowable_features['possible_opcode_list'],
        allowable_features['possible_is_start_of_path'],
        allowable_features['possible_is_LCDnode'],
        allowable_features['possible_cluster_group_num'],
        allowable_features['LUT'],
        allowable_features['DSP'],
        allowable_features['FF'],
        # 新增的流水线特征维度
        allowable_features['possible_region_is_pipelined'],
        allowable_features['region_ii_bucket'],
        allowable_features['region_pipe_depth_bucket'],
    ]))


def edge_to_feature_vector(edge: Dict) -> List[int]:
    """将边转换为特征向量"""
    bond_feature = [
        safe_index(allowable_features['possible_edge_type_list'], int(edge['edge_type'])),
        allowable_features['possible_is_back_edge'].index(int(edge['is_back_edge']))
    ]
    return bond_feature


def get_edge_feature_dims() -> List[int]:
    """获取边特征维度"""
    return list(map(len, [
        allowable_features['possible_edge_type_list'],
        allowable_features['possible_is_back_edge']
    ]))


# ============================================================================
# 图解析函数
# ============================================================================

def parse_xml_into_graph_single(xml_file: str, hierarchical: bool = False, region: bool = False) -> Optional[nx.DiGraph]:
    """解析XML文件为图"""
    prefix = ''
    G = nx.DiGraph()
    parser = et.parse(xml_file)
    root = parser.getroot()
    cdfg = root.findall('*/cdfg')[0]

    rtl_res_table = get_rtl_hash_table(root)

    # 添加边
    edge_id_max = -1
    for edges in cdfg.iter('edges'):
        for edge in edges.iter('item'):
            source = edge.find('source_obj').text
            sink = edge.find('sink_obj').text
            edge_id = edge.find('id').text
            edge_id_max = max(int(edge_id), edge_id_max)
            is_back_edge = edge.find('is_back_edge').text
            edge_type = edge.find('edge_type').text
            G.add_edges_from([(prefix + source, prefix + sink, {
                'edge_name': prefix + edge_id,
                'is_back_edge': is_back_edge,
                'edge_type': edge_type
            })])

    # 添加节点属性
    for nodes in cdfg.iter('nodes'):
        for node in nodes.findall('item'):
            node_id = node.findall('*/*/id')[0].text
            node_name = prefix + node_id

            if node_name not in G.nodes():
                op_code = node.findall('opcode')[0].text
                if op_code == 'ret':
                    G.add_node(node_name)
                    G.nodes[node_name]['node_name'] = node_name
                    G.nodes[node_name]['category'] = 'nodes'
                    G.nodes[node_name]['bitwidth'] = node.findall('*/bitwidth')[0].text
                    G.nodes[node_name]['opcode'] = node.findall('opcode')[0].text
                    G.nodes[node_name]['m_Display'] = node.findall('m_Display')[0].text
                    G.nodes[node_name]['m_isOnCriticalPath'] = node.findall('m_isOnCriticalPath')[0].text
                    G.nodes[node_name]['m_isStartOfPath'] = node.findall('m_isStartOfPath')[0].text
                    G.nodes[node_name]['m_delay'] = node.findall('m_delay')[0].text
                    G.nodes[node_name]['m_topoIndex'] = node.findall('m_topoIndex')[0].text
                    G.nodes[node_name]['m_isLCDNode'] = node.findall('m_isLCDNode')[0].text
                    G.nodes[node_name]['m_clusterGroupNumber'] = node.findall('m_clusterGroupNumber')[0].text
                    G.nodes[node_name]['type'] = node.findall('*/*/type')[0].text
                    G.nodes[node_name]['LUT'] = '0'
                    G.nodes[node_name]['FF'] = '0'
                    G.nodes[node_name]['DSP'] = '0'
                continue

            G.nodes[node_name]['node_name'] = node_name
            G.nodes[node_name]['category'] = 'nodes'
            G.nodes[node_name]['bitwidth'] = node.findall('*/bitwidth')[0].text
            G.nodes[node_name]['opcode'] = node.findall('opcode')[0].text
            G.nodes[node_name]['m_Display'] = node.findall('m_Display')[0].text
            G.nodes[node_name]['m_isOnCriticalPath'] = node.findall('m_isOnCriticalPath')[0].text
            G.nodes[node_name]['m_isStartOfPath'] = node.findall('m_isStartOfPath')[0].text
            G.nodes[node_name]['m_delay'] = node.findall('m_delay')[0].text
            G.nodes[node_name]['m_topoIndex'] = node.findall('m_topoIndex')[0].text
            G.nodes[node_name]['m_isLCDNode'] = node.findall('m_isLCDNode')[0].text
            G.nodes[node_name]['m_clusterGroupNumber'] = node.findall('m_clusterGroupNumber')[0].text
            G.nodes[node_name]['type'] = node.findall('*/*/type')[0].text
            G.nodes[node_name]['LUT'] = '0'
            G.nodes[node_name]['FF'] = '0'
            G.nodes[node_name]['DSP'] = '0'
            t_rtlname = node.findall('*/*/rtlName')[0].text
            if t_rtlname != None:
                if t_rtlname in rtl_res_table.keys():
                    res_name = rtl_res_table[t_rtlname].keys()
                    for i in res_name:
                        G.nodes[node_name][i] = rtl_res_table[t_rtlname][i]

    # 处理blocks
    for nodes in cdfg.iter('blocks'):
        for node in nodes.findall('item'):
            node_id = node.findall('*/id')[0].text
            node_name = prefix + node_id

            if node_name not in G.nodes():
                continue
            G.nodes[node_name]['node_name'] = node_name
            G.nodes[node_name]['category'] = 'blocks'
            G.nodes[node_name]['type'] = node.findall('*/type')[0].text

    # 处理ports
    for nodes in cdfg.iter('ports'):
        for node in nodes.findall('item'):
            node_id = node.findall('*/*/id')[0].text
            node_name = prefix + node_id

            if node_name not in G.nodes():
                continue
            G.nodes[node_name]['node_name'] = node_name
            G.nodes[node_name]['category'] = 'ports'
            G.nodes[node_name]['type'] = node.findall('*/*/type')[0].text
            G.nodes[node_name]['bitwidth'] = node.findall('*/bitwidth')[0].text
            G.nodes[node_name]['direction'] = node.findall('direction')[0].text
            G.nodes[node_name]['if_type'] = node.findall('if_type')[0].text
            G.nodes[node_name]['array_size'] = node.findall('array_size')[0].text

    # 解析并标注流水区域信息（仅为节点添加属性，不改变拓扑）
    try:
        # 默认给所有节点填充0，避免缺失键（仅当 region 开启时对节点进行填充）
        if region:
            for n in G.nodes():
                G.nodes[n]['region_is_pipelined'] = G.nodes[n].get('region_is_pipelined', 0)
                G.nodes[n]['region_ii'] = G.nodes[n].get('region_ii', 0)
                G.nodes[n]['region_pipe_depth'] = G.nodes[n].get('region_pipe_depth', 0)

        pipeline_region_count = 0
        ii_values: List[int] = []
        pipe_depth_values: List[int] = []

        # 收集区域以便分层模式创建节点
        region_records = []  # (region_key, is_pipelined, ii, depth, basic_block_ids)

        # cdfg_regions: 循环/区域级别（优先依据 mII/mDepth/mIsDfPipe）
        for idx, reg in enumerate(root.findall('.//cdfg_regions//item')):
            try:
                mII = int(reg.findtext('mII', default='-1'))
            except Exception:
                mII = -1
            try:
                mDepth = int(reg.findtext('mDepth', default='0'))
            except Exception:
                mDepth = 0
            try:
                mIsDfPipe = int(reg.findtext('mIsDfPipe', default='0'))
            except Exception:
                mIsDfPipe = 0
            is_pipelined = 1 if ((mII is not None and mII > 0) or (mDepth is not None and mDepth > 0) or (mIsDfPipe == 1)) else 0

            # 标注到 basic_blocks 的节点上（仅当 region 开启时）
            bb_ids: List[str] = []
            bb_list = reg.find('basic_blocks')
            if bb_list is not None:
                for bb in bb_list.findall('item'):
                    bb_id = bb.text
                    if bb_id is None:
                        continue
                    bb_ids.append(bb_id)
                    if region:
                        node_name = prefix + bb_id
                        if node_name in G.nodes():
                            G.nodes[node_name]['region_is_pipelined'] = is_pipelined
                            G.nodes[node_name]['region_ii'] = max(0, mII if mII is not None and mII > 0 else 0)
                            G.nodes[node_name]['region_pipe_depth'] = max(0, mDepth if mDepth is not None and mDepth > 0 else 0)

            # 记录区域信息用于层次化
            region_key = reg.findtext('mTag', default=f'region_{idx}') or f'region_{idx}'
            region_records.append((region_key, is_pipelined, max(0, mII if mII and mII > 0 else 0), max(0, mDepth if mDepth and mDepth > 0 else 0), bb_ids))

            if is_pipelined:
                pipeline_region_count += 1
                if mII is not None and mII > 0:
                    ii_values.append(mII)
                if mDepth is not None and mDepth > 0:
                    pipe_depth_values.append(mDepth)

        # 顶层 regions: interval/pipe_depth（仅用于全局旁证）
        top_intervals: List[int] = []
        top_pipe_depths: List[int] = []
        for reg in root.findall('.//regions//item'):
            try:
                iv = int(reg.findtext('interval', default='0'))
            except Exception:
                iv = 0
            try:
                pd = int(reg.findtext('pipe_depth', default='0'))
            except Exception:
                pd = 0
            if iv > 0:
                top_intervals.append(iv)
            if pd > 0:
                top_pipe_depths.append(pd)

        # 资源/信号旁证
        pipeline_components_present = 0
        pipeline_signals_present = 0
        try:
            for it in root.findall('.//res/dp_component_resource//item/first'):
                txt = (it.text or '').lower()
                if 'flow_control_loop_pipe' in txt:
                    pipeline_components_present = 1
                    break
        except Exception:
            pass
        try:
            # 迭代寄存器/多路复用相关
            for it in root.findall('.//res/dp_register_resource//item/first'):
                if it is not None and it.text and 'ap_enable_reg_pp' in it.text:
                    pipeline_signals_present = 1
                    break
            if pipeline_signals_present == 0:
                for it in root.findall('.//res/dp_multiplexer_resource//item/first'):
                    if it is not None and it.text and 'ap_enable_reg_pp' in it.text:
                        pipeline_signals_present = 1
                        break
        except Exception:
            pass

        # 计算全局指标
        avg_ii = float(sum(ii_values) / len(ii_values)) if ii_values else 0.0
        max_pipe_depth = int(max(pipe_depth_values) if pipe_depth_values else (max(top_pipe_depths) if top_pipe_depths else 0))
        has_pipeline = 1 if (pipeline_region_count > 0 or len(top_intervals) > 0 or max_pipe_depth > 0 or pipeline_components_present == 1 or pipeline_signals_present == 1) else 0

        # 写入到图的全局属性
        G.graph['has_pipeline'] = has_pipeline
        G.graph['pipeline_region_count'] = pipeline_region_count
        G.graph['avg_ii'] = avg_ii
        G.graph['max_pipe_depth'] = max_pipe_depth
        G.graph['pipeline_components_present'] = pipeline_components_present
        G.graph['pipeline_signals_present'] = pipeline_signals_present

        # 分层模式：增加区域节点与包含关系边（独立于 region 标志）
        if hierarchical and region_records:
            for idx, (rk, is_p, ii, depth, bb_ids) in enumerate(region_records):
                region_node_name = f"region::{rk}::{idx}"
                if region_node_name not in G:
                    G.add_node(region_node_name)
                G.nodes[region_node_name]['node_name'] = region_node_name
                G.nodes[region_node_name]['category'] = 'regions'
                G.nodes[region_node_name]['type'] = 'region'
                G.nodes[region_node_name]['bitwidth'] = '0'
                G.nodes[region_node_name]['opcode'] = 'misc'
                G.nodes[region_node_name]['m_isOnCriticalPath'] = '0'
                G.nodes[region_node_name]['m_isStartOfPath'] = '0'
                G.nodes[region_node_name]['m_isLCDNode'] = '0'
                G.nodes[region_node_name]['m_clusterGroupNumber'] = '-1'
                G.nodes[region_node_name]['LUT'] = '0'
                G.nodes[region_node_name]['FF'] = '0'
                G.nodes[region_node_name]['DSP'] = '0'
                # 区域节点的流水属性
                G.nodes[region_node_name]['region_is_pipelined'] = is_p
                G.nodes[region_node_name]['region_ii'] = ii
                G.nodes[region_node_name]['region_pipe_depth'] = depth

                # 边：region_contains（region -> basic_block）
                for bb_id in bb_ids:
                    bb_node = prefix + bb_id
                    if bb_node in G:
                        G.add_edge(region_node_name, bb_node, edge_name=f"rcontains_{idx}_{bb_id}", is_back_edge='0', edge_type='4')
    except Exception:
        # 静默失败，不影响原有流程
        pass

    # 移除常量节点
    for nodes in cdfg.iter('consts'):
        for node in nodes.findall('item'):
            node_id = node.findall('*/*/id')[0].text
            node_name = prefix + node_id
            if node_name in G.nodes():
                G.remove_node(node_name)

    return G