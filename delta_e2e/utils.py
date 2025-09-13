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
    'node_category': ['nodes', 'blocks', 'ports', 'misc'],
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
        'misc'
    ],
    'possible_is_start_of_path': [0, 1, 'misc'],
    'possible_is_LCDnode': [0, 1, 'misc'],
    'possible_cluster_group_num': [-1] + list(range(0, 256)) + ['misc'],
    'LUT': list(range(0, 1000)) + ['misc'],
    'DSP': list(range(0, 11)) + ['misc'],
    'FF': list(range(0, 1000)) + ['misc'],
    'possible_edge_type_list': [1, 2, 3, 'misc'],
    'possible_is_back_edge': [0, 1],
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
    """获取操作码类型"""
    t = 'misc'
    if opcode in {'br', 'ret', 'switch'}:
        t = 'terminator'
    elif opcode in {'add', 'dadd', 'fadd', 'sub', 'dsub', 'fsub', 'mul', 'dmul', 'fmul', 'udiv', 'ddiv', 'fdiv', 'sdiv', 'urem', 'srem', 'frem', 'dexp', 'dsqrt'}:
        t = 'binary_unary'
    elif opcode in {'shl', 'lshr', 'ashr', 'and', 'xor', 'or'}:
        t = 'bitwise'
    elif opcode in {'uitofp', 'sitofp', 'uitodp', 'sitodp', 'bitconcatenate', 'bitcast', 'zext', 'sext', 'fpext', 'trunc', 'fptrunc'}:
        t = 'conversion'
    elif opcode in {'alloca', 'load', 'store', 'read', 'write', 'getelementptr'}:
        t = 'memory'
    elif opcode in {'extractvalue', 'insertvalue'}:
        t = 'aggregate'
    elif opcode in {'phi', 'call', 'icmp', 'dcmp', 'fcmp', 'select', 'bitselect', 'partselect', 'mux', 'dacc', 'sparsemux'}:
        t = 'other'
    if t == 'misc':
        print(f"opcode: {opcode}")
    return t


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
    if node == {}:
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
            len(allowable_features['FF']) - 1
        ]
        return node_feature

    if node['category'] == 'nodes':
        node_feature = [
            safe_index(allowable_features['node_category'], node['category']),
            safe_index(allowable_features['bitwidth'], int(node['bitwidth'])),
            safe_index(allowable_features['opcode_category'], opcode_type(node['opcode'])),
            safe_index(allowable_features['possible_opcode_list'], node['opcode']),
            safe_index(allowable_features['possible_is_start_of_path'], int(node['m_isStartOfPath'])),
            safe_index(allowable_features['possible_is_LCDnode'], int(node['m_isLCDNode'])),
            safe_index(allowable_features['possible_cluster_group_num'], int(node['m_clusterGroupNumber'])),
            safe_index(allowable_features['LUT'], int(node['LUT'])),
            safe_index(allowable_features['DSP'], int(node['DSP'])),
            safe_index(allowable_features['FF'], int(node['FF']))
        ]
    elif node['category'] == 'ports':
        node_feature = [
            safe_index(allowable_features['node_category'], node['category']),
            safe_index(allowable_features['bitwidth'], int(node['bitwidth'])),
            len(allowable_features['opcode_category']) - 1,
            len(allowable_features['possible_opcode_list']) - 1,
            len(allowable_features['possible_is_start_of_path']) - 1,
            len(allowable_features['possible_is_LCDnode']) - 1,
            len(allowable_features['possible_cluster_group_num']) - 1,
            len(allowable_features['LUT']) - 1,
            len(allowable_features['DSP']) - 1,
            len(allowable_features['FF']) - 1
        ]
    elif node['category'] == 'blocks':
        node_feature = [
            safe_index(allowable_features['node_category'], node['category']),
            len(allowable_features['bitwidth']) - 1,
            len(allowable_features['opcode_category']) - 1,
            len(allowable_features['possible_opcode_list']) - 1,
            len(allowable_features['possible_is_start_of_path']) - 1,
            len(allowable_features['possible_is_LCDnode']) - 1,
            len(allowable_features['possible_cluster_group_num']) - 1,
            len(allowable_features['LUT']) - 1,
            len(allowable_features['DSP']) - 1,
            len(allowable_features['FF']) - 1
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

def parse_xml_into_graph_single(xml_file: str) -> Optional[nx.DiGraph]:
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

    # 移除常量节点
    for nodes in cdfg.iter('consts'):
        for node in nodes.findall('item'):
            node_id = node.findall('*/*/id')[0].text
            node_name = prefix + node_id
            if node_name in G.nodes():
                G.remove_node(node_name)

    return G