#!/usr/bin/env python3
"""
ForgeHLS数据预处理脚本
将ForgeHLS数据集处理为GNN可用的格式

输入:
- forgehls_dataset_names: 要处理的数据集名称列表（写死）
- designs_base_dir: 设计文件根目录（可变参数）

输出:
- dataset_save_dir: 最终数据集保存目录
"""

import xml.etree.cElementTree as et
import networkx as nx
import matplotlib.pyplot as plt
import json
import subprocess
import time
from os import path
import glob
import pandas as pd
import re
import os
import shutil
from tqdm import tqdm
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from multiprocessing import Pool
from functools import partial
from sklearn import model_selection
import gzip
import concurrent.futures

# 导入dataset_csv模块的函数
from multiprocessing.util import is_abstract_socket_namespace
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
import tiktoken


def gather_csynth_data(root_dir, output_csv):
    """从csynth.xml文件收集数据"""
    print("Gathering csynth data...")
    root_dir = Path(root_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    csynth_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file == 'csynth.xml':
                file_path = os.path.join(root, file)
                csynth_files.append(file_path)

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for result in executor.map(process_csynth_file, csynth_files):
            if result:
                results.append(result)
    
    with open(output_csv, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['File Path', 'Part', 'TargetClockPeriod', 'Best-caseLatency', 'Worst-caseLatency', 'BRAM_18K', 'LUT', 'DSP', 'FF', 'Avialable_BRAM_18K', 'Avialable_LUT', 'Avialable_DSP', 'Avialable_FF'])
        csv_writer.writerows(results)
    
    print(f"Gathering raw data done. Saved in {output_csv}")


def process_csynth_file(file_path):
    """处理单个csynth.xml文件"""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        part = root.find('.//UserAssignments/Part').text
        target_clock_period = root.find('.//UserAssignments/TargetClockPeriod').text
        best_case_latency = root.find('.//PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency').text
        worst_case_latency = root.find('.//PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency').text
        bram_18k = root.find('.//AreaEstimates/Resources/BRAM_18K').text
        lut = root.find('.//AreaEstimates/Resources/LUT').text
        dsp = root.find('.//AreaEstimates/Resources/DSP').text
        ff = root.find('.//AreaEstimates/Resources/FF').text
        available_bram_18k = root.find('.//AreaEstimates/AvailableResources/BRAM_18K').text
        available_lut = root.find('.//AreaEstimates/AvailableResources/LUT').text
        available_dsp = root.find('.//AreaEstimates/AvailableResources/DSP').text
        available_ff = root.find('.//AreaEstimates/AvailableResources/FF').text

        return [file_path, part, target_clock_period, best_case_latency, worst_case_latency, 
                bram_18k, lut, dsp, ff, available_bram_18k, available_lut, available_dsp, available_ff]
    except (ET.ParseError, AttributeError) as e:
        print(f"Error processing {file_path}: {str(e)}")
    return None


def print_info(df):
    """打印数据集信息"""
    print(f"kernels number: {df['algo_name'].nunique() if 'algo_name' in df.columns else 0} designs number: {df.shape[0]}")


def base_feature(df):
    """添加基础特征"""
    if df.empty:
        print("DataFrame为空，无法添加特征")
        return df
        
    # ResourceMetric - 使用向量化操作代替apply
    df['BRAM_18K'] = pd.to_numeric(df['BRAM_18K'], errors='coerce')
    df['LUT'] = pd.to_numeric(df['LUT'], errors='coerce')
    df['DSP'] = pd.to_numeric(df['DSP'], errors='coerce')
    df['FF'] = pd.to_numeric(df['FF'], errors='coerce')
    df['Avialable_BRAM_18K'] = pd.to_numeric(df['Avialable_BRAM_18K'], errors='coerce')
    df['Avialable_LUT'] = pd.to_numeric(df['Avialable_LUT'], errors='coerce')
    df['Avialable_DSP'] = pd.to_numeric(df['Avialable_DSP'], errors='coerce')
    df['Avialable_FF'] = pd.to_numeric(df['Avialable_FF'], errors='coerce')
    
    bram_ratio = df['BRAM_18K'] / df['Avialable_BRAM_18K'].replace(0, float('inf'))
    lut_ratio = df['LUT'] / df['Avialable_LUT'].replace(0, float('inf'))
    dsp_ratio = df['DSP'] / df['Avialable_DSP'].replace(0, float('inf'))
    ff_ratio = df['FF'] / df['Avialable_FF'].replace(0, float('inf'))
    
    # 处理无穷大值
    bram_ratio = bram_ratio.replace(float('inf'), 0)
    lut_ratio = lut_ratio.replace(float('inf'), 0)
    dsp_ratio = dsp_ratio.replace(float('inf'), 0)
    ff_ratio = ff_ratio.replace(float('inf'), 0)
    
    df['ResourceMetric'] = (bram_ratio + lut_ratio + dsp_ratio + ff_ratio) / 4

    # 提取路径信息
    df['File Path'] = df['File Path'].astype(str)
    path_parts = df['File Path'].str.split('/')
    max_parts = path_parts.str.len().max()

    df['is_kernel'] = 'unknown'
    df.loc[df['File Path'].str.contains('kernels'), 'is_kernel'] = True
    df.loc[df['File Path'].str.contains('design_'), 'is_kernel'] = False

    df['design_id'] = 'unknown'
    df['source_name'] = 'unknown'
    df['algo_name'] = 'unknown'
    df['solution_name'] = 'unknown'
    df['project_name'] = 'unknown'

    kernel_rows = df['is_kernel'] == True
    if kernel_rows.any() and max_parts >= 7:
        try:
            df.loc[kernel_rows, 'source_name'] = path_parts.loc[kernel_rows].str[-7]
            df.loc[kernel_rows, 'algo_name'] = path_parts.loc[kernel_rows].str[-6]
            df.loc[kernel_rows, 'project_name'] = path_parts.loc[kernel_rows].str[-5]
            df.loc[kernel_rows, 'solution_name'] = path_parts.loc[kernel_rows].str[-4]
        except IndexError:
            print("Some paths are too short to extract 'source_name', 'algo_name', 'project_name', 'solution_name'.")

    design_rows = df['is_kernel'] == False
    if design_rows.any() and max_parts >= 8:
        try:
            #algo_name -7
            df.loc[design_rows, 'source_name'] = path_parts.loc[design_rows].str[-8]
            df.loc[design_rows, 'algo_name'] = path_parts.loc[design_rows].str[-7]
            df.loc[design_rows, 'design_id'] = path_parts.loc[design_rows].str[-6]
            df.loc[design_rows, 'project_name'] = path_parts.loc[design_rows].str[-5]
            df.loc[design_rows, 'solution_name'] = path_parts.loc[design_rows].str[-4]
            
        except IndexError:
            print("Some paths are too short to extract 'design_id'.")
            df.loc[design_rows, 'design_id'] = 'unknown'

    df = df.reset_index(drop=True)
    print_info(df)
    return df


def delete_undef(df):
    """删除undefined值"""
    if df.empty:
        print("DataFrame为空，无法处理undefined值")
        return df
        
    df['Best-caseLatency'] = df['Best-caseLatency'].replace('undef', pd.NA)
    
    valid_groups = []
    for algo_name, group in df.groupby('algo_name'):
        if not group['Best-caseLatency'].isna().any():
            valid_groups.append(group)
    
    if not valid_groups:
        print("没有找到有效的设计（所有设计的Best-caseLatency都包含undef）")
        return pd.DataFrame(columns=df.columns)
    
    df = pd.concat(valid_groups)
    df = df.reset_index(drop=True)
    
    df['Best-caseLatency'] = pd.to_numeric(df['Best-caseLatency'], errors='coerce')
    df['Worst-caseLatency'] = pd.to_numeric(df['Worst-caseLatency'], errors='coerce')

    print(f"After deleting 'Best-caseLatency' = 'undef' cases")
    print_info(df)

    # 处理零延迟情况
    grouped = df.groupby('algo_name')
    results = []
    for algo_name, group in grouped:
        group = group.sort_values(by=['Best-caseLatency', 'ResourceMetric'])
        group = group.reset_index(drop=True)
        non_zero_latency_group = group[group['Best-caseLatency'] != 0]
        
        if len(non_zero_latency_group) > 0:
            results.append(non_zero_latency_group)
        else:
            results.append(group)
    
    if results:
        df = pd.concat(results, ignore_index=True)
    else:
        df = pd.DataFrame(columns=df.columns)
    
    print(f"After deleting zero latency cases")
    print_info(df)
    
    return df


def delete_overlap_and_overfitting(df):
    """删除重叠和过拟合数据"""
    if df.empty:
        print("DataFrame为空，无法处理重叠和过拟合")
        return df
        
    df = df.drop_duplicates(subset=['algo_name', 'Best-caseLatency', 'ResourceMetric'], keep='first')
    print(f"After deleting duplicate 'Best-caseLatency' and 'ResourceMetric' combinations")
    print_info(df)

    def remove_invalid_designs_optimized(group):
        group = group.sort_values(by=['ResourceMetric', 'Best-caseLatency'])
        keep_mask = pd.Series(True, index=group.index)
        
        for i in range(len(group)):
            if not keep_mask.iloc[i]:
                continue
                
            resource_i = group['ResourceMetric'].iloc[i]
            latency_i = group['Best-caseLatency'].iloc[i]
            
            for j in range(i+1, len(group)):
                if not keep_mask.iloc[j]:
                    continue
                    
                resource_j = group['ResourceMetric'].iloc[j]
                latency_j = group['Best-caseLatency'].iloc[j]
                
                if resource_i == resource_j and latency_i == latency_j:
                    keep_mask.iloc[j] = False
        
        return group[keep_mask]

    results = []
    for algo_name, group in df.groupby('algo_name'):
        optimized_group = remove_invalid_designs_optimized(group)
        results.append(optimized_group)
    
    if results:
        df = pd.concat(results, ignore_index=True)
    else:
        df = pd.DataFrame(columns=df.columns)

    print(f"After deleting overlapping cases")
    print_info(df)
    return df


def add_is_pareto(df):
    """添加Pareto前沿标记"""
    if df.empty:
        print("DataFrame为空，无法添加Pareto信息")
        return df
        
    df['is_pareto'] = False
    
    def process_algo_group(group_data):
        group = group_data.copy()
        goal1 = group['ResourceMetric'].values
        goal2 = group['Best-caseLatency'].values
        is_pareto = [True] * len(goal1)
        
        for i in range(len(goal1)):
            for j in range(len(goal1)):
                if i != j:
                    if (goal1[j] <= goal1[i] and goal2[j] <= goal2[i]) and (goal1[j] < goal1[i] or goal2[j] < goal2[i]):
                        is_pareto[i] = False
                        break
        
        group['is_pareto'] = is_pareto
        return group
    
    processed_groups = []
    for algo_name, group in df.groupby('algo_name'):
        processed_group = process_algo_group(group)
        processed_groups.append(processed_group)
    
    df = pd.concat(processed_groups, ignore_index=True)
    print(f"Successfully added 'is_pareto' column")
    return df


def embed_source_code(df):
    """嵌入源代码信息"""
    if df.empty:
        print("DataFrame为空，无法嵌入源代码")
        return df
        
    source_code_cache = {}
    
    def get_source_code(file_path):
        if file_path in source_code_cache:
            return source_code_cache[file_path]
            
        base_dir = Path(file_path).parents[4]
        source_list = []
        
        for root, dirs, files in os.walk(base_dir):
            if '.autopilot' in dirs:
                dirs.remove('.autopilot')
            for fname in files:
                if fname.endswith((".c", ".cpp", ".h", ".hpp")):
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, "r") as sf:
                            content = sf.read()
                        source_list.append({
                            "file_name": fname,
                            "file_content": content
                        })
                    except Exception as e:
                        print(f"Error reading {fpath}: {e}")
        
        source_code_cache[file_path] = source_list
        return source_list
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        source_codes = list(executor.map(get_source_code, df['File Path']))
    
    df['source_code'] = source_codes
    df['code_length'] = [sum([len(file['file_content']) for file in sc if file['file_name'].endswith((".c", ".cpp"))]) for sc in source_codes]
   
    def count_tokens(code):
        if pd.isna(code):
            return 0
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(code))
    df['token_count'] = df['source_code'].apply(lambda x: sum([count_tokens(file['file_content']) for file in x if file['file_name'].endswith((".c", ".cpp"))]))

    pragma_pattern = re.compile(r'#pragma.*')
    
    def get_pragma_number(source_code):
        pragma_number = 0
        for file in source_code:
            content = file['file_content']
            pragma_number += len(pragma_pattern.findall(content))
        return pragma_number
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        pragma_numbers = list(executor.map(get_pragma_number, df['source_code']))
    
    df['pragma_number'] = pragma_numbers
    print(f"Successfully embedded source code")
    return df


def analysis(df, save_path, dataset_name):
    """生成分析数据"""
    if df.empty:
        print("DataFrame为空，无法生成分析")
        analysis_df = pd.DataFrame(columns=['algo_name', 'source_name', 'pragma_number', 'design_number', 'code_length', 'token_count'])
    else:
        analysis_df = df.groupby('algo_name').agg({
            'source_name': 'first',
            'pragma_number': 'first',
            'code_length': 'first',
            'token_count': 'first'
        }).reset_index()
        
        analysis_df['design_number'] = df.groupby('algo_name').size().values
        analysis_df = analysis_df.sort_values('algo_name')
    
    analysis_csv_path = os.path.join(save_path, f'analysis_of_designs_{dataset_name}.csv')
    analysis_df.to_csv(analysis_csv_path, index=False)
    print(f"Successfully saved 'analysis_of_designs' to {analysis_csv_path}")


def add_top_function_name(df):
    """添加顶层函数名"""
    if df.empty:
        print("DataFrame为空，无法添加顶层函数名")
        return df
        
    top_function_cache = {}
    
    def get_top_function_name(file_path):
        if file_path in top_function_cache:
            return top_function_cache[file_path]
            
        try:
            xml_dir = os.path.dirname(file_path)
            top_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(xml_dir))))
            top_function_name_path = os.path.join(top_dir, 'top_function_name.txt')
            
            if os.path.exists(top_function_name_path):
                with open(top_function_name_path, 'r') as f:
                    top_function_name = f.read().strip()
                    top_function_cache[file_path] = top_function_name
                    return top_function_name
            else:
                print(f"找不到顶层函数名文件: {top_function_name_path}")
                return "unknown"
        except Exception as e:
            print(f"获取顶层函数名时出错: {file_path}, 错误: {str(e)}")
            return "unknown"
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        top_functions = list(executor.map(get_top_function_name, df['File Path']))
    
    df['top_function_name'] = top_functions
    return df


def dataset_csv(search_path, save_path):
    """处理数据集CSV"""
    os.makedirs(save_path, exist_ok=True)
    
    dataset_name = os.path.basename(search_path.rstrip('/'))
    print(f"dataset_name: {dataset_name}")
    raw_data_csv_path = os.path.join(save_path, f'raw_data_of_designs_{dataset_name}.csv')
    
    if not os.path.exists(search_path):
        print(f"搜索路径不存在: {search_path}")
        with open(raw_data_csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['File Path', 'Part', 'TargetClockPeriod', 'Best-caseLatency', 'Worst-caseLatency', 
                                   'BRAM_18K', 'LUT', 'DSP', 'FF', 'Avialable_BRAM_18K', 'Avialable_LUT', 'Avialable_DSP', 'Avialable_FF'])
        print(f"已创建空的原始数据CSV: {raw_data_csv_path}")
        return
    
    gather_csynth_data(search_path, raw_data_csv_path)

    print(f"read csv from {raw_data_csv_path}")
    df = pd.read_csv(raw_data_csv_path)
    
    if df.empty:
        print("读取的CSV文件为空，将跳过后续处理")
        return
    
    df = add_top_function_name(df)
    df = base_feature(df)
    df = embed_source_code(df)
    analysis(df, save_path, dataset_name)

    df = delete_undef(df)
    if not df.empty:
        df = delete_overlap_and_overfitting(df)
        df = add_is_pareto(df)

        output_json_path = os.path.join(save_path, f'data_of_designs_{dataset_name}.json')
        df.to_json(output_json_path, orient='records', force_ascii=False, indent=4)
        print(f"The JSON file is saved in {output_json_path}")
    else:
        print("数据处理后为空，不生成最终JSON文件")


def get_rtl_hash_table(root):
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


def parse_xml_into_graph_single(xml_file):
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
            G.add_edges_from([(prefix + source, prefix + sink, {'edge_name': prefix + edge_id, 'is_back_edge': is_back_edge, 'edge_type': edge_type})])

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
                    G.nodes[node_name]['category']='nodes'
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

            if node_name not in G.nodes():
                continue
            for v in G.neighbors(node_name):
                G.nodes[v]['const'] = node_name
                G.nodes[v]['const-bitwidth'] = node.findall('*/bitwidth')[0].text
            G.remove_node(node_name)

    return G


def json_save(G, fname):
    """保存单个图为JSON"""
    f = open(fname + '.json', 'w')
    G_dict = dict(nodes=[[n, G.nodes[n]] for n in G.nodes()], \
                edges=[(e[0], e[1], G.edges[e]) for e in G.edges()])
    json.dump(G_dict, f)
    f.close()


def json_save_graphs(Gs, fname):
    """保存多个图为JSON"""
    f = open(fname + '.json', 'w')
    G_dict = dict()
    G_dict['nodes'] = []
    G_dict['edges'] = []
    for G in Gs:
        for n in G.nodes():
            G_dict['nodes'].append([n, G.nodes[n]])
        for e in G.edges():
            G_dict['edges'].append((e[0], e[1], G.edges[e]))
    json.dump(G_dict, f)
    f.close()


def get_real_perf(fname):
    """获取真实性能数据"""
    f = open(fname + '.json', 'r')
    d = json.load(f)
    f.close()
    DSP=d['DSP']
    LUT=d['LUT']
    FF=d['FF']
    return DSP, LUT, FF


def find_cpp_c_files(search_dir, max_depth=5):
    """查找C/C++文件"""
    cpp_c_files = []
    
    all_dirs = []
    for root, dirs, _ in os.walk(search_dir):
        if '.autopilot' in dirs:
            dirs.remove('.autopilot')
        rel_path = os.path.relpath(root, search_dir)
        depth = 0 if rel_path == "." else rel_path.count(os.sep) + 1
        
        if depth < max_depth:
            all_dirs.append(root)
    
    for root in tqdm(all_dirs, desc="搜索目录"):
        cpp_c_files.extend([os.path.join(root, file) for file in os.listdir(root) 
                        if file.endswith(('.cpp', '.c'))])
    
    return cpp_c_files


def process_single_file(c_file, designs_dir):
    """处理单个C/C++文件并生成对应的.adb文件"""
    try:
        abs_path = os.path.abspath(c_file)
        path_parts = abs_path.split(os.sep)
        if len(path_parts) < 4:
            return []
            
        file_name = os.path.splitext(path_parts[-1])[0]
        design_id = path_parts[-2]
        kernel_name = path_parts[-3]
        source_name = path_parts[-4]
        
        prefix = f"{source_name}-{kernel_name}-{design_id}-"
        adb_dir = os.path.dirname(c_file)
        
        adb_files = []
        for root, dirs, files in os.walk(adb_dir):
            for file in files:
                if file.endswith('.adb') and len(file.split('.')) == 2:
                    adb_files.append(os.path.join(root, file))
        
        adb_files = list(set(adb_files))

        target_dir = os.path.join('real_case', source_name + '_adb')
        os.makedirs(target_dir, exist_ok=True)
        
        for adb_file in adb_files:
            original_adb_name = os.path.basename(adb_file)
            new_adb_name = f"{prefix}{original_adb_name}"
            target_path = os.path.join(target_dir, new_adb_name)
            shutil.copy2(adb_file, target_path)
            
        return []
    except Exception as e:
        return [f"处理 {c_file} 时出错: {str(e)}"]


def process_c_files_to_adb(designs_dir, c_files):
    """处理C/C++文件并生成对应的.adb文件"""
    os.makedirs('real_case', exist_ok=True)
    
    cpu_count = multiprocessing.cpu_count()
    workers = min(cpu_count * 8, 256)
    
    pbar = tqdm(total=len(c_files), desc="处理文件")
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_single_file, c_file, designs_dir) for c_file in c_files]
        
        for future in as_completed(futures):
            try:
                results = future.result()
                if results:
                    print(f"处理文件出现 {len(results)} 个错误")
            except Exception as e:
                print(f"处理文件时发生异常: {str(e)}")
            finally:
                pbar.update(1)
    
    pbar.close()


def check_max_node_id(node_string):
    """检查最大节点ID"""
    node_array=[]
    for n in node_string:
        node_array.append(int(n))
    max_id=max(node_array)
    return max_id


def process_single_graph_file(adb_file, max_id=0):
    """处理单个ADB文件"""
    try:
        g = parse_xml_into_graph_single(adb_file)
        if max_id > 0:
            mapping = {n:str(int(n)+max_id) for n in g.nodes}
            g = nx.relabel_nodes(g, mapping)
        return g
    except Exception as e:
        print(f"处理文件 {adb_file} 时出错: {str(e)}")
        return None


def process_fname(fname, graph_dir, result_dir, graph_mapping):
    """处理单个文件名对应的图"""
    try:
        graph_num = len(graph_mapping[fname])
        print(f"Processing {fname} with {graph_num} graphs")
        adb_files = glob.glob('real_case/' + graph_dir + fname + '-*')
        
        if graph_num > 1:
            max_id = 0
            G = []
            for adb_file in adb_files:
                g = process_single_graph_file(adb_file, max_id)
                if g is not None:
                    G.append(g)
                    max_id = check_max_node_id(g.nodes) + 1
            if G:
                json_save_graphs(G, 'real_case/' + result_dir + '/' + fname)
        else:
            if adb_files:
                g = process_single_graph_file(adb_files[0])
                if g is not None:
                    json_save(g, 'real_case/' + result_dir + '/' + fname)
        return fname
    except Exception as e:
        print(f"处理 {fname} 时出错: {str(e)}")
        return None


# 特征定义
allowable_features = {
    'node_category' : ['nodes', 'blocks', 'ports', 'misc'], 
    'bitwidth' : list(range(0, 256)) + ['misc'], 
    'opcode_category' : ['terminator','binary_unary', 'bitwise', 'conversion','memory','aggregate','other','misc'], 
    'possible_opcode_list' : [
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
    'possible_is_LCDnode':[0, 1, 'misc'],
    'possible_cluster_group_num': [-1] + list(range(0, 256)) + ['misc'],
    'LUT': list(range(0, 1000)) + ['misc'],
    'DSP': list(range(0, 11)) + ['misc'],
    'FF': list(range(0, 1000)) + ['misc'],
    
    'possible_edge_type_list' : [1, 2, 3, 'misc'], 
    'possible_is_back_edge': [0, 1],
}


def safe_index(l, e):
    """安全获取元素索引"""
    try:
        return l.index(e)
    except:
        return len(l) - 1


def opcode_type(opcode):
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


def node_to_feature_vector(node):
    """将节点转换为特征向量"""
    if node=={}:
        node_feature = [
                len(allowable_features['node_category'])-1,
                len(allowable_features['bitwidth'])-1,
                len(allowable_features['opcode_category'])-1,
                len(allowable_features['possible_opcode_list'])-1,
                len(allowable_features['possible_is_start_of_path'])-1,
                len(allowable_features['possible_is_LCDnode'])-1,
                len(allowable_features['possible_cluster_group_num'])-1,
                len(allowable_features['LUT'])-1,
                len(allowable_features['DSP'])-1,
                len(allowable_features['FF'])-1
                ]
        return node_feature
        
    if node['category']=='nodes':
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
    elif node['category']=='ports':
        node_feature = [
                safe_index(allowable_features['node_category'], node['category']),
                safe_index(allowable_features['bitwidth'], int(node['bitwidth'])),
                len(allowable_features['opcode_category'])-1,
                len(allowable_features['possible_opcode_list'])-1,
                len(allowable_features['possible_is_start_of_path'])-1,
                len(allowable_features['possible_is_LCDnode'])-1,
                len(allowable_features['possible_cluster_group_num'])-1,
                len(allowable_features['LUT'])-1,
                len(allowable_features['DSP'])-1,
                len(allowable_features['FF'])-1
                ]
    elif node['category']=='blocks':
        node_feature = [
                safe_index(allowable_features['node_category'], node['category']),
                len(allowable_features['bitwidth'])-1,
                len(allowable_features['opcode_category'])-1,
                len(allowable_features['possible_opcode_list'])-1,
                len(allowable_features['possible_is_start_of_path'])-1,
                len(allowable_features['possible_is_LCDnode'])-1,
                len(allowable_features['possible_cluster_group_num'])-1,
                len(allowable_features['LUT'])-1,
                len(allowable_features['DSP'])-1,
                len(allowable_features['FF'])-1
                ]
    return node_feature


def get_node_feature_dims():
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


def edge_to_feature_vector(edge):
    """将边转换为特征向量"""
    bond_feature = [
                safe_index(allowable_features['possible_edge_type_list'], int(edge['edge_type'])),
                allowable_features['possible_is_back_edge'].index(int(edge['is_back_edge']))
            ]
    return bond_feature


def get_edge_feature_dims():
    """获取边特征维度"""
    return list(map(len, [
        allowable_features['possible_edge_type_list'],
        allowable_features['possible_is_back_edge']
        ]))


def create_dataset_structure(metric, dataset_name, save_dir):
    """创建数据集目录结构并压缩文件"""
    dataset_dir = f"./dataset/cdfg_{metric}_all_numerical_{dataset_name}/"

    os.makedirs(f"{dataset_dir}mapping", exist_ok=False)
    os.makedirs(f"{dataset_dir}split/scaffold", exist_ok=False)
    os.makedirs(f"{dataset_dir}raw", exist_ok=False)
    
    def compress_file(src_path, dst_path):
        with open(src_path, 'rb') as f_in:
            with gzip.open(dst_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return dst_path
    
    compression_tasks = [
        (f"{save_dir}mapping.csv", f"{dataset_dir}mapping/mapping.csv.gz"),
        (f"{save_dir}train.csv", f"{dataset_dir}split/scaffold/train.csv.gz"),
        (f"{save_dir}valid.csv", f"{dataset_dir}split/scaffold/valid.csv.gz"),
        (f"{save_dir}test.csv", f"{dataset_dir}split/scaffold/test.csv.gz"),
        (f"{save_dir}node-feat.csv", f"{dataset_dir}raw/node-feat.csv.gz"),
        (f"{save_dir}edge.csv", f"{dataset_dir}raw/edge.csv.gz"),
        (f"{save_dir}edge-feat.csv", f"{dataset_dir}raw/edge-feat.csv.gz"),
        (f"{save_dir}num-node-list.csv", f"{dataset_dir}raw/num-node-list.csv.gz"),
        (f"{save_dir}num-edge-list.csv", f"{dataset_dir}raw/num-edge-list.csv.gz"),
        (f"{save_dir}graph-label-{metric}.csv", f"{dataset_dir}raw/graph-label.csv.gz")
    ]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(compress_file, src, dst) for src, dst in compression_tasks]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                completed_file = future.result()
                print(f"已完成: {completed_file}")
            except Exception as e:
                print(f"压缩文件时出错: {str(e)}")
    
    print(f"数据集 cdfg_{metric}_all_numerical_{dataset_name} 创建完成")


def pre_process_forgehls(designs_base_dir, dataset_save_dir="./dataset/"):
    """
    ForgeHLS数据预处理主函数
    
    Args:
        designs_base_dir: 设计文件根目录
        dataset_save_dir: 数据集保存目录
    """
    
    # 固定的数据集名称列表
    forgehls_dataset_names = ['PolyBench', 'CHStone', 'MachSuite', "rosetta", 
                              "rtl_module", "rtl_ip", "rtl_chip", 
                            #   "Vitis-HLS-Introductory-Examples-flatten", 
                              "operators", "leetcode_hls_algorithms", "hls_algorithms"]
    
    print(f"开始处理ForgeHLS数据集...")
    print(f"设计文件根目录: {designs_base_dir}")
    print(f"数据集保存目录: {dataset_save_dir}")
    print(f"要处理的数据集: {forgehls_dataset_names}")
    
    # ==================== STEP 1: 处理每个数据集 ====================
    
    for dataset_name in forgehls_dataset_names:
        print(f"\n\n========== 开始处理数据集: {dataset_name} ==========\n")
        designs_dir = f'{designs_base_dir}/{dataset_name}'

        # 调用dataset_csv处理数据
        dataset_csv(designs_dir, "./csv")
        data_of_designs_json_path = f'./csv/data_of_designs_{dataset_name}.json'

        # 清理并创建目录
        for suffix in ['', '_adb', '_ds']:
            dir_path = f'./real_case/{dataset_name}{suffix}'
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            os.makedirs(dir_path)

        # 处理C/C++文件到ADB文件
        print("开始搜索C/C++文件...")
        c_files = find_cpp_c_files(designs_dir)
        print(f"找到 {len(c_files)} 个C/C++文件，开始处理...")

        # 统计设计数量
        design_count = {}
        for c_file in c_files:
            try:
                path_parts = os.path.abspath(c_file).split(os.sep)
                if len(path_parts) >= 4:
                    source_name = path_parts[-4]
                    kernel_name = path_parts[-3]
                    design_id = path_parts[-2]
                    key = f"{source_name}-{kernel_name}"
                    if key not in design_count:
                        design_count[key] = set()
                    design_count[key].add(design_id)
            except Exception as e:
                print(f"统计设计时出错: {str(e)}")

        print("\n设计数量统计:")
        total_designs = 0
        for key, designs in design_count.items():
            design_num = len(designs)
            total_designs += design_num
            print(f"  {key}: {design_num}个设计")
        print(f"总计: {total_designs}个设计\n")

        # 创建必要的目录
        source_names = set()
        for c_file in c_files:
            try:
                path_parts = os.path.abspath(c_file).split(os.sep)
                if len(path_parts) >= 4:
                    source_names.add(path_parts[-4])
            except:
                pass

        for source_name in source_names:
            os.makedirs(os.path.join('real_case', source_name + '_adb'), exist_ok=True)

        # 批量处理文件
        batch_size = 5000
        total_batches = (len(c_files) + batch_size - 1) // batch_size

        for i in range(total_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(c_files))
            batch = c_files[start_idx:end_idx]
            print(f"处理批次 {i+1}/{total_batches}，文件数量: {len(batch)}")
            process_c_files_to_adb(designs_dir, batch)

        print("ADB文件处理完成！")

        # 生成post-HLS JSON文件
        if os.path.exists(data_of_designs_json_path):
            df = pd.read_json(data_of_designs_json_path, orient='records', lines=False)
            for index, row in df.iterrows():
                source_name = row['source_name']
                kernel_name = row['algo_name']
                design_id = row['design_id']
                dsp = row['DSP']
                lut = row['LUT']
                ff = row['FF']
                save_path = os.path.join('real_case', f'{dataset_name}', f'posthls_{source_name}-{kernel_name}-{design_id}.json')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'w') as f:
                    json.dump({"DSP": dsp, "LUT": lut, "FF": ff}, f)
                print(f"Saved {save_path} (index: {index})")

        # 处理图映射
        result_dir = f'{dataset_name}'
        graph_dir = f'{dataset_name}_adb/'

        graph_mapping = dict()
        for adb_file in glob.glob('real_case/' + graph_dir + '*.adb'):
            _, _, file_name = adb_file.split('/')
            parts = file_name.rsplit('-', 1)
            fname = parts[0]
            func_name = parts[1]
            if fname not in graph_mapping:
                graph_mapping[fname] = [func_name]
            else:
                graph_mapping[fname].append(func_name)

        # 处理图文件
        os.makedirs(os.path.join('real_case', result_dir), exist_ok=True)

        total_files = len(graph_mapping.keys())
        print(f"开始处理总共 {total_files} 个文件...")

        with Pool() as pool:
            process_func = partial(process_fname, graph_dir=graph_dir, result_dir=result_dir, graph_mapping=graph_mapping)
            results = []
            for result in tqdm(pool.imap_unordered(process_func, list(graph_mapping.keys())), total=total_files):
                results.append(result)

        successful = [r for r in results if r is not None]
        print(f"图处理完成! 成功: {len(successful)}/{total_files}")

        # 转换为CSV格式
        result_dir = f'{dataset_name}/'
        prefix = 'posthls_'

        graph_mapping_list = []
        num_node_list = []
        num_edge_list = []

        DSP = []
        LUT = []
        FF = []

        node_feat = []
        edge_list = []
        edge_feat = []

        processed_count = 0
        skipped_count = 0
        total_files = len(glob.glob('real_case/' + result_dir + prefix + '*.json'))
        print(f"总文件数: {total_files}")

        for perf_file in tqdm(glob.glob('real_case/' + result_dir + prefix + '*.json'), desc="处理文件"):
            _, _, file_name = perf_file.split('/')
            graph_name = file_name.replace(prefix,'')

            try:
                f = open('real_case/'+result_dir + graph_name, 'r')
                d = json.load(f)
                f.close()
                nodes=d['nodes']
                edges=d['edges']

                try:
                    dsp, lut, ff = get_real_perf(perf_file.replace('.json',''))
                    
                    num_node_list.append(len(nodes))
                    num_edge_list.append(len(edges))
                    graph_mapping_list.append(result_dir + graph_name)
                    
                    # 归一化
                    DSP.append(dsp)
                    LUT.append(lut/1000)
                    FF.append(ff/1000)

                    node_index_map = dict()
                    index = 0

                    for n in nodes:
                        if n[0] not in node_index_map:
                            node_index_map[n[0]] = index
                        node_feat.append(node_to_feature_vector(n[1]))
                        index = index + 1
                    
                    for e in edges:
                        source = node_index_map[e[0]]
                        sink = node_index_map[e[1]]
                        edge_list.append([source,sink])
                        edge_feat.append(edge_to_feature_vector(e[2]))
                    processed_count += 1
                except Exception as e:
                    print(f"Error processing {perf_file}: {e}") 
                    print(traceback.format_exc())
                    skipped_count += 1
                    continue
            except Exception as e:
                skipped_count += 1
                continue

        print(f"处理完成: {processed_count}/{total_files} 文件")
        print(f"跳过: {skipped_count}/{total_files} 文件 ({skipped_count/total_files*100:.2f}%)")

        # 保存CSV文件
        ds_dir = f'{dataset_name}_ds'                
        save_dir = 'real_case/' + ds_dir + '/'
        os.makedirs(save_dir, exist_ok=True)
        
        mapping = pd.DataFrame({'orignal code':graph_mapping_list , 'DSP' : DSP , 'LUT' : LUT, 'FF' : FF})
        NODE_num = pd.DataFrame(num_node_list)
        NODE = pd.DataFrame(node_feat)
        EDGE_num = pd.DataFrame(num_edge_list)
        EDGE_list = pd.DataFrame(edge_list)
        EDGE_feat = pd.DataFrame(edge_feat)

        graph_label_dsp = pd.DataFrame(DSP)
        graph_label_lut = pd.DataFrame(LUT)
        graph_label_ff = pd.DataFrame(FF)

        # 保存CSV文件
        mapping.to_csv(save_dir + 'mapping.csv', index = False)
        NODE_num.to_csv(save_dir + 'num-node-list.csv', index = False, header = False)
        NODE.to_csv(save_dir + 'node-feat.csv', index = False, header = False)

        EDGE_num.to_csv(save_dir + 'num-edge-list.csv', index = False, header = False)
        EDGE_list.to_csv(save_dir + 'edge.csv', index = False, header=False)
        EDGE_feat.to_csv(save_dir + 'edge-feat.csv', index = False, header = False)

        graph_label_dsp.to_csv(save_dir + 'graph-label-dsp.csv', index = False, header = False)
        graph_label_lut.to_csv(save_dir + 'graph-label-lut.csv', index = False, header = False)
        graph_label_ff.to_csv(save_dir + 'graph-label-ff.csv', index = False, header = False)

        print(f"数据集 {dataset_name} 的步骤1处理完成")

    # ==================== STEP 2: 合并所有数据集 ====================
    
    print("\n\n========== 开始步骤2: 合并所有数据集 ==========\n")
    
    # 查找可用的数据集
    case_dir_all = []
    for ds_name in forgehls_dataset_names:
        ds_dir = f'real_case/{ds_name}_ds/'
        if os.path.exists(ds_dir):
            case_dir_all.append(ds_name)
            print(f"找到数据集: {ds_name}")
        else:
            print(f"数据集目录不存在: {ds_dir}")

    print(f"\n总共找到 {len(case_dir_all)} 个可用数据集:")
    for dir_path in case_dir_all:
        print(f" - {dir_path}")

    if not case_dir_all:
        print(f"警告: 未找到匹配的目录")
        return

    # 合并所有数据
    graph_mapping_list = []
    edge_feat = []
    edge_list = []
    node_feat = []

    DSP = []
    LUT = []
    FF = []

    num_node_list = []
    num_edge_list = []

    for case_dir in case_dir_all:
        ds_dir = f'real_case/{case_dir}_ds/'
        graph_mapping_list += pd.read_csv(ds_dir + 'mapping.csv').values.tolist()
        edge_feat += pd.read_csv(ds_dir + 'edge-feat.csv', header = None).values.tolist()
        edge_list += pd.read_csv(ds_dir + 'edge.csv', header = None).values.tolist()
        node_feat += pd.read_csv(ds_dir + 'node-feat.csv', header = None).values.tolist()

        DSP += pd.read_csv(ds_dir + 'graph-label-dsp.csv', header = None).values.tolist()
        LUT += pd.read_csv(ds_dir + 'graph-label-lut.csv', header = None).values.tolist()
        FF += pd.read_csv(ds_dir + 'graph-label-ff.csv', header = None).values.tolist()

        num_node_list += pd.read_csv(ds_dir + 'num-node-list.csv', header = None).values.tolist()
        num_edge_list += pd.read_csv(ds_dir + 'num-edge-list.csv', header = None).values.tolist()

    # 保存合并后的数据集
    dataset_name = "forgehls_kernels"
    save_dir = f'real_case/dataset_ready_for_GNN_csv/{dataset_name}/'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    mapping = pd.DataFrame({'orignal code' : graph_mapping_list, 'DSP' : DSP, 'LUT' : LUT, 'FF' : FF})
    NODE_num = pd.DataFrame(num_node_list)
    EDGE_num = pd.DataFrame(num_edge_list)

    graph_label_dsp = pd.DataFrame(DSP)
    graph_label_lut = pd.DataFrame(LUT)
    graph_label_ff = pd.DataFrame(FF)

    NODE = pd.DataFrame(node_feat)
    EDGE_list = pd.DataFrame(edge_list)
    EDGE_feat = pd.DataFrame(edge_feat)

    mapping.to_csv(save_dir + 'mapping.csv', index = False)
    NODE_num.to_csv(save_dir + 'num-node-list.csv', index = False, header = False)
    EDGE_num.to_csv(save_dir + 'num-edge-list.csv', index = False, header = False)

    graph_label_dsp.to_csv(save_dir + 'graph-label-dsp.csv', index = False, header = False)
    graph_label_lut.to_csv(save_dir + 'graph-label-lut.csv', index = False, header = False)
    graph_label_ff.to_csv(save_dir + 'graph-label-ff.csv', index = False, header = False)

    NODE.to_csv(save_dir + 'node-feat.csv', index = False, header = False)
    EDGE_list.to_csv(save_dir + 'edge.csv', index = False, header = False)
    EDGE_feat.to_csv(save_dir + 'edge-feat.csv', index = False, header = False)

    # 生成训练/验证/测试集分割
    basis = len(pd.read_csv(f'{save_dir}/graph-label-dsp.csv', header=None))
    print(f"总样本数: {basis}")

    indices = [i for i in range(basis)]
    train_indices, temp_indices = model_selection.train_test_split(indices, train_size=0.8, random_state=42)
    valid_indices, test_indices = model_selection.train_test_split(temp_indices, train_size=0.5, random_state=42)

    train_list = pd.DataFrame(sorted(train_indices))
    train_list.to_csv(save_dir + 'train.csv', index=False, header=False)

    valid_list = pd.DataFrame(sorted(valid_indices))
    valid_list.to_csv(save_dir + 'valid.csv', index=False, header=False)

    test_list = pd.DataFrame(sorted(test_indices))
    test_list.to_csv(save_dir + 'test.csv', index=False, header=False)

    # 创建最终数据集结构
    os.makedirs(dataset_save_dir, exist_ok=True)
    metrics = ["lut", "ff", "dsp"]
    
    for metric in metrics:
        final_dataset_dir = f"{dataset_save_dir}/cdfg_{metric}_all_numerical_{dataset_name}/"
        if os.path.exists(final_dataset_dir):
            shutil.rmtree(final_dataset_dir)

    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(create_dataset_structure, metric, dataset_name, save_dir) for metric in metrics]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"创建数据集时出错: {str(e)}")

    print("所有数据集创建完成")
    print(f"Please check the dataset in {dataset_save_dir}")
    
    # 清理临时文件
    if os.path.exists('./real_case'):
        shutil.rmtree('./real_case')
    if os.path.exists('./csv'):
        shutil.rmtree('./csv')
    
    print("ForgeHLS数据预处理完成！")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='ForgeHLS数据预处理脚本')
    parser.add_argument('--designs_base_dir', type=str, 
                        default='/home/user/zedongpeng/workspace/Huggingface/forgehls_lite_10designs',
                        help='设计文件根目录')
    parser.add_argument('--dataset_save_dir', type=str, default='./dataset/',
                        help='数据集保存目录')
    
    args = parser.parse_args()
    
    pre_process_forgehls(args.designs_base_dir, args.dataset_save_dir) 