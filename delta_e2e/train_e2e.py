#!/usr/bin/env python3
"""
End-to-End Differential Learning Training Script
===============================================

This script processes raw kernel and design folders to create paired datasets
and train differential GNN models for performance prediction.

Paths:
- Design: /home/user/zedongpeng/workspace/Huggingface/forgehls_lite_10designs/source_name/algo_name/design_**/
- Kernel: /home/user/zedongpeng/workspace/HLS-Perf-Prediction-with-GNNs/Graphs/forgehls_kernels/kernels/source_name/algo_name/

Author: Zedong Peng
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool
from torch_geometric.nn import RGCNConv, FastRGCNConv, BatchNorm, GINConv
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional
import re
import shutil
import glob
import swanlab
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal
import atexit
import gc
import sys

_shutdown_requested = False

def _graceful_cleanup():
    try:
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    except Exception:
        pass
    try:
        swanlab.finish()
    except Exception:
        pass


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print(f"收到信号 {signum}，准备安全退出...")
    raise KeyboardInterrupt

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)
atexit.register(_graceful_cleanup)

# 图处理函数导入（支持包内与脚本直跑两种方式）
try:
    from delta_e2e.utils import (
        parse_xml_into_graph_single, node_to_feature_vector, edge_to_feature_vector,
        get_node_feature_dims, get_edge_feature_dims
    )
except ModuleNotFoundError:
    from utils import (
        parse_xml_into_graph_single, node_to_feature_vector, edge_to_feature_vector,
        get_node_feature_dims, get_edge_feature_dims
    )

# 归一化因子已移除，直接使用原始值

# 可用资源数量（用于计算ulti-RMSE）
AVAILABLE_RESOURCES = { 
    'dsp': 9024,
    'lut': 1303680,
    'ff': 2607360,
    'bram_18k': 4032
}


# ============================================================================
# Data Collection and Pairing
# ============================================================================

class E2EDifferentialProcessor:
    """端到端差值数据处理器"""
    
    def __init__(self, kernel_base_dir: str, design_base_dir: str, output_dir: str,
                 cache_root: str = "./graph_cache", rebuild_cache: bool = False, hierarchical: bool = False, region: bool = False, max_workers: Optional[int] = None):
        self.kernel_base_dir = kernel_base_dir
        self.design_base_dir = design_base_dir
        self.output_dir = output_dir
        self.rebuild_cache = rebuild_cache
        self.hierarchical = hierarchical  # 是否启用分层区域节点
        self.region = region  # 是否启用 region 信息
        # 并行线程数（仅 CLI 配置；<=0 或 None 则自动）
        if isinstance(max_workers, int) and max_workers > 0:
            self.max_workers = int(max_workers)
        else:
            self.max_workers = min(32, (os.cpu_count() or 8))
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 图缓存路径 - 根据kernel/design根目录生成隔离的子目录，避免不同数据源相互污染
        kb = os.path.abspath(self.kernel_base_dir)
        db = os.path.abspath(self.design_base_dir)
        # 在cache key中加入特征版本，避免特征维度变化导致旧缓存失配
        feature_version = "featv=20250924.0"
        cache_key_src = f"kb={kb}|db={db}|hier={'1' if self.hierarchical else '0'}|region={'1' if self.region else '0'}|{feature_version}"
        cache_key = hashlib.md5(cache_key_src.encode("utf-8")).hexdigest()[:12]
        self.graph_cache_dir = os.path.join(cache_root, cache_key)
        os.makedirs(self.graph_cache_dir, exist_ok=True)
        # 写入元信息，方便排查
        try:
            meta_path = os.path.join(self.graph_cache_dir, "meta.json")
            if not os.path.exists(meta_path):
                with open(meta_path, 'w') as mf:
                    json.dump({
                        "kernel_base_dir": kb,
                        "design_base_dir": db,
                        "cache_key": cache_key,
                        "hierarchical": self.hierarchical,
                        "region": self.region,
                        "feature_version": feature_version,
                        "max_workers": int(self.max_workers)
                    }, mf, indent=2)
        except Exception:
            pass
        
        # 静默初始化，减少console输出
        pass
    
    def collect_all_data(self) -> List[Dict]:
        """收集所有kernel-design配对数据（全局任务并发）"""
        # 检查是否存在缓存的配对数据
        cache_file = os.path.join(self.graph_cache_dir, "paired_graphs.json")
        if (not self.rebuild_cache) and os.path.exists(cache_file):
            return self._load_cached_pairs(cache_file)
        
        pairs: List[Dict] = []
        kernel_base = Path(self.kernel_base_dir)
        design_base = Path(self.design_base_dir)
        
        # 检查路径
        if not kernel_base.exists():
            print(f"错误: Kernel路径不存在: {kernel_base}")
            return pairs
        if not design_base.exists():
            print(f"错误: Design路径不存在: {design_base}")
            return pairs
        
        # 先扫描，构建 (kernel_data, design_dir, source_name, algo_name, design_id) 全局任务列表
        tasks: List[Tuple[Dict, Path, str, str, str]] = []
        
        for source_dir in tqdm(list(design_base.iterdir()), desc="扫描数据源"):
            if not source_dir.is_dir():
                continue
            source_name = source_dir.name
            print("start scanning source_dir: ", source_dir)
            
            for algo_dir in source_dir.iterdir():
                if not algo_dir.is_dir():
                    continue
                algo_name = algo_dir.name
                
                # 对应 kernel
                kernel_path = kernel_base / source_name / algo_name
                if not kernel_path.exists():
                    continue
                
                kernel_data = self._collect_kernel_data(kernel_path, source_name, algo_name)
                if not kernel_data:
                    continue
                
                # 收集所有 design 目录
                design_dirs = [d for d in algo_dir.iterdir() if d.is_dir() and d.name.startswith('design_')]
                if not design_dirs:
                    continue
                
                for design_dir in design_dirs:
                    design_id = design_dir.name
                    tasks.append((kernel_data, design_dir, source_name, algo_name, design_id))
            print("end scanning source_dir: ", source_dir)
        
        if not tasks:
            # 无任务
            return pairs
        
        def _process_task(task: Tuple[Dict, Path, str, str, str]):
            kernel_data, design_dir, source_name, algo_name, design_id = task
            try:
                # print("start processing design_dir: ", design_dir)
                design_data = self._collect_design_data(design_dir, source_name, algo_name, design_id)
                # print("end processing design_dir: ", design_dir)
                if design_data:
                    return self._create_pair(kernel_data, design_data)
                return None
            except Exception:
                return None
        
        from concurrent.futures import ThreadPoolExecutor as _TPE
        with _TPE(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_task = {executor.submit(_process_task, task): task for task in tasks}
            
            # 使用tqdm显示进度，包含成功/失败统计
            with tqdm(total=len(tasks), desc="生成配对", ncols=100, 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                success_count = 0
                for future in as_completed(future_to_task):
                    result = future.result()
                    if result:
                        pairs.append(result)
                        success_count += 1
                    pbar.update(1)
                    pbar.set_postfix({'成功': success_count, '失败': pbar.n - success_count})
        
        # 缓存配对数据
        if pairs:
            self._save_cached_pairs(pairs, cache_file)
        
        return pairs
    
    def _collect_kernel_data(self, kernel_path: Path, source_name: str, algo_name: str) -> Optional[Dict]:
        """收集单个kernel的数据"""
        try:
            # 查找csynth.xml文件
            csynth_files = list(kernel_path.rglob("csynth.xml"))
            if not csynth_files:
                return None
            
            # 解析性能数据
            perf_data = self._parse_csynth_xml(csynth_files[0])
            if not perf_data:
                return None
            
            # 查找图文件（限制文件名最多包含一个点，避免处理 *.*.adb）
            graph_files = [p for p in kernel_path.rglob("*.adb") if p.name.count('.') <= 1]
            if not graph_files:
                return None

            # 处理多ADB并图
            graph_data = self._process_graphs(graph_files)
            if not graph_data:
                return None
            
            return {
                'type': 'kernel',
                'source_name': source_name,
                'algo_name': algo_name,
                'base_path': str(kernel_path),
                'performance': perf_data,
                'graph': graph_data,
                'pragma_info': {'pragma_count': 0}  # kernel没有pragma
            }
            
        except Exception as e:
            return None
    
    def _collect_design_data(self, design_path: Path, source_name: str, algo_name: str, design_id: str) -> Optional[Dict]:
        """收集单个design的数据"""
        try:
            # 查找csynth.xml文件
            csynth_files = list(design_path.rglob("csynth.xml"))
            if not csynth_files:
                return None
            
            # 解析性能数据
            perf_data = self._parse_csynth_xml(csynth_files[0])
            if not perf_data:
                return None
            
            # 查找图文件（限制文件名最多包含一个点，避免处理 *.*.adb）
            graph_files = [p for p in design_path.rglob("*.adb") if p.name.count('.') <= 1]
            if not graph_files:
                return None

            # 处理多ADB并图
            graph_data = self._process_graphs(graph_files)
            if not graph_data:
                return None
            
            # 提取pragma信息
            pragma_info = self._extract_pragma_info(design_path)
            
            return {
                'type': 'design',
                'source_name': source_name,
                'algo_name': algo_name,
                'design_id': design_id,
                'base_path': str(design_path),
                'performance': perf_data,
                'graph': graph_data,
                'pragma_info': pragma_info
            }
            
        except Exception as e:
            return None
    
    def _parse_csynth_xml(self, xml_path: Path) -> Optional[Dict]:
        """解析csynth.xml文件"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 提取性能数据
            best_latency = root.find('.//PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency')
            dsp = root.find('.//AreaEstimates/Resources/DSP')
            lut = root.find('.//AreaEstimates/Resources/LUT')
            ff = root.find('.//AreaEstimates/Resources/FF')
            
            if (best_latency is not None and best_latency.text != 'undef' and
                dsp is not None and lut is not None and ff is not None):
                
                return {
                    'best_latency': float(best_latency.text),
                    'DSP': int(dsp.text),
                    'LUT': int(lut.text),
                    'FF': int(ff.text)
                }
        except Exception as e:
            return None
        
        return None
    
    def _process_graph_file(self, adb_path: Path) -> Optional[Data]:
        """处理图文件，转换为PyTorch Geometric格式"""
        try:
            # 解析XML图（根据外部开关选择是否启用分层）
            from argparse import Namespace
            hierarchical_flag = False
            try:
                # 在训练主逻辑中会以 processor.hierarchical 传入
                hierarchical_flag = getattr(self, 'hierarchical', False)
            except Exception:
                hierarchical_flag = False
            region_flag = False
            try:
                region_flag = getattr(self, 'region', False)
            except Exception:
                region_flag = False
            # 优先尝试带 region 形参与 hierarchical 的解析；若不支持，则回退
            try:
                G = parse_xml_into_graph_single(str(adb_path), hierarchical=hierarchical_flag, region=region_flag)
            except TypeError:
                G = parse_xml_into_graph_single(str(adb_path), hierarchical=hierarchical_flag)
            if G is None or len(G.nodes()) == 0:
                return None
            
            # 转换为PyTorch Geometric格式
            nodes = list(G.nodes())
            edges = list(G.edges())
            
            if len(nodes) == 0:
                return None
            
            # 构建节点特征
            node_features = []
            node_mapping = {node: i for i, node in enumerate(nodes)}
            
            for node in nodes:
                node_attr = G.nodes[node] if node in G.nodes else {}
                node_feat = node_to_feature_vector(node_attr)
                node_features.append(node_feat)
            
            # 构建边索引和边特征
            edge_index = []
            edge_features = []
            
            for edge in edges:
                if edge[0] in node_mapping and edge[1] in node_mapping:
                    source_idx = node_mapping[edge[0]]
                    target_idx = node_mapping[edge[1]]
                    edge_index.append([source_idx, target_idx])
                    
                    edge_attr = G.edges[edge] if edge in G.edges else {}
                    edge_feat = edge_to_feature_vector(edge_attr)
                    edge_features.append(edge_feat)
            
            if len(edge_index) == 0:
                # 如果没有边，创建自环
                edge_index = [[i, i] for i in range(len(nodes))]
                edge_features = [[0, 0] for _ in range(len(nodes))]
            
            # 转换为tensor
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float) if edge_features else None
            
            # 携带图级流水线属性（若存在）
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            try:
                data.has_pipeline = int(G.graph.get('has_pipeline', 0))
                data.pipeline_region_count = int(G.graph.get('pipeline_region_count', 0))
                data.avg_ii = float(G.graph.get('avg_ii', 0.0))
                data.max_pipe_depth = int(G.graph.get('max_pipe_depth', 0))
                data.pipeline_components_present = int(G.graph.get('pipeline_components_present', 0))
                data.pipeline_signals_present = int(G.graph.get('pipeline_signals_present', 0))
            except Exception:
                pass
            
            return data
            
        except Exception as e:
            print(f"处理图文件失败 {adb_path}: {e}")
            return None
    
    def _process_graphs(self, adb_paths: List[Path]) -> Optional[Data]:
        """将多个ADB并为一个PyG图（节点重标，特征拼接，边索引偏移）。"""
        try:
            hierarchical_flag = False
            try:
                hierarchical_flag = getattr(self, 'hierarchical', False)
            except Exception:
                hierarchical_flag = False

            # 并行处理每个 ADB 为 Data，然后合并
            datas: List[Optional[Data]] = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for d in executor.map(self._process_graph_file, adb_paths):
                    if d is not None:
                        datas.append(d)

            if not datas:
                return None

            all_x: List[torch.Tensor] = []
            all_edge_index: List[torch.Tensor] = []
            all_edge_attr: List[torch.Tensor] = []
            node_offset = 0

            # 聚合图级流水线指标
            has_pipeline_any = 0
            pipeline_region_count_sum = 0
            avg_ii_list: List[float] = []
            max_pipe_depth_max = 0
            components_any = 0
            signals_any = 0

            for data in datas:
                x = data.x
                ei = data.edge_index + node_offset
                ea = data.edge_attr if data.edge_attr is not None else torch.zeros((ei.size(1), 2), dtype=torch.float)

                all_x.append(x)
                all_edge_index.append(ei)
                all_edge_attr.append(ea)

                # 聚合图级属性
                try:
                    has_pipeline_any = max(has_pipeline_any, int(getattr(data, 'has_pipeline', 0)))
                    pipeline_region_count_sum += int(getattr(data, 'pipeline_region_count', 0))
                    ai = float(getattr(data, 'avg_ii', 0.0))
                    if ai > 0:
                        avg_ii_list.append(ai)
                    mpd = int(getattr(data, 'max_pipe_depth', 0))
                    if mpd > max_pipe_depth_max:
                        max_pipe_depth_max = mpd
                    components_any = max(components_any, int(getattr(data, 'pipeline_components_present', 0)))
                    signals_any = max(signals_any, int(getattr(data, 'pipeline_signals_present', 0)))
                except Exception:
                    pass

                node_offset += x.size(0)

            # 合并
            x_cat = torch.cat(all_x, dim=0)
            edge_index_cat = torch.cat(all_edge_index, dim=1)
            edge_attr_cat = torch.cat(all_edge_attr, dim=0) if all_edge_attr else None

            data_merged = Data(x=x_cat, edge_index=edge_index_cat, edge_attr=edge_attr_cat)
            # 写入聚合属性
            data_merged.has_pipeline = has_pipeline_any
            data_merged.pipeline_region_count = pipeline_region_count_sum
            data_merged.avg_ii = (sum(avg_ii_list) / len(avg_ii_list)) if avg_ii_list else 0.0
            data_merged.max_pipe_depth = max_pipe_depth_max
            data_merged.pipeline_components_present = components_any
            data_merged.pipeline_signals_present = signals_any
            data_merged.num_modules = len(datas)

            return data_merged
        except Exception as e:
            print(f"并图失败: {e}")
            return None
    
    def _extract_pragma_info(self, design_path: Path) -> Dict:
        """提取pragma信息"""
        pragma_info = {
            'pragma_count': 0,
            'pragma_types': {},
            'optimization_pragmas': 0
        }
        
        # 查找源代码文件
        source_files = []
        for ext in ['.c', '.cpp', '.h', '.hpp']:
            source_files.extend(list(design_path.rglob(f"*{ext}")))
        
        pragma_patterns = {
            'PIPELINE': r'#pragma\s+HLS\s+PIPELINE',
            'UNROLL': r'#pragma\s+HLS\s+UNROLL',
            'ARRAY_PARTITION': r'#pragma\s+HLS\s+ARRAY_PARTITION',
            'DATAFLOW': r'#pragma\s+HLS\s+DATAFLOW',
            'INLINE': r'#pragma\s+HLS\s+INLINE',
            'INTERFACE': r'#pragma\s+HLS\s+INTERFACE'
        }
        
        for source_file in source_files:
            try:
                with open(source_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                for pragma_type, pattern in pragma_patterns.items():
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    count = len(matches)
                    pragma_info['pragma_types'][pragma_type] = pragma_info['pragma_types'].get(pragma_type, 0) + count
                    pragma_info['pragma_count'] += count
                    
                    if pragma_type in ['PIPELINE', 'UNROLL', 'DATAFLOW', 'ARRAY_PARTITION']:
                        pragma_info['optimization_pragmas'] += count
                        
            except Exception as e:
                print(f"读取源文件失败 {source_file}: {e}")
        
        return pragma_info
    
    def _create_pair(self, kernel_data: Dict, design_data: Dict) -> Optional[Dict]:
        """创建kernel-design配对"""
        try:
            # 验证数据完整性
            if not all([
                kernel_data.get('performance'),
                design_data.get('performance'),
                kernel_data.get('graph'),
                design_data.get('graph')
            ]):
                return None
            
            # 获取性能数据
            k_perf = kernel_data['performance']
            d_perf = design_data['performance']
            
            # 计算性能差值（不归一化，保持原始值）
            performance_delta = {
                'latency_delta': d_perf['best_latency'] - k_perf['best_latency'],
                'dsp_delta': d_perf['DSP'] - k_perf['DSP'],
                'lut_delta': d_perf['LUT'] - k_perf['LUT'],
                'ff_delta': d_perf['FF'] - k_perf['FF']
            }
            
            # 添加性能标签到图数据 - 使用原始值，不进行归一化
            kernel_graph = kernel_data['graph']
            design_graph = design_data['graph']
            
            kernel_graph.y = torch.tensor([
                k_perf['DSP'],
                k_perf['LUT'],
                k_perf['FF'],
                k_perf['best_latency']
            ], dtype=torch.float).unsqueeze(0)
            
            design_graph.y = torch.tensor([
                d_perf['DSP'],
                d_perf['LUT'],
                d_perf['FF'],
                d_perf['best_latency']
            ], dtype=torch.float).unsqueeze(0)
            
            return {
                'pair_id': f"{kernel_data['source_name']}_{kernel_data['algo_name']}_{design_data['design_id']}",
                'kernel_graph': kernel_graph,
                'design_graph': design_graph,
                'performance_delta': performance_delta,
                'pragma_info': design_data['pragma_info'],
                'kernel_info': kernel_data,
                'design_info': design_data
            }
            
        except Exception as e:
            print(f"创建配对失败: {e}")
            return None
    
    def _save_cached_pairs(self, pairs: List[Dict], cache_file: str):
        """保存配对数据到缓存"""
        # 将图数据转换为可序列化格式
        cached_pairs = []
        for pair in tqdm(pairs, desc="缓存图数据"):
            cached_pair = {
                'pair_id': pair['pair_id'],
                'performance_delta': pair['performance_delta'],
                'pragma_info': pair['pragma_info'],
                'kernel_info': {k: v for k, v in pair['kernel_info'].items() if k != 'graph'},
                'design_info': {k: v for k, v in pair['design_info'].items() if k != 'graph'},
                # 保存图的tensor数据
                'kernel_graph_data': {
                    'x': pair['kernel_graph'].x.tolist(),
                    'edge_index': pair['kernel_graph'].edge_index.tolist(),
                    'edge_attr': pair['kernel_graph'].edge_attr.tolist() if pair['kernel_graph'].edge_attr is not None else None,
                    'y': pair['kernel_graph'].y.tolist()
                },
                'kernel_graph_meta': {
                    'has_pipeline': int(getattr(pair['kernel_graph'], 'has_pipeline', 0)),
                    'pipeline_region_count': int(getattr(pair['kernel_graph'], 'pipeline_region_count', 0)),
                    'avg_ii': float(getattr(pair['kernel_graph'], 'avg_ii', 0.0)),
                    'max_pipe_depth': int(getattr(pair['kernel_graph'], 'max_pipe_depth', 0)),
                    'pipeline_components_present': int(getattr(pair['kernel_graph'], 'pipeline_components_present', 0)),
                    'pipeline_signals_present': int(getattr(pair['kernel_graph'], 'pipeline_signals_present', 0))
                },
                'design_graph_data': {
                    'x': pair['design_graph'].x.tolist(),
                    'edge_index': pair['design_graph'].edge_index.tolist(),
                    'edge_attr': pair['design_graph'].edge_attr.tolist() if pair['design_graph'].edge_attr is not None else None,
                    'y': pair['design_graph'].y.tolist()
                },
                'design_graph_meta': {
                    'has_pipeline': int(getattr(pair['design_graph'], 'has_pipeline', 0)),
                    'pipeline_region_count': int(getattr(pair['design_graph'], 'pipeline_region_count', 0)),
                    'avg_ii': float(getattr(pair['design_graph'], 'avg_ii', 0.0)),
                    'max_pipe_depth': int(getattr(pair['design_graph'], 'max_pipe_depth', 0)),
                    'pipeline_components_present': int(getattr(pair['design_graph'], 'pipeline_components_present', 0)),
                    'pipeline_signals_present': int(getattr(pair['design_graph'], 'pipeline_signals_present', 0))
                }
            }
            cached_pairs.append(cached_pair)
        
        with open(cache_file, 'w') as f:
            json.dump(cached_pairs, f, indent=2)
    
    def _load_cached_pairs(self, cache_file: str) -> List[Dict]:
        """从缓存加载配对数据"""
        with open(cache_file, 'r') as f:
            cached_pairs = json.load(f)
        
        # 重建图数据
        pairs = []
        for cached_pair in tqdm(cached_pairs, desc="加载缓存图数据"):
            try:
                # 重建kernel图
                k_data = cached_pair['kernel_graph_data']
                kernel_graph = Data(
                    x=torch.tensor(k_data['x'], dtype=torch.float),
                    edge_index=torch.tensor(k_data['edge_index'], dtype=torch.long),
                    edge_attr=torch.tensor(k_data['edge_attr'], dtype=torch.float) if k_data['edge_attr'] else None,
                    y=torch.tensor(k_data['y'], dtype=torch.float)
                )
                
                # 重建design图
                d_data = cached_pair['design_graph_data']
                design_graph = Data(
                    x=torch.tensor(d_data['x'], dtype=torch.float),
                    edge_index=torch.tensor(d_data['edge_index'], dtype=torch.long),
                    edge_attr=torch.tensor(d_data['edge_attr'], dtype=torch.float) if d_data['edge_attr'] else None,
                    y=torch.tensor(d_data['y'], dtype=torch.float)
                )
                
                # 恢复图级流水线元数据（如存在）
                try:
                    k_meta = cached_pair.get('kernel_graph_meta', {})
                    kernel_graph.has_pipeline = int(k_meta.get('has_pipeline', 0))
                    kernel_graph.pipeline_region_count = int(k_meta.get('pipeline_region_count', 0))
                    kernel_graph.avg_ii = float(k_meta.get('avg_ii', 0.0))
                    kernel_graph.max_pipe_depth = int(k_meta.get('max_pipe_depth', 0))
                    kernel_graph.pipeline_components_present = int(k_meta.get('pipeline_components_present', 0))
                    kernel_graph.pipeline_signals_present = int(k_meta.get('pipeline_signals_present', 0))
                except Exception:
                    pass
                try:
                    d_meta = cached_pair.get('design_graph_meta', {})
                    design_graph.has_pipeline = int(d_meta.get('has_pipeline', 0))
                    design_graph.pipeline_region_count = int(d_meta.get('pipeline_region_count', 0))
                    design_graph.avg_ii = float(d_meta.get('avg_ii', 0.0))
                    design_graph.max_pipe_depth = int(d_meta.get('max_pipe_depth', 0))
                    design_graph.pipeline_components_present = int(d_meta.get('pipeline_components_present', 0))
                    design_graph.pipeline_signals_present = int(d_meta.get('pipeline_signals_present', 0))
                except Exception:
                    pass
                
                pair = {
                    'pair_id': cached_pair['pair_id'],
                    'kernel_graph': kernel_graph,
                    'design_graph': design_graph,
                    'performance_delta': cached_pair['performance_delta'],
                    'pragma_info': cached_pair['pragma_info'],
                    'kernel_info': cached_pair['kernel_info'],
                    'design_info': cached_pair['design_info']
                }
                
                pairs.append(pair)
                
            except Exception as e:
                continue
        
        return pairs


# ============================================================================
# Differential GNN Model
# ============================================================================

class SimpleDifferentialGNN(nn.Module):
    """支持多种GNN架构的差值学习模型"""
    
    def __init__(self, node_dim: int, hidden_dim: int = 128, num_layers: int = 3, 
                 dropout: float = 0.1, target_metric: str = 'dsp', differential: bool = True,
                 gnn_type: str = 'gcn'):
        super().__init__()
        self.target_metric = target_metric
        self.hidden_dim = hidden_dim
        self.differential = differential
        self.gnn_type = gnn_type.lower()
        
        # 节点编码器
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        
        # GNN层 - 支持多种架构
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            if self.gnn_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif self.gnn_type == 'gin':
                # GIN需要一个MLP作为参数
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                self.convs.append(GINConv(mlp))
            elif self.gnn_type == 'rgcn':
                # RGCN需要边类型数量，使用特征定义中的边类型维度自动获取
                self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations=get_edge_feature_dims()[0], num_bases=30))
            elif self.gnn_type == 'fast_rgcn':
                self.convs.append(FastRGCNConv(hidden_dim, hidden_dim, num_relations=get_edge_feature_dims()[0], num_bases=30))
            else:
                raise ValueError(f"不支持的GNN类型: {self.gnn_type}. 支持的类型: gcn, gin, rgcn, fast_rgcn")
            
            # 为所有GNN类型添加归一化层（使用LayerNorm以避免单样本BatchNorm错误）
            from torch_geometric.nn import LayerNorm
            self.batch_norms.append(LayerNorm(hidden_dim))
        
        # 预测头 - 根据模式选择输入维度
        input_dim = hidden_dim * 2 if differential else hidden_dim  # differential: kernel+design, direct: design only
        self.prediction_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 指标索引映射
        self.metric_idx = {'dsp': 0, 'lut': 1, 'ff': 2, 'latency': 3}[target_metric]
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode_graph(self, data):
        """编码图 - 支持多种GNN架构"""
        x = self.node_encoder(data.x.float())
        
        for i, conv in enumerate(self.convs):
            if self.gnn_type == 'gcn':
                x = conv(x, data.edge_index)
            elif self.gnn_type == 'gin':
                # GIN使用标准的消息传递
                x = conv(x, data.edge_index)
            elif self.gnn_type in ['rgcn', 'fast_rgcn']:
                # RGCN需要边类型
                edge_type = data.edge_attr[:, 0].long() if hasattr(data, 'edge_attr') and data.edge_attr is not None else torch.zeros(data.edge_index.size(1), dtype=torch.long, device=data.edge_index.device)
                x = conv(x, data.edge_index, edge_type)
            
            # 应用BatchNorm和激活函数
            x = self.batch_norms[i](x)
            if i < len(self.convs) - 1:  # 最后一层不用ReLU
                x = F.relu(x)
        
        # 图级别池化
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
        # 改为 sum 池化以更贴近资源求和语义
        return global_add_pool(x, batch)
    
    def forward(self, kernel_graph, design_graph, pragma_count):
        """前向传播 - 支持差分和直接预测两种模式"""
        # 编码图
        design_repr = self.encode_graph(design_graph)
        
        if self.differential:
            # 差分模式：使用kernel+design
            kernel_repr = self.encode_graph(kernel_graph)
            combined = torch.cat([kernel_repr, design_repr], dim=-1)
            prediction = self.prediction_head(combined)
            return {'delta_pred': prediction}
        else:
            # 直接预测模式：只使用design
            prediction = self.prediction_head(design_repr)
            return {'direct_pred': prediction}


# ============================================================================
# Dataset Class
# ============================================================================

class E2EDifferentialDataset(torch.utils.data.Dataset):
    """端到端差值学习数据集"""
    
    def __init__(self, pairs: List[Dict], target_metric: str = 'dsp'):
        self.pairs = pairs
        self.target_metric = target_metric
        self.metric_idx = {'dsp': 0, 'lut': 1, 'ff': 2, 'latency': 3}[target_metric]
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        kernel_graph = pair['kernel_graph']
        design_graph = pair['design_graph']
        
        # 提取目标指标的性能值
        kernel_perf = kernel_graph.y[0, self.metric_idx].unsqueeze(0)
        design_perf = design_graph.y[0, self.metric_idx].unsqueeze(0)
        
        # 计算差值
        delta_key = f'{self.target_metric}_delta' if self.target_metric != 'latency' else 'latency_delta'
        perf_delta = torch.tensor([pair['performance_delta'][delta_key]], dtype=torch.float)
        
        # Pragma计数
        pragma_count = torch.tensor([pair['pragma_info']['pragma_count']], dtype=torch.long)
        
        return {
            'kernel_graph': kernel_graph,
            'design_graph': design_graph,
            'kernel_perf': kernel_perf,
            'design_perf': design_perf,
            'performance_delta': perf_delta,
            'pragma_count': pragma_count,
            'pair_id': pair['pair_id']
        }


def differential_collate_fn(batch):
    """自定义批处理函数"""
    kernel_graphs = [item['kernel_graph'] for item in batch]
    design_graphs = [item['design_graph'] for item in batch]
    
    kernel_perfs = torch.cat([item['kernel_perf'] for item in batch], dim=0)
    design_perfs = torch.cat([item['design_perf'] for item in batch], dim=0)
    perf_deltas = torch.cat([item['performance_delta'] for item in batch], dim=0)
    pragma_counts = torch.cat([item['pragma_count'] for item in batch], dim=0)
    
    # 使用PyG的Batch直接拼接，避免在collate中再启动DataLoader
    kernel_batch = Batch.from_data_list(kernel_graphs)
    design_batch = Batch.from_data_list(design_graphs)
    
    return {
        'kernel_graph': kernel_batch,
        'design_graph': design_batch,
        'kernel_perf': kernel_perfs,
        'design_perf': design_perfs,
        'performance_delta': perf_deltas,
        'pragma_count': pragma_counts
    }


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, device, train_loader, optimizer, grad_accum_steps=1, max_grad_norm=1.0):
    """训练一个epoch"""
    model.train()
    loss_sum_gpu = torch.zeros(1, device=device)
    batch_count = 0
    
    optimizer.zero_grad(set_to_none=True)
    
    for step_idx, batch in enumerate(train_loader):
        # 移动到设备
        kernel_graph = batch['kernel_graph'].to(device, non_blocking=True)
        design_graph = batch['design_graph'].to(device, non_blocking=True)
        pragma_count = batch['pragma_count'].to(device, non_blocking=True)
        
        # 前向传播
        predictions = model(kernel_graph, design_graph, pragma_count)
        
        # 计算损失
        targets = {
            'kernel_perf': batch['kernel_perf'].to(device, non_blocking=True),
            'design_perf': batch['design_perf'].to(device, non_blocking=True),
            'performance_delta': batch['performance_delta'].to(device, non_blocking=True)
        }
        
        if model.differential:
            target_values = targets['performance_delta'].squeeze()
            pred_output = predictions['delta_pred'].squeeze()
        else:
            target_values = targets['design_perf'].squeeze()
            pred_output = predictions['direct_pred'].squeeze()
        
        # 计算原始值空间的损失 (MAE)
        loss = F.l1_loss(pred_output, target_values)
        
        # 梯度累积缩放
        loss_scaled = loss / max(1, grad_accum_steps)
        
        # 反向传播
        loss_scaled.backward()
        
        # 仅在累积到步数或最后一个batch时执行step
        should_step = ((step_idx + 1) % max(1, grad_accum_steps) == 0) or (step_idx + 1 == len(train_loader))
        if should_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        # 累加用于日志的loss（不使用.item()，避免每步同步）
        loss_sum_gpu += loss.detach()
        batch_count += 1
    
    # 仅在epoch末同步到CPU
    return (loss_sum_gpu.item() / batch_count) if batch_count > 0 else float('inf')


def evaluate_model(model, data_loader, device, target_metric='dsp', return_predictions=False):
    """评估模型性能 - 只评估差值"""
    model.eval()
    total_loss = 0
    batch_count = 0
    
    all_delta_preds = []
    all_delta_true = []
    
    with torch.no_grad():
        for batch in data_loader:
            # 移动到设备
            kernel_graph = batch['kernel_graph'].to(device, non_blocking=True)
            design_graph = batch['design_graph'].to(device, non_blocking=True)
            pragma_count = batch['pragma_count'].to(device, non_blocking=True)
            
            # 前向传播
            predictions = model(kernel_graph, design_graph, pragma_count)
            
            # 准备目标值
            targets = {
                'kernel_perf': batch['kernel_perf'].to(device),
                'design_perf': batch['design_perf'].to(device),
                'performance_delta': batch['performance_delta'].to(device)
            }
            
            if model.differential:
                # 差分模式：使用原始差值目标值
                target_values = targets['performance_delta'].squeeze()
                pred_output = predictions['delta_pred'].squeeze()
            else:
                # 直接预测模式：使用原始design性能目标值
                target_values = targets['design_perf'].squeeze()
                pred_output = predictions['direct_pred'].squeeze()
            
            # 计算原始值空间的损失
            loss = F.l1_loss(pred_output, target_values)
            total_batch_loss = loss
            
            total_loss += total_batch_loss.item()
            batch_count += 1
            
            # 收集预测结果（使用原始值，无归一化）
            all_delta_preds.append(pred_output.cpu())
            all_delta_true.append(target_values.cpu())
    
    # 计算评估指标
    avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
    
    if all_delta_preds:
        delta_preds = torch.cat(all_delta_preds, dim=0)
        delta_true = torch.cat(all_delta_true, dim=0)
        
        # 计算差值指标 - 直接使用原始值
        target_metric = getattr(model, 'target_metric', 'dsp')
        
        # 计算原始值空间的MAE和RMSE（数据已经是原始值）
        delta_mae = F.l1_loss(delta_preds.squeeze(), delta_true.squeeze()).item()
        delta_rmse = torch.sqrt(F.mse_loss(delta_preds.squeeze(), delta_true.squeeze())).item()
        
        # 计算ulti-RMSE (actual-RMSE / available-resource-num)
        if target_metric not in ('dsp', 'lut', 'ff'):
            raise ValueError(f"ulti-RMSE 仅支持 dsp/lut/ff，当前为: {target_metric}. 请将 --target_metric 设置为 dsp/lut/ff，或修改代码以跳过该指标。")
        available_resource_num = AVAILABLE_RESOURCES.get(target_metric, None)
        if available_resource_num is None:
            raise ValueError(f"未配置 AVAILABLE_RESOURCES['{target_metric}']。请在文件顶部的 AVAILABLE_RESOURCES 中填入该资源的总数量。")
        if not isinstance(available_resource_num, (int, float)) or available_resource_num <= 0:
            raise ValueError(f"AVAILABLE_RESOURCES['{target_metric}'] 必须为正数，当前为: {available_resource_num}")
        delta_ulti_rmse = delta_rmse / available_resource_num
        
        # 计算R2指标 - 修复维度问题
        delta_true_sq = delta_true.squeeze()
        delta_preds_sq = delta_preds.squeeze()
        
        # R2 (决定系数) - 只计算差值的R2
        def calculate_r2(y_true, y_pred):
            ss_res = torch.sum((y_true - y_pred) ** 2)
            ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
            return (1 - ss_res / (ss_tot + 1e-8)).item()
        
        delta_r2 = calculate_r2(delta_true_sq, delta_preds_sq)
        
        metrics_dict = {
            'avg_loss': avg_loss,
            'delta_mae': delta_mae,
            'delta_rmse': delta_rmse,
            'delta_ulti_rmse': delta_ulti_rmse,
            'delta_r2': delta_r2
        }
        
        # 只在需要时添加tensor数据（用于散点图）
        if len(delta_preds_sq) > 0:
            metrics_dict.update({
                'delta_preds': delta_preds_sq,
                'delta_true': delta_true_sq
            })
        
        return metrics_dict
    
    return {'avg_loss': avg_loss, 'delta_mae': 0, 'delta_rmse': 0, 'delta_ulti_rmse': 0, 'delta_r2': 0}


def _create_prediction_plots(test_metrics: Dict, target_metric: str, output_dir: str):
    """创建差值预测vs真实值散点图"""
    delta_true = test_metrics['delta_true'].cpu().numpy()
    delta_preds = test_metrics['delta_preds'].cpu().numpy()
    
    # 差值预测散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(delta_true, delta_preds, alpha=0.6, s=50)
    min_val = min(delta_true.min(), delta_preds.min())
    max_val = max(delta_true.max(), delta_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
    plt.xlabel(f'True Delta {target_metric.upper()} (Real Resource Units)')
    plt.ylabel(f'Predicted Delta {target_metric.upper()} (Real Resource Units)')
    plt.title(f'Performance Delta Prediction (Δ{target_metric.upper()}) - Real Resource Scale')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    scatter_path = os.path.join(output_dir, f'prediction_scatter_{target_metric}.png')
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 上传到SwanLab
    swanlab.log({"prediction_scatter": swanlab.Image(scatter_path)})


# =========================================================================
# Table logging helper (SwanLab)
# =========================================================================

def _log_table_safe(table_key: str, columns: List[str], rows: List[List], output_dir: str):
    """已停用的表格上传函数：不再向 SwanLab 上传表格或生成 CSV。"""
    return


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='端到端差值学习GNN训练')
    
    # 路径参数
    parser.add_argument('--kernel_base_dir', type=str,
                        default='/home/user/zedongpeng/workspace/HLS-Perf-Prediction-with-GNNs/Graphs/forgehls_kernels/',
                        help='Kernel数据根目录')
    parser.add_argument('--design_base_dir', type=str,
                        default='/home/user/zedongpeng/workspace/Huggingface/forgehls_lite_10designs/',
                        help='Design数据根目录')
    parser.add_argument('--ood_design_base_dir', type=str,
                        default='/home/user/zedongpeng/workspace/Huggingface/forgehls_benchmark',
                        help='OOD Design数据根目录（可选，不存在则跳过 OOD 评估）')
    
    # 模型参数
    parser.add_argument('--target_metric', type=str, default='dsp',
                        choices=['dsp', 'lut', 'ff', 'latency'],
                        help='目标预测指标')
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=3, help='GNN层数')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    parser.add_argument('--gnn_type', type=str, default='gcn',
                        choices=['gcn', 'gin', 'rgcn', 'fast_rgcn'],
                        help='GNN架构类型')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--device', type=int, default=0, help='GPU设备ID')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='梯度累积步数（>1 可提高有效批大小）')
    parser.add_argument('--loader_workers', type=int, default=8, help='DataLoader worker 数量（>0 启用多进程加载）')
    parser.add_argument('--prefetch_factor', type=int, default=4, help='DataLoader 预取批数（num_workers>0 时有效）')
    parser.add_argument('--persistent_workers', type=str, default='true', choices=['true', 'false'], help='是否持久化 DataLoader workers')
    # 新增：控制 DataLoader 是否使用 pinned memory（大数据集/CPU 内存紧张时建议关闭）
    parser.add_argument('--pin_memory', type=str, default='true', choices=['true', 'false'], help='是否使用 DataLoader pin_memory')
    
    # 输出与缓存参数（统一到项目 GNN 目录下，不受当前工作目录影响）
    project_root = Path(__file__).resolve().parents[1]  # 指向 GNN 目录
    default_output_dir = str(project_root / 'differential_output_e2e')
    default_cache_root = str(project_root / 'graph_cache')

    parser.add_argument('--output_dir', type=str, default=default_output_dir, help='输出目录')
    # 缓存参数
    parser.add_argument('--cache_root', type=str, default=default_cache_root, help='图数据缓存根目录')
    parser.add_argument('--rebuild_cache', action='store_true', help='忽略已有缓存并重新构建')
    
    # 调试参数
    parser.add_argument('--max_pairs', type=int, default=None, help='最大配对数量（调试用）')
    
    # 模式参数
    parser.add_argument('--differential', type=str, default='true', 
                        choices=['true', 'false'],
                        help='是否使用差分学习模式：true=差分学习(kernel+design)，false=直接预测(仅design)')
    
    # 训练策略参数
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Linear warmup 的 epoch 数（0 表示关闭）')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='学习率下限（避免过小导致训练停滞）')
    
    # 层次化图开关
    parser.add_argument('--hierarchical', type=str, default='off', choices=['on', 'off'],
                        help='是否启用多层次结构构图（hierarchical graph），on/off')
    # Region 信息开关
    parser.add_argument('--region', type=str, default='off', choices=['on', 'off'],
                        help='是否启用 region 信息（构图元数据中记录），on/off')

    # 并行参数
    parser.add_argument('--max_workers', type=int, default=-1,
                        help='数据处理并行线程数（仅 CLI；<=0 表示自动）')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"e2e_delta_{args.target_metric}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化SwanLab
    swanlab.init(
        project="HLS-Differential-Learning",
        experiment_name=f"E2E_Delta_{args.target_metric}_{timestamp}",
        config=vars(args),
        logdir=output_dir
    )
    
    # ==================== 数据处理 ====================
    
    processor = E2EDifferentialProcessor(
        kernel_base_dir=args.kernel_base_dir,
        design_base_dir=args.design_base_dir,
        output_dir=output_dir,
        cache_root=args.cache_root,
        rebuild_cache=args.rebuild_cache,
        hierarchical=(args.hierarchical.lower() == 'on'),
        region=(args.region.lower() == 'on'),
        max_workers=(None if args.max_workers is None or args.max_workers <= 0 else args.max_workers)
    )
    
    # 收集配对数据
    pairs = processor.collect_all_data()
    
    if not pairs:
        print("错误: 未找到有效的kernel-design配对")
        swanlab.finish()
        return
    
    # 额外：构建 OOD 配对（如果提供）
    ood_pairs = []
    if args.ood_design_base_dir and os.path.exists(args.ood_design_base_dir):
        ood_processor = E2EDifferentialProcessor(
            kernel_base_dir=args.kernel_base_dir,
            design_base_dir=args.ood_design_base_dir,
            output_dir=output_dir,
            cache_root=args.cache_root,
            rebuild_cache=args.rebuild_cache,
            hierarchical=(args.hierarchical.lower() == 'on'),
            region=(args.region.lower() == 'on'),
            max_workers=(None if args.max_workers is None or args.max_workers <= 0 else args.max_workers)
        )
        ood_pairs = ood_processor.collect_all_data()
    else:
        print(f"提示: OOD 路径不可用或未提供，将跳过 OOD 评估: {args.ood_design_base_dir}")
    
    # 记录数据集信息
    swanlab.log({"dataset/total_pairs": len(pairs)})
    if ood_pairs:
        swanlab.log({"dataset/ood_pairs": len(ood_pairs)})
    # 记录是否使用层次化图
    swanlab.log({"config/hierarchical": (args.hierarchical.lower() == 'on')})
    # 记录是否使用 region 信息
    swanlab.log({"config/region": (args.region.lower() == 'on')})
    # 记录流水线旁证的总体占比（设计图）
    try:
        has_pipe_count = sum(int(getattr(p['design_graph'], 'has_pipeline', 0)) for p in pairs)
        swanlab.log({
            "dataset/has_pipeline_ratio": has_pipe_count / max(1, len(pairs))
        })
    except Exception:
        pass

    # 限制配对数量（调试用）
    if args.max_pairs and len(pairs) > args.max_pairs:
        pairs = pairs[:args.max_pairs]
        swanlab.log({"dataset/limited_pairs": args.max_pairs})
    
    # 保存配对信息
    pairs_info = []
    for pair in pairs:
        info = {
            'pair_id': pair['pair_id'],
            'source_name': pair['kernel_info']['source_name'],
            'algo_name': pair['kernel_info']['algo_name'],
            'design_id': pair['design_info']['design_id'],
            'pragma_count': pair['pragma_info']['pragma_count'],
            'performance_delta': pair['performance_delta'],
            # 记录图级流水相关旁证（设计图）
            'design_has_pipeline': int(getattr(pair['design_graph'], 'has_pipeline', 0)),
            'design_pipeline_region_count': int(getattr(pair['design_graph'], 'pipeline_region_count', 0)),
            'design_avg_ii': float(getattr(pair['design_graph'], 'avg_ii', 0.0)),
            'design_max_pipe_depth': int(getattr(pair['design_graph'], 'max_pipe_depth', 0)),
            'design_pipeline_components_present': int(getattr(pair['design_graph'], 'pipeline_components_present', 0)),
            'design_pipeline_signals_present': int(getattr(pair['design_graph'], 'pipeline_signals_present', 0)),
            # kernel 旁证（便于差分分析）
            'kernel_has_pipeline': int(getattr(pair['kernel_graph'], 'has_pipeline', 0)),
            'kernel_pipeline_region_count': int(getattr(pair['kernel_graph'], 'pipeline_region_count', 0)),
            'kernel_avg_ii': float(getattr(pair['kernel_graph'], 'avg_ii', 0.0)),
            'kernel_max_pipe_depth': int(getattr(pair['kernel_graph'], 'max_pipe_depth', 0)),
        }
        pairs_info.append(info)
    
    pairs_path = os.path.join(output_dir, 'kernel_design_pairs.json')
    with open(pairs_path, 'w') as f:
        json.dump(pairs_info, f, indent=2)
    
    # 不再上传表格形式的配对信息，保留 JSON 文件以便线下查看
    
    # 创建数据集
    dataset = E2EDifferentialDataset(pairs, args.target_metric)
    ood_dataset = E2EDifferentialDataset(ood_pairs, args.target_metric) if ood_pairs else None
    
    # 数据集划分 - 8:1:1随机划分
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    valid_size = int(0.1 * total_size)
    test_size = total_size - train_size - valid_size
    
    indices = torch.randperm(total_size)
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:train_size + valid_size]
    test_indices = indices[train_size + valid_size:]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(dataset, valid_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    ood_test_dataset = ood_dataset  if ood_dataset is not None else None
    
    # 记录数据划分信息
    swanlab.log({
        "dataset/train_size": len(train_dataset),
        "dataset/valid_size": len(valid_dataset), 
        "dataset/test_size": len(test_dataset)
    })
    
    # 不再以表格形式记录数据集划分；仅保留标量日志
    
    # 创建数据加载器
    loader_workers = max(0, int(args.loader_workers))
    persistent = (args.persistent_workers.lower() == 'true')
    loader_common_kwargs = {
        'collate_fn': differential_collate_fn,
        'num_workers': loader_workers,
        'pin_memory': (args.pin_memory.lower() == 'true'),
        'persistent_workers': (persistent if loader_workers > 0 else False)
    }
    # 为 DataLoader 指定 spawn 上下文，确保 Ctrl+C 能可靠终止子进程
    if loader_workers > 0:
        try:
            loader_common_kwargs['multiprocessing_context'] = 'spawn'
        except Exception:
            pass
    if loader_workers > 0 and int(args.prefetch_factor) > 0:
        loader_common_kwargs['prefetch_factor'] = int(args.prefetch_factor)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        **loader_common_kwargs
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False,
        **loader_common_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        **loader_common_kwargs
    )
    ood_test_loader = None
    if ood_test_dataset is not None:
        ood_test_loader = torch.utils.data.DataLoader(
            ood_test_dataset, batch_size=args.batch_size, shuffle=False,
            **loader_common_kwargs
        )
    
    # ==================== 创建模型 ====================
    
    # 获取特征维度
    sample_pair = dataset[0]
    node_dim = sample_pair['kernel_graph'].x.size(1)
    
    # 创建模型
    differential_mode = args.differential.lower() == 'true'
    model = SimpleDifferentialGNN(
        node_dim=node_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        differential=differential_mode,
        dropout=args.dropout,
        target_metric=args.target_metric,
        gnn_type=args.gnn_type
    ).to(device)
    
    # 优化器和调度器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=15)
    
    # 学习率工具函数
    def _set_optimizer_lr(optim_obj, lr_value):
        for group in optim_obj.param_groups:
            group['lr'] = lr_value
    
    def _get_optimizer_lr(optim_obj):
        return optim_obj.param_groups[0]['lr']
    
    # 记录模型信息
    model_params = sum(p.numel() for p in model.parameters())
    swanlab.log({
        "model/node_dim": node_dim,
        "model/total_params": model_params,
        "model/hidden_dim": args.hidden_dim,
        "model/num_layers": args.num_layers
    })
    
    # ==================== 训练循环 ====================
    
    train_losses = []
    valid_losses = []
    test_losses = []
    valid_metrics_history = []
    test_metrics_history = []
    ood_metrics_history = []
    lr_history = []
    
    best_valid_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        # 线性 warmup（仅在前 warmup_epochs 个 epoch 生效）
        if args.warmup_epochs and epoch <= args.warmup_epochs:
            warmup_factor = float(epoch) / float(max(1, args.warmup_epochs))
            current_warmup_lr = max(args.lr * warmup_factor, args.min_lr)
            _set_optimizer_lr(optimizer, current_warmup_lr)
            swanlab.log({"optimizer/warmup_factor": warmup_factor})
        
        # 训练
        train_loss = train_epoch(
            model, device, train_loader, optimizer,
            grad_accum_steps=args.grad_accum_steps
        )
        train_losses.append(train_loss)
        
        # 验证
        valid_metrics = evaluate_model(
            model, valid_loader, device, args.target_metric
        )
        valid_loss = valid_metrics['avg_loss']
        valid_losses.append(valid_loss)
        valid_metrics_history.append(valid_metrics)
        
        # ID 测试
        id_test_metrics = evaluate_model(
            model, test_loader, device, args.target_metric
        )
        id_test_loss = id_test_metrics['avg_loss']
        test_losses.append(id_test_loss)
        test_metrics_history.append(id_test_metrics)
        
        # OOD 测试（可选）
        ood_test_metrics = None
        if ood_test_loader is not None:
            ood_test_metrics = evaluate_model(
                model, ood_test_loader, device, args.target_metric
            )
            ood_metrics_history.append(ood_test_metrics)
        
        # 学习率调度（warmup 期间不触发 ReduceLROnPlateau）
        if not args.warmup_epochs or epoch > args.warmup_epochs:
            scheduler.step(valid_loss)
            # 下限保护，避免 lr 过小
            try:
                cur_lr = scheduler.get_last_lr()[0]
            except Exception:
                cur_lr = _get_optimizer_lr(optimizer)
            if cur_lr < args.min_lr:
                _set_optimizer_lr(optimizer, args.min_lr)
        
        try:
            current_lr = scheduler.get_last_lr()[0]
        except Exception:
            current_lr = _get_optimizer_lr(optimizer)
        lr_history.append(current_lr)
        
        # 记录到SwanLab（ID test / OOD test 区分）
        swanlab.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "valid/loss": valid_loss,
            "valid/delta_mae": valid_metrics.get('delta_mae', 0),
            "valid/delta_rmse": valid_metrics.get('delta_rmse', 0),
            "valid/delta_ulti_rmse": valid_metrics.get('delta_ulti_rmse', 0),
            "valid/delta_r2": valid_metrics.get('delta_r2', 0),
            "id_test/loss": id_test_loss,
            "id_test/delta_mae": id_test_metrics.get('delta_mae', 0),
            "id_test/delta_rmse": id_test_metrics.get('delta_rmse', 0),
            "id_test/delta_ulti_rmse": id_test_metrics.get('delta_ulti_rmse', 0),
            "id_test/delta_r2": id_test_metrics.get('delta_r2', 0),
            "optimizer/lr": current_lr,
            **({
                "ood_test/loss": ood_test_metrics.get('avg_loss', 0),
                "ood_test/delta_mae": ood_test_metrics.get('delta_mae', 0),
                "ood_test/delta_rmse": ood_test_metrics.get('delta_rmse', 0),
                "ood_test/delta_ulti_rmse": ood_test_metrics.get('delta_ulti_rmse', 0),
                "ood_test/delta_r2": ood_test_metrics.get('delta_r2', 0)
            } if ood_test_metrics is not None else {})
        })
        
        # 保存最佳模型（基于验证集）
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            
            model_path = os.path.join(output_dir, f'best_e2e_delta_{args.target_metric}_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss': valid_loss,
                'id_test_metrics': id_test_metrics,
                'ood_test_metrics': ood_test_metrics,
                'args': vars(args)
            }, model_path)
            
            swanlab.log({"best_model/epoch": epoch, "best_model/valid_loss": valid_loss})
    
    # ==================== 保存结果 ====================
    
    # 最佳epoch的结果
    best_ood_metrics = None
    if test_metrics_history:
        best_test = test_metrics_history[best_epoch - 1]
        
        # 记录最终结果到SwanLab
        swanlab.log({
            "final/best_epoch": best_epoch,
            "final/best_valid_loss": best_valid_loss,
            "final/id_delta_mae": best_test.get('delta_mae', 0),
            "final/id_delta_rmse": best_test.get('delta_rmse', 0),
            "final/id_delta_ulti_rmse": best_test.get('delta_ulti_rmse', 0),
            "final/id_delta_r2": best_test.get('delta_r2', 0)
        })
        
        # 如果 OOD 可用，也记录最佳 epoch 对应的 OOD 指标（同一 epoch 下）
        if 'ood_test_loader' in locals() and ood_test_loader is not None:
            best_ood_metrics = evaluate_model(model, ood_test_loader, device, args.target_metric)
            swanlab.log({
                "final/ood_delta_mae": best_ood_metrics.get('delta_mae', 0),
                "final/ood_delta_rmse": best_ood_metrics.get('delta_rmse', 0),
                "final/ood_delta_ulti_rmse": best_ood_metrics.get('delta_ulti_rmse', 0),
                "final/ood_delta_r2": best_ood_metrics.get('delta_r2', 0)
            })
        
        # 创建最佳epoch的预测vs真实散点图（仅 ID test ）
        if 'delta_preds' in best_test and 'delta_true' in best_test:
            _create_prediction_plots(best_test, args.target_metric, output_dir)
        
        print(f"训练完成! 最佳模型 ({args.target_metric.upper()}):")
        print(f"  Epoch: {best_epoch}")
        print(f"  [ID] 差值MAE: {best_test.get('delta_mae', 0):.6f}")
        print(f"  [ID] 差值RMSE: {best_test.get('delta_rmse', 0):.6f}")
        print(f"  [ID] 差值ulti-RMSE: {best_test.get('delta_ulti_rmse', 0):.8f}")
        print(f"  [ID] 差值R²: {best_test.get('delta_r2', 0):.4f}")
        if 'ood_test_loader' in locals() and ood_test_loader is not None:
            if best_ood_metrics is not None:
                print(f"  [OOD] 差值MAE: {best_ood_metrics.get('delta_mae', 0):.6f}")
                print(f"  [OOD] 差值RMSE: {best_ood_metrics.get('delta_rmse', 0):.6f}")
                print(f"  [OOD] 差值ulti-RMSE: {best_ood_metrics.get('delta_ulti_rmse', 0):.8f}")
                print(f"  [OOD] 差值R²: {best_ood_metrics.get('delta_r2', 0):.4f}")
    
    # 保存训练历史 - 清理tensor数据避免JSON序列化错误
    clean_valid_history = []
    clean_test_history = []
    
    for metrics in valid_metrics_history:
        clean_metrics = {k: v for k, v in metrics.items() if not isinstance(v, torch.Tensor)}
        clean_valid_history.append(clean_metrics)
    
    for metrics in test_metrics_history:
        clean_metrics = {k: v for k, v in metrics.items() if not isinstance(v, torch.Tensor)}
        clean_test_history.append(clean_metrics)
    
    results = {
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'id_test_losses': test_losses,
        'valid_metrics_history': clean_valid_history,
        'id_test_metrics_history': clean_test_history,
        'ood_test_metrics_history': (
            [
                {k: v for k, v in m.items() if not isinstance(v, torch.Tensor)}
                for m in ood_metrics_history
            ] if ood_metrics_history else []
        ),
        'best_epoch': best_epoch,
        'best_valid_loss': best_valid_loss,
        'args': vars(args),
        'num_pairs': len(pairs)
    }
    
    results_path = os.path.join(output_dir, 'e2e_differential_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 创建训练曲线图 (仅四张图 -> 2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'E2E Differential Learning - {args.target_metric.upper()}', fontsize=16)
    
    epochs = range(1, len(train_losses) + 1)
    
    # 合并 Loss 与 MAE（因为 loss == MAE）: 展示 train(valid loss=mae), valid/test delta_mae 三条曲线
    if valid_metrics_history:
        delta_mae_valid = [m.get('delta_mae', 0) for m in valid_metrics_history]
        delta_mae_id_test = [m.get('delta_mae', 0) for m in test_metrics_history]
        delta_mae_ood_test = [m.get('delta_mae', 0) for m in ood_metrics_history] if ood_metrics_history else []
    else:
        delta_mae_valid, delta_mae_id_test, delta_mae_ood_test = [], [], []

    axes[0, 0].plot(epochs, train_losses, color='blue', label='Train (Loss=MAE)', linewidth=2)
    if delta_mae_valid:
        axes[0, 0].plot(epochs, delta_mae_valid, color='orange', label='Valid MAE', linewidth=2)
    if test_metrics_history:
        axes[0, 0].plot(epochs, delta_mae_id_test, color='purple', label='ID Test MAE', linewidth=2)
    if delta_mae_ood_test:
        axes[0, 0].plot(epochs, delta_mae_ood_test, color='red', label='OOD Test MAE', linewidth=2)
    
    axes[0, 0].axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label='Best Epoch')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MAE (Loss)')
    axes[0, 0].set_title('MAE (Loss) - Train / Valid / ID Test / OOD Test')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 差值RMSE (放在 (0,1))
    if valid_metrics_history:
        delta_rmse_valid = [m.get('delta_rmse', 0) for m in valid_metrics_history]
        delta_rmse_id_test = [m.get('delta_rmse', 0) for m in test_metrics_history]
        delta_rmse_ood_test = [m.get('delta_rmse', 0) for m in ood_metrics_history] if ood_metrics_history else []
        axes[0, 1].plot(epochs, delta_rmse_valid, 'cyan', label='Valid Delta RMSE', linewidth=2)
        axes[0, 1].plot(epochs, delta_rmse_id_test, 'magenta', label='ID Test Delta RMSE', linewidth=2)
        if delta_rmse_ood_test:
            axes[0, 1].plot(epochs, delta_rmse_ood_test, color='darkred', label='OOD Test Delta RMSE', linewidth=2)
        axes[0, 1].axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Delta RMSE')
        axes[0, 1].set_title('Delta RMSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # 差值R2曲线 (放在 (1,0))
    if valid_metrics_history:
        delta_r2_valid = [m.get('delta_r2', 0) for m in valid_metrics_history]
        delta_r2_id_test = [m.get('delta_r2', 0) for m in test_metrics_history]
        delta_r2_ood_test = [m.get('delta_r2', 0) for m in ood_metrics_history] if ood_metrics_history else []
        axes[1, 0].plot(epochs, delta_r2_valid, 'brown', label='Valid Delta R²', linewidth=2)
        axes[1, 0].plot(epochs, delta_r2_id_test, 'pink', label='ID Test Delta R²', linewidth=2)
        if delta_r2_ood_test:
            axes[1, 0].plot(epochs, delta_r2_ood_test, color='firebrick', label='OOD Test Delta R²', linewidth=2)
        axes[1, 0].axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].set_title('Delta R²')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # 学习率曲线 (放在 (1,1))
    if lr_history:
        axes[1, 1].plot(epochs, lr_history, 'green', label='Learning Rate', linewidth=2)
        axes[1, 1].axvline(x=best_epoch, color='blue', linestyle='--', alpha=0.7, label='Best Epoch')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'e2e_differential_{args.target_metric}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 上传图片到SwanLab
    swanlab.log({"training_curves": swanlab.Image(plot_path)})

    
    # 完成实验
    swanlab.finish()
    
    print(f"结果保存在: {output_dir}")
    print(f"SwanLab实验: HLS-Differential-Learning")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("检测到中断(KeyboardInterrupt)，正在清理并退出...")
        try:
            _graceful_cleanup()
        except Exception:
            pass
        sys.exit(130)
