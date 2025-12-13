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
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, PNAConv
from torch_geometric.nn import RGCNConv, FastRGCNConv, BatchNorm, GINConv
from torch_geometric.utils import degree
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
from swanlab.data.modules.custom_charts.echarts import Scatter, options as opts
from swanlab.data.modules.custom_charts import Echarts
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import signal
import atexit
import gc
import sys
import multiprocessing as mp

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    mp.set_sharing_strategy("file_descriptor")  # avoid /dev/shm mmap errors
except Exception:
    pass

_shutdown_requested = False


def log_status(message: str, swan_key: str = "status/message", step: Optional[int] = None):
    """Print a status message and mirror it to SwanLab (as text; optional numeric step)."""
    print(message, flush=True)
    try:
        payload = {swan_key: swanlab.Text(message)}
        if step is not None:
            payload["status/step"] = float(step)
        swanlab.log(payload)
    except Exception:
        pass

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


def _configure_mp(start_method: str):
    """Configure multiprocessing start method early to control DataLoader worker behavior."""
    current = mp.get_start_method(allow_none=True)
    if current == start_method:
        return
    try:
        mp.set_start_method(start_method, force=True)
    except Exception as exc:
        log_status(f"[MP] Failed to set start_method={start_method}: {exc}")


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

# 可用资源数量（用于计算ulti-RMSE）
AVAILABLE_RESOURCES = { 
    'dsp': 9024,
    'lut': 1303680,
    'ff': 2607360,
    'bram_18k': 4032
}


LOSS_FUNCTION_REGISTRY = {
    'mae': F.l1_loss,
    'mse': F.mse_loss,
    'smoothl1': F.smooth_l1_loss,
    'smooth_l1': F.smooth_l1_loss  # alias for convenience
}


def resolve_loss_function(loss_type: str):
    """Return a PyTorch loss callable based on a string identifier."""
    key = (loss_type or 'mae').lower()
    if key not in LOSS_FUNCTION_REGISTRY:
        supported = ', '.join(sorted({k for k in LOSS_FUNCTION_REGISTRY if '_' not in k}))
        raise ValueError(f"Unsupported loss_type '{loss_type}'. Supported losses: {supported}")
    return LOSS_FUNCTION_REGISTRY[key]


# ============================================================================
# Data Collection and Pairing
# ============================================================================

class E2EDifferentialProcessor:
    """端到端差值数据处理器"""
    
    def __init__(self, kernel_base_dir: str, design_base_dir: str, output_dir: str,
                 cache_root: str = "./graph_cache", rebuild_cache: bool = False, hierarchical: bool = False, region: bool = False, max_workers: Optional[int] = None,
                 use_code_feature: bool = False, code_model_path: Optional[str] = None, code_cache_root: Optional[str] = None,
                 code_pooling: str = "last_token", code_max_length: int = 2048, code_normalize: bool = True, code_batch_size: int = 8):
        self.kernel_base_dir = kernel_base_dir
        self.design_base_dir = design_base_dir
        self.output_dir = output_dir
        self.rebuild_cache = rebuild_cache
        self.hierarchical = hierarchical  # 是否启用分层区域节点
        self.region = region  # 是否启用 region 信息
        self.use_code_feature = bool(use_code_feature)
        self.code_model_path = code_model_path
        self.code_pooling = code_pooling
        self.code_max_length = code_max_length
        self.code_normalize = bool(code_normalize)
        self.code_batch_size = max(1, int(code_batch_size) if code_batch_size is not None else 8)
        self.code_cache_root = code_cache_root or cache_root
        self.code_cache_dir = os.path.join(self.code_cache_root, "code_embeddings")
        os.makedirs(self.code_cache_dir, exist_ok=True)
        self._code_embedder = None
        
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
        code_tag = ""
        if self.use_code_feature:
            code_tag = f"|code=1|pool={self.code_pooling}|mlen={self.code_max_length}|norm={'1' if self.code_normalize else '0'}|model={self.code_model_path or 'none'}"
        cache_key_src = f"kb={kb}|db={db}|hier={'1' if self.hierarchical else '0'}|region={'1' if self.region else '0'}|{feature_version}{code_tag}"
        cache_key = hashlib.md5(cache_key_src.encode("utf-8")).hexdigest()[:12]
        self.graph_cache_dir = os.path.join(cache_root, cache_key)
        os.makedirs(self.graph_cache_dir, exist_ok=True)
        self.pairs_dir = os.path.join(self.graph_cache_dir, "pairs")
        self.index_path = os.path.join(self.graph_cache_dir, "pairs_index.jsonl")
        os.makedirs(self.pairs_dir, exist_ok=True)
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
                        "max_workers": int(self.max_workers),
                        "use_code_feature": self.use_code_feature,
                        "code_model_path": self.code_model_path,
                        "code_pooling": self.code_pooling,
                        "code_max_length": int(self.code_max_length),
                        "code_normalize": self.code_normalize,
                        "code_batch_size": int(self.code_batch_size),
                        "code_cache_dir": self.code_cache_dir
                    }, mf, indent=2)
        except Exception:
            pass
        
        # 静默初始化，减少console输出
        pass

    def _get_code_embedder(self):
        """懒加载代码嵌入模型，仅在启用 code 特征时使用。"""
        if self._code_embedder is not None:
            return self._code_embedder
        if not self.code_model_path:
            raise ValueError("use_code_feature=True 时必须提供 --code_model_path")
        try:
            from LLM_embedding import LLMEmbedder
        except Exception as exc:
            raise ImportError(f"无法导入 LLMEmbedder，请检查 LLM_embedding.py 是否可用: {exc}")
        self._code_embedder = LLMEmbedder(
            self.code_model_path,
            pooling=self.code_pooling,
            max_length=int(self.code_max_length),
            normalize=self.code_normalize
        )
        return self._code_embedder

    def _read_design_code(self, design_path: Path) -> Tuple[Optional[str], Optional[str]]:
        """读取 design 目录中的 .c/.cpp 源码并返回拼接字符串及哈希。"""
        code_files = sorted(
            [p for p in design_path.rglob("*") if p.suffix in (".c", ".cpp")],
            key=lambda p: p.name
        )
        if not code_files:
            return None, None
        contents: List[str] = []
        for code_file in code_files:
            try:
                with open(code_file, 'r', encoding='utf-8', errors='ignore') as cf:
                    contents.append(cf.read())
            except Exception:
                continue
        if not contents:
            return None, None
        code_text = "\n\n".join(contents)
        code_hash = hashlib.md5(code_text.encode("utf-8")).hexdigest()
        return code_text, code_hash

    def _code_cache_path(self, design_path: Path, code_hash: str) -> str:
        """生成代码嵌入缓存路径。"""
        cache_key_src = f"path={design_path}|hash={code_hash}|model={self.code_model_path}|pool={self.code_pooling}|mlen={self.code_max_length}|norm={'1' if self.code_normalize else '0'}"
        cache_key = hashlib.md5(cache_key_src.encode("utf-8")).hexdigest()
        return os.path.join(self.code_cache_dir, f"{cache_key}.pt")

    def _load_cached_code_embedding(self, cache_path: str) -> Optional[Tuple[torch.Tensor, str]]:
        try:
            cached = torch.load(cache_path, map_location='cpu')
            emb = cached.get('embedding')
            meta = cached.get('meta', {})
            code_hash = meta.get('code_hash')
            if emb is not None:
                return emb, code_hash
        except Exception:
            return None
        return None

    def _store_code_embeddings(self, batch_entries: List[Tuple[Path, str, str]]):
        """对一批 code 文本做嵌入并写入缓存。"""
        if not batch_entries:
            return
        embedder = self._get_code_embedder()
        texts = [entry[1] for entry in batch_entries]
        with torch.inference_mode():
            embeddings = embedder.encode(texts)
        for emb, (design_path, _, code_hash) in zip(embeddings, batch_entries):
            cache_path = self._code_cache_path(design_path, code_hash)
            meta = {
                "design_path": str(design_path),
                "code_hash": code_hash,
                "model_path": self.code_model_path,
                "pooling": self.code_pooling,
                "max_length": int(self.code_max_length),
                "normalize": self.code_normalize
            }
            try:
                torch.save({
                    "embedding": emb.detach().cpu(),
                    "meta": meta
                }, cache_path)
            except Exception as exc:
                print(f"保存代码嵌入失败 {design_path}: {exc}")

    def _precompute_design_code_embeddings(self, design_paths: List[Path]):
        """预先对未缓存的 design 源码做批量嵌入，避免训练时反复加载模型。"""
        if not self.use_code_feature:
            return
        unique_paths: List[Path] = []
        seen: set[str] = set()
        for p in design_paths:
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            unique_paths.append(p)

        to_encode: List[Tuple[Path, str, str]] = []
        for dpath in unique_paths:
            code_text, code_hash = self._read_design_code(dpath)
            if not code_text or not code_hash:
                continue
            cache_path = self._code_cache_path(dpath, code_hash)
            if os.path.exists(cache_path):
                continue
            to_encode.append((dpath, code_text, code_hash))

        if not to_encode:
            return

        log_status(f"[Step3.3] 预编码 design 源码嵌入: {len(to_encode)} 个样本（batch={self.code_batch_size}）")
        with tqdm(
            total=len(to_encode),
            desc="预编码设计源码",
            ncols=100,
            unit="design"
        ) as pbar:
            for i in range(0, len(to_encode), self.code_batch_size):
                batch_entries = to_encode[i:i + self.code_batch_size]
                self._store_code_embeddings(batch_entries)
                pbar.update(len(batch_entries))
        log_status(f"[Step3.3] 设计源码嵌入预编码完成，写入 {len(to_encode)} 个缓存")

    def _get_design_code_embedding(self, design_path: Path) -> Tuple[Optional[torch.Tensor], Optional[str]]:
        """获取单个 design 的代码嵌入（优先使用缓存）。"""
        if not self.use_code_feature:
            return None, None
        code_text, code_hash = self._read_design_code(design_path)
        if not code_text or not code_hash:
            return None, None
        cache_path = self._code_cache_path(design_path, code_hash)
        cached = self._load_cached_code_embedding(cache_path)
        if cached is not None:
            emb, _ = cached
            return emb, code_hash
        # 缓存不存在则即时编码并写入缓存
        try:
            embedder = self._get_code_embedder()
            with torch.inference_mode():
                emb = embedder.encode([code_text])[0]
            meta = {
                "design_path": str(design_path),
                "code_hash": code_hash,
                "model_path": self.code_model_path,
                "pooling": self.code_pooling,
                "max_length": int(self.code_max_length),
                "normalize": self.code_normalize
            }
            torch.save({
                "embedding": emb.detach().cpu(),
                "meta": meta
            }, cache_path)
            return emb.detach().cpu(), code_hash
        except Exception as exc:
            print(f"编码 design 代码失败 {design_path}: {exc}")
            return None, None
    
    def collect_all_data(self, limit: Optional[int] = None, materialize: bool = True) -> List[Dict]:
        """收集所有kernel-design配对数据（全局任务并发）"""
        # 检查是否存在缓存的配对数据
        if (not self.rebuild_cache) and os.path.exists(self.index_path):
            log_status(f"[Step3] 检测到缓存索引，直接加载配对数据: {self.index_path}")
            return self._load_cached_pairs(limit=limit, materialize=materialize)

        cache_file = os.path.join(self.graph_cache_dir, "paired_graphs.json")
        if (not self.rebuild_cache) and os.path.exists(cache_file):
            # 兼容旧格式：加载一次后转换为流式缓存
            log_status(f"[Step3] 检测到旧版缓存 {cache_file}，开始转换为新格式索引...")
            cached = self._load_cached_pairs_from_legacy(cache_file)
            self._save_cached_pairs(cached)
            try:
                os.remove(cache_file)
            except Exception:
                pass
            log_status("[Step3] 旧版缓存转换完成，重新加载新索引")
            return self._load_cached_pairs(limit=limit, materialize=materialize)

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
        
        log_status(f"[Step3.1] 开始扫描设计目录，准备构建图缓存: {design_base}")
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
        log_status(f"[Step3.1] 扫描完成，发现 {len(tasks)} 个kernel-design任务，开始构建图缓存与配对")
        
        # 预先编码设计源码，避免在并行过程中重复加载模型
        if self.use_code_feature:
            design_paths_for_code = [t[1] for t in tasks]
            log_status(f"[Step3.2] 开始预编码设计源码嵌入，共 {len(design_paths_for_code)} 个 design")
            self._precompute_design_code_embeddings(design_paths_for_code)

        def _run_with_executor(executor_cls):
            log_status(f"[Step3.4] 并行处理 {len(tasks)} 个任务构建缓存，max_workers={self.max_workers}")
            extra_kwargs = {}
            # 当系统有 CUDA 或已加载大模型时，使用 spawn 上下文避免 fork 后 CUDA 复用报错
            if executor_cls is ProcessPoolExecutor:
                try:
                    extra_kwargs["mp_context"] = mp.get_context("spawn")
                except Exception:
                    pass
            with executor_cls(max_workers=self.max_workers, **extra_kwargs) as executor:
                future_to_task = {executor.submit(self._process_task, task): task for task in tasks}
                
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

        try:
            _run_with_executor(ProcessPoolExecutor)
        except Exception as exc:
            print(f"进程池并行构建缓存失败，改用线程池: {exc}")
            pairs.clear()
            _run_with_executor(ThreadPoolExecutor)
        
        # 缓存配对数据
        if pairs:
            self._save_cached_pairs(pairs)
            log_status(f"[Step3.5] 图缓存与配对完成，共生成 {len(pairs)} 条记录，已写入缓存目录 {self.graph_cache_dir}")

        if materialize:
            if limit is not None and len(pairs) > limit:
                return pairs[:limit]
            return pairs

        return self._load_cached_pairs(limit=limit, materialize=False)

    def _process_task(self, task: Tuple[Dict, Path, str, str, str]) -> Optional[Dict]:
        """单个kernel-design配对的处理任务（供并行执行使用）"""
        kernel_data, design_dir, source_name, algo_name, design_id = task
        try:
            design_data = self._collect_design_data(design_dir, source_name, algo_name, design_id)
            if design_data:
                return self._create_pair(kernel_data, design_data)
            return None
        except Exception:
            return None
    
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

            design_code_embedding = None
            code_hash = None
            if self.use_code_feature:
                design_code_embedding, code_hash = self._get_design_code_embedding(design_path)
            
            return {
                'type': 'design',
                'source_name': source_name,
                'algo_name': algo_name,
                'design_id': design_id,
                'base_path': str(design_path),
                'performance': perf_data,
                'graph': graph_data,
                'pragma_info': pragma_info,
                'design_code_embedding': design_code_embedding,
                'code_hash': code_hash
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
            
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            
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

            for data in datas:
                x = data.x
                ei = data.edge_index + node_offset
                ea = data.edge_attr if data.edge_attr is not None else torch.zeros((ei.size(1), 2), dtype=torch.float)

                all_x.append(x)
                all_edge_index.append(ei)
                all_edge_attr.append(ea)

                node_offset += x.size(0)

            # 合并
            x_cat = torch.cat(all_x, dim=0)
            edge_index_cat = torch.cat(all_edge_index, dim=1)
            edge_attr_cat = torch.cat(all_edge_attr, dim=0) if all_edge_attr else None

            data_merged = Data(x=x_cat, edge_index=edge_index_cat, edge_attr=edge_attr_cat)
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
                'design_code_embedding': design_data.get('design_code_embedding'),
                'code_hash': design_data.get('code_hash'),
                'kernel_info': kernel_data,
                'design_info': design_data
            }
            
        except Exception as e:
            print(f"创建配对失败: {e}")
            return None
    
    def _save_cached_pairs(self, pairs: List[Dict]):
        """保存配对数据到缓存"""
        # 清理旧缓存文件，避免陈旧数据残留
        try:
            for existing in Path(self.pairs_dir).glob("*.pt"):
                existing.unlink()
        except Exception:
            pass

        tmp_index_path = self.index_path + ".tmp"
        with open(tmp_index_path, 'w', encoding='utf-8') as index_f:
            for pair in tqdm(pairs, desc="缓存图数据"):
                # 确保图数据在 CPU 上存储
                kernel_graph = pair['kernel_graph'].to('cpu')
                design_graph = pair['design_graph'].to('cpu')
                design_code_embedding = None
                if pair.get('design_code_embedding') is not None:
                    try:
                        design_code_embedding = pair['design_code_embedding'].detach().cpu()
                    except Exception:
                        design_code_embedding = pair['design_code_embedding']

                payload = {
                    'pair_id': pair['pair_id'],
                    'kernel_graph': kernel_graph,
                    'design_graph': design_graph,
                    'performance_delta': pair['performance_delta'],
                    'pragma_info': pair['pragma_info'],
                    'design_code_embedding': design_code_embedding,
                    'code_hash': pair.get('code_hash'),
                    'kernel_info': {k: v for k, v in pair['kernel_info'].items() if k != 'graph'},
                    'design_info': {k: v for k, v in pair['design_info'].items() if k not in ('graph', 'design_code_embedding')}
                }
                is_valid = self._is_pair_payload_valid(payload)

                pair_file = f"pairs/{pair['pair_id']}.pt"
                torch.save(payload, os.path.join(self.graph_cache_dir, pair_file))

                meta = {
                    'pair_id': pair['pair_id'],
                    'file': pair_file,
                    'design_base_path': payload['design_info'].get('base_path'),
                    'kernel_base_path': payload['kernel_info'].get('base_path'),
                    'pragma_count': int(payload['pragma_info'].get('pragma_count', 0)),
                    'design_num_nodes': int(design_graph.num_nodes),
                    'kernel_num_nodes': int(kernel_graph.num_nodes),
                    'has_code': bool(design_code_embedding is not None),
                    'code_hash': payload.get('code_hash'),
                    'is_valid': bool(is_valid)
                }
                index_f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        os.replace(tmp_index_path, self.index_path)

    def _is_performance_within_available(self, perf: Optional[Dict]) -> bool:
        """检查性能字典中的资源是否不超过可用上限"""
        if not perf:
            return True
        for raw_key, raw_value in perf.items():
            if not isinstance(raw_key, str):
                continue
            normalized_key = raw_key.lower()
            if normalized_key not in AVAILABLE_RESOURCES:
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if value > 0.1 * AVAILABLE_RESOURCES[normalized_key]:
                return False
        return True

    def _is_pair_payload_valid(self, payload: Dict) -> bool:
        """判定缓存样本是否满足资源限制"""
        kernel_perf = payload.get('kernel_info', {}).get('performance')
        design_perf = payload.get('design_info', {}).get('performance')
        if not self._is_performance_within_available(kernel_perf):
            return False
        if not self._is_performance_within_available(design_perf):
            return False
        return True

    def _load_cached_pairs(self, limit: Optional[int] = None, materialize: bool = True) -> List[Dict]:
        """从缓存加载配对数据（流式）"""
        records: List[Dict] = []
        filtered_out = 0

        if os.path.exists(self.index_path):
            with open(self.index_path, 'r', encoding='utf-8') as index_f:
                for line in index_f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    if materialize:
                        pair_path = os.path.join(self.graph_cache_dir, entry['file'])
                        if not os.path.exists(pair_path):
                            continue
                        try:
                            payload = torch.load(pair_path, map_location='cpu')
                            if not self._is_pair_payload_valid(payload):
                                filtered_out += 1
                                continue
                            records.append(payload)
                        except Exception:
                            continue
                    else:
                        pair_path = os.path.join(self.graph_cache_dir, entry['file'])
                        if not os.path.exists(pair_path):
                            continue
                        entry['file'] = pair_path
                        is_valid = entry.get('is_valid')
                        if is_valid is False:
                            filtered_out += 1
                            continue
                        if is_valid is None:
                            try:
                                payload = torch.load(pair_path, map_location='cpu')
                            except Exception:
                                filtered_out += 1
                                continue
                            if not self._is_pair_payload_valid(payload):
                                filtered_out += 1
                                continue
                        records.append(entry)

                    if limit is not None and len(records) >= limit:
                        break
            if filtered_out > 0:
                print(f"过滤资源超限的缓存样本: {filtered_out}")
            return records

        # 兜底：无新格式索引，返回空列表
        return records

    def _load_cached_pairs_from_legacy(self, cache_file: str) -> List[Dict]:
        """兼容旧版 JSON 缓存格式"""
        with open(cache_file, 'r', encoding='utf-8') as f:
            cached_pairs = json.load(f)

        pairs = []
        for cached_pair in tqdm(cached_pairs, desc="加载旧缓存图数据"):
            try:
                k_data = cached_pair['kernel_graph_data']
                kernel_graph = Data(
                    x=torch.tensor(k_data['x'], dtype=torch.float),
                    edge_index=torch.tensor(k_data['edge_index'], dtype=torch.long),
                    edge_attr=torch.tensor(k_data['edge_attr'], dtype=torch.float) if k_data['edge_attr'] else None,
                    y=torch.tensor(k_data['y'], dtype=torch.float)
                )

                d_data = cached_pair['design_graph_data']
                design_graph = Data(
                    x=torch.tensor(d_data['x'], dtype=torch.float),
                    edge_index=torch.tensor(d_data['edge_index'], dtype=torch.long),
                    edge_attr=torch.tensor(d_data['edge_attr'], dtype=torch.float) if d_data['edge_attr'] else None,
                    y=torch.tensor(d_data['y'], dtype=torch.float)
                )

                pair = {
                    'pair_id': cached_pair['pair_id'],
                    'kernel_graph': kernel_graph,
                    'design_graph': design_graph,
                    'performance_delta': cached_pair['performance_delta'],
                    'pragma_info': cached_pair['pragma_info'],
                    'design_code_embedding': cached_pair.get('design_code_embedding'),
                    'code_hash': cached_pair.get('code_hash'),
                    'kernel_info': cached_pair['kernel_info'],
                    'design_info': cached_pair['design_info']
                }

                if not self._is_pair_payload_valid(pair):
                    continue

                pairs.append(pair)

            except Exception:
                continue

        return pairs


# ============================================================================
# Differential GNN Model
# ============================================================================

class SimpleDifferentialGNN(nn.Module):
    """支持多种GNN架构的差值学习模型"""
    
    def __init__(self, node_dim: int, hidden_dim: int = 128, num_layers: int = 3, 
                 dropout: float = 0.1, target_metric: str = 'dsp', differential: bool = True,
                 gnn_type: str = 'gcn', kernel_baseline: str = 'learned',
                 pna_deg: Optional[torch.Tensor] = None, edge_dim: Optional[int] = None,
                 use_code_feature: bool = False, code_dim: Optional[int] = None):
        super().__init__()
        self.target_metric = target_metric
        self.hidden_dim = hidden_dim
        self.differential = differential
        self.gnn_type = gnn_type.lower()
        self.kernel_baseline = kernel_baseline.lower()
        self.edge_dim = max(0, edge_dim or 0)
        self.use_code_feature = bool(use_code_feature)
        self.code_dim = code_dim
        if self.use_code_feature and (self.code_dim is None or self.code_dim <= 0):
            raise ValueError("use_code_feature=True 需要提供有效的 code_dim")

        # 节点编码器
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        
        # GNN层 - 支持多种架构
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.edge_encoder: Optional[nn.Module] = None
        pna_deg_tensor: Optional[torch.Tensor] = None
        if self.gnn_type == 'pna':
            if pna_deg is None or pna_deg.numel() == 0:
                raise ValueError("PNA 架构需要提供非空的度直方图 (pna_deg)。")
            pna_deg_tensor = pna_deg.detach().clone().to(torch.long)
            if self.edge_dim > 0:
                self.edge_encoder = nn.Linear(self.edge_dim, hidden_dim)
        
        for _ in range(num_layers):
            if self.gnn_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif self.gnn_type == 'gin':
                # GIN需要一个MLP作为参数
                # 添加LayerNorm以稳定梯度流，避免激活分布偏移
                # LayerNorm对特征维度归一化，不依赖batch大小，适合图数据
                # 模式: Linear -> LayerNorm -> ReLU -> Linear
                # 
                # 重要经验：如果不加LayerNorm，GIN在DSP任务上loss基本不下降，无法正常学习
                # 这是因为GIN的MLP缺少归一化导致梯度流不稳定，激活分布偏移
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),  # 归一化后激活，稳定第一层的梯度
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                    # 最后一层不添加LayerNorm，因为GIN层后还有全局LayerNorm
                )
                self.convs.append(GINConv(mlp))
            elif self.gnn_type == 'rgcn':
                # RGCN需要边类型数量，使用特征定义中的边类型维度自动获取
                self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations=get_edge_feature_dims()[0], num_bases=30))
            elif self.gnn_type == 'fast_rgcn':
                self.convs.append(FastRGCNConv(hidden_dim, hidden_dim, num_relations=get_edge_feature_dims()[0], num_bases=30))
            elif self.gnn_type == 'pna':
                aggregators = ['mean', 'min', 'max', 'std']
                scalers = ['identity', 'amplification', 'attenuation']
                edge_dim_param = hidden_dim if self.edge_encoder is not None else None
                self.convs.append(
                    PNAConv(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        aggregators=aggregators,
                        scalers=scalers,
                        deg=pna_deg_tensor,
                        edge_dim=edge_dim_param,
                        towers=1,
                        pre_layers=1,
                        post_layers=1,
                        divide_input=False
                    )
                )
            else:
                raise ValueError(f"不支持的GNN类型: {self.gnn_type}. 支持的类型: gcn, gin, rgcn, fast_rgcn, pna")
            
            # 为所有GNN类型添加归一化层（使用LayerNorm以避免单样本BatchNorm错误）
            from torch_geometric.nn import LayerNorm
            self.batch_norms.append(LayerNorm(hidden_dim))
        
        # 代码模态投影
        self.code_proj: Optional[nn.Module] = None
        if self.use_code_feature:
            self.code_proj = nn.Sequential(
                nn.LayerNorm(self.code_dim),
                nn.Linear(self.code_dim, hidden_dim)
            )
        
        def _make_head(input_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )

        if self.differential:
            if self.kernel_baseline not in ('learned', 'oracle'):
                raise ValueError("kernel_baseline 必须为 learned 或 oracle")
            # kernel 头只基于 kernel 表征
            if self.kernel_baseline == 'learned':
                self.kernel_head = _make_head(hidden_dim)
            # delta 头融合多种交互项
            delta_input_dim = hidden_dim * 2
            if self.use_code_feature:
                delta_input_dim += hidden_dim  # 设计代码模态
            self.delta_input_norm = nn.LayerNorm(delta_input_dim)
            self.delta_head = _make_head(delta_input_dim)
        else:
            # 直接预测头
            design_input_dim = hidden_dim + (hidden_dim if self.use_code_feature else 0)
            self.design_head = _make_head(design_input_dim)

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
            elif self.gnn_type == 'pna':
                edge_attr = None
                if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                    edge_attr = data.edge_attr.to(x.device, dtype=torch.float)
                    if self.edge_dim > 0 and edge_attr.size(1) != self.edge_dim:
                        raise ValueError(f"PNA 期望的边特征维度为 {self.edge_dim}, 但得到 {edge_attr.size(1)}")
                elif self.edge_dim > 0:
                    edge_attr = torch.zeros(
                        (data.edge_index.size(1), self.edge_dim),
                        dtype=torch.float,
                        device=x.device
                    )
                edge_features = self.edge_encoder(edge_attr) if (self.edge_encoder is not None and edge_attr is not None) else edge_attr
                x = conv(x, data.edge_index, edge_features)
            else:
                raise ValueError(f"不支持的GNN类型: {self.gnn_type}")
            
            # 应用BatchNorm和激活函数
            x = self.batch_norms[i](x)
            if i < len(self.convs) - 1:  # 最后一层不用ReLU
                x = F.relu(x)
        
        # 图级别池化
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
        # 改为 sum 池化以更贴近资源求和语义
        return global_add_pool(x, batch)
    
    def forward(self, kernel_graph, design_graph, pragma_count, design_code: Optional[torch.Tensor] = None):
        """前向传播 - 支持差分和直接预测两种模式"""
        # 编码图
        design_repr = self.encode_graph(design_graph)
        code_repr = None
        if self.use_code_feature:
            if design_code is None:
                raise ValueError("use_code_feature=True 时需要提供 design_code")
            code_repr = self.code_proj(design_code.float()) if self.code_proj is not None else design_code.float()

        if self.differential:
            # 差分模式：先预测 kernel，再预测 delta
            kernel_repr = self.encode_graph(kernel_graph)
            if self.kernel_baseline == 'learned':
                kernel_pred = self.kernel_head(kernel_repr)
            else:
                if not hasattr(kernel_graph, 'y'):
                    raise ValueError("oracle kernel 基线需要 kernel_graph.y 提供真实指标")
                kernel_pred = kernel_graph.y[:, self.metric_idx].unsqueeze(-1).to(kernel_repr.device)

            # 构建 delta 输入：包含 kernel/design 表征及其交互
            delta_parts = [kernel_repr, design_repr]
            if code_repr is not None:
                delta_parts.append(code_repr)
            delta_input = torch.cat(delta_parts, dim=-1)
            delta_input = self.delta_input_norm(delta_input) # add layer norm to delta input, avoid gradient explosion in DSP training.

            delta_pred = self.delta_head(delta_input)

            design_pred = kernel_pred + delta_pred

            return {
                'kernel_pred': kernel_pred,
                'delta_pred': delta_pred,
                'design_pred': design_pred
            }
        else:
            # 直接预测模式：只使用design
            design_parts = [design_repr]
            if code_repr is not None:
                design_parts.append(code_repr)
            design_input = torch.cat(design_parts, dim=-1)
            prediction = self.design_head(design_input)
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
            'design_code': pair.get('design_code_embedding'),
            'pair_id': pair['pair_id']
        }


def differential_collate_fn(batch):
    """自定义批处理函数"""
    kernel_graphs = [item['kernel_graph'] for item in batch]
    design_graphs = [item['design_graph'] for item in batch]
    design_codes = [item.get('design_code') for item in batch]
    
    kernel_perfs = torch.cat([item['kernel_perf'] for item in batch], dim=0)
    design_perfs = torch.cat([item['design_perf'] for item in batch], dim=0)
    perf_deltas = torch.cat([item['performance_delta'] for item in batch], dim=0)
    pragma_counts = torch.cat([item['pragma_count'] for item in batch], dim=0)
    
    # 使用PyG的Batch直接拼接，避免在collate中再启动DataLoader
    kernel_batch = Batch.from_data_list(kernel_graphs)
    design_batch = Batch.from_data_list(design_graphs)

    design_code_batch = None
    if any(dc is not None for dc in design_codes):
        first_dc = next((dc for dc in design_codes if dc is not None), None)
        if first_dc is not None:
            first_tensor = first_dc if isinstance(first_dc, torch.Tensor) else torch.tensor(first_dc)
            code_dim = first_tensor.shape[-1]
            code_tensors: List[torch.Tensor] = []
            for dc in design_codes:
                if dc is None:
                    code_tensors.append(torch.zeros(code_dim, dtype=first_tensor.dtype))
                else:
                    code_tensors.append(dc if isinstance(dc, torch.Tensor) else torch.tensor(dc))
            design_code_batch = torch.stack(code_tensors, dim=0)
    
    return {
        'kernel_graph': kernel_batch,
        'design_graph': design_batch,
        'kernel_perf': kernel_perfs,
        'design_perf': design_perfs,
        'performance_delta': perf_deltas,
        'pragma_count': pragma_counts,
        'design_code': design_code_batch
    }


# ============================================================================
# Target statistics & normalization
# ============================================================================


class RobustScaler:
    """Robust缩放器：使用median/IQR（退回std）进行归一化。"""

    def __init__(self, median: float, scale: float, p05: float, p95: float, p25: float, p75: float):
        self.median = float(median)
        self.scale = float(scale) if scale > 1e-12 else 1e-12
        self.p05 = float(p05)
        self.p95 = float(p95)
        self.p25 = float(p25)
        self.p75 = float(p75)

    def transform_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        median = torch.tensor(self.median, device=tensor.device, dtype=tensor.dtype)
        scale = torch.tensor(self.scale, device=tensor.device, dtype=tensor.dtype)
        return (tensor - median) / scale

    def to_dict(self) -> Dict[str, float]:
        return {
            "median": self.median,
            "scale": self.scale,
            "p05": self.p05,
            "p25": self.p25,
            "p75": self.p75,
            "p95": self.p95
        }


def _compute_basic_stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {}
    p05, p25, p50, p75, p95 = np.percentile(arr, [5, 25, 50, 75, 95])
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "median": float(p50),
        "p05": float(p05),
        "p25": float(p25),
        "p75": float(p75),
        "p95": float(p95)
    }


def _extract_metric_values(pairs: List[Dict], metric_idx: int, delta_key: str) -> Dict[str, List[float]]:
    kernel_vals: List[float] = []
    design_vals: List[float] = []
    delta_vals: List[float] = []
    for pair in pairs:
        kernel_vals.append(float(pair['kernel_graph'].y[0, metric_idx].item()))
        design_vals.append(float(pair['design_graph'].y[0, metric_idx].item()))
        delta_vals.append(float(pair['performance_delta'][delta_key]))
    return {"kernel": kernel_vals, "design": design_vals, "delta": delta_vals}


def _fit_robust_scaler(values: List[float]) -> RobustScaler:
    stats = _compute_basic_stats(values)
    scale = stats["p75"] - stats["p25"]
    if scale <= 1e-12:
        scale = stats["std"] if stats["std"] > 1e-12 else 1.0
    return RobustScaler(
        median=stats["median"],
        scale=scale,
        p05=stats["p05"],
        p25=stats["p25"],
        p75=stats["p75"],
        p95=stats["p95"]
    )


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, device, train_loader, optimizer, loss_fn=None, grad_accum_steps=1,
               max_grad_norm=1.0, use_tqdm: bool = False, progress_desc: Optional[str] = None,
               normalizers: Optional[Dict[str, RobustScaler]] = None):
    """训练一个epoch"""
    model.train()
    loss_fn = loss_fn or F.l1_loss
    loss_sum_gpu = torch.zeros(1, device=device)
    batch_count = 0
    
    optimizer.zero_grad(set_to_none=True)
    
    iterator = train_loader
    if use_tqdm:
        iterator = tqdm(train_loader, desc=progress_desc or "Train", ncols=100)

    for step_idx, batch in enumerate(iterator):
        # 移动到设备
        kernel_graph = batch['kernel_graph'].to(device, non_blocking=True)
        design_graph = batch['design_graph'].to(device, non_blocking=True)
        pragma_count = batch['pragma_count'].to(device, non_blocking=True)
        design_code = batch.get('design_code')
        if design_code is not None:
            design_code = design_code.to(device, non_blocking=True)
        
        # 前向传播
        predictions = model(kernel_graph, design_graph, pragma_count, design_code)
        
        # 计算损失
        targets = {
            'kernel_perf': batch['kernel_perf'].to(device, non_blocking=True),
            'design_perf': batch['design_perf'].to(device, non_blocking=True),
            'performance_delta': batch['performance_delta'].to(device, non_blocking=True)
        }

        def _norm(tensor: torch.Tensor, key: str) -> torch.Tensor:
            if normalizers and key in normalizers and normalizers[key] is not None:
                return normalizers[key].transform_tensor(tensor)
            return tensor
        
        if model.differential:
            design_pred = predictions['design_pred'].squeeze()
            delta_pred = predictions['delta_pred'].squeeze()
            kernel_pred = predictions['kernel_pred'].squeeze()

            target_design = targets['design_perf'].squeeze()
            target_delta = targets['performance_delta'].squeeze()
            target_kernel = targets['kernel_perf'].squeeze()

            design_loss = loss_fn(_norm(design_pred, 'design'), _norm(target_design, 'design'))
            delta_loss = loss_fn(_norm(delta_pred, 'delta'), _norm(target_delta, 'delta'))
            kernel_loss = loss_fn(_norm(kernel_pred, 'kernel'), _norm(target_kernel, 'kernel'))

            loss = kernel_loss + delta_loss + 0.2 * design_loss
        else:
            design_pred = predictions['direct_pred'].squeeze()
            target_design = targets['design_perf'].squeeze()
            loss = loss_fn(_norm(design_pred, 'design'), _norm(target_design, 'design'))
        
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


def evaluate_model(model, data_loader, device, target_metric='dsp', return_predictions=False, loss_fn=None,
                   normalizers: Optional[Dict[str, RobustScaler]] = None):
    """评估模型性能 - 返回设计绝对指标并可选记录差值诊断信息"""
    model.eval()
    loss_fn = loss_fn or F.l1_loss
    total_loss = 0
    batch_count = 0

    all_design_preds = []
    all_design_true = []
    all_delta_preds = []
    all_delta_true = []
    all_kernel_preds = []
    all_kernel_true = []

    with torch.no_grad():
        for batch in data_loader:
            # 移动到设备
            kernel_graph = batch['kernel_graph'].to(device, non_blocking=True)
            design_graph = batch['design_graph'].to(device, non_blocking=True)
            pragma_count = batch['pragma_count'].to(device, non_blocking=True)
            design_code = batch.get('design_code')
            if design_code is not None:
                design_code = design_code.to(device, non_blocking=True)

            # 前向传播
            predictions = model(kernel_graph, design_graph, pragma_count, design_code)

            # 准备目标值
            targets = {
                'kernel_perf': batch['kernel_perf'].to(device),
                'design_perf': batch['design_perf'].to(device),
                'performance_delta': batch['performance_delta'].to(device)
            }

            def _norm(tensor: torch.Tensor, key: str) -> torch.Tensor:
                if normalizers and key in normalizers and normalizers[key] is not None:
                    return normalizers[key].transform_tensor(tensor)
                return tensor

            if model.differential:
                # 差分模式：预测 kernel、delta，并组合成设计指标
                design_pred = predictions['design_pred'].squeeze()
                delta_pred = predictions['delta_pred'].squeeze()
                kernel_pred = predictions['kernel_pred'].squeeze()

                target_design = targets['design_perf'].squeeze()
                delta_true = targets['performance_delta'].squeeze()
                kernel_true = targets['kernel_perf'].squeeze()

                design_loss = loss_fn(_norm(design_pred, 'design'), _norm(target_design, 'design'))
                delta_loss = loss_fn(_norm(delta_pred, 'delta'), _norm(delta_true, 'delta'))
                kernel_loss = loss_fn(_norm(kernel_pred, 'kernel'), _norm(kernel_true, 'kernel'))
                total_batch_loss = kernel_loss + delta_loss + 0.2 * design_loss
            else:
                design_pred = predictions['direct_pred'].squeeze()
                target_design = targets['design_perf'].squeeze()
                delta_pred = None
                delta_true = None
                kernel_pred = None
                kernel_true = None

                design_loss = loss_fn(_norm(design_pred, 'design'), _norm(target_design, 'design'))
                total_batch_loss = design_loss

            total_loss += total_batch_loss.item()
            batch_count += 1

            # 收集预测结果（使用原始值，无归一化）
            all_design_preds.append(design_pred.detach().cpu())
            all_design_true.append(target_design.detach().cpu())
            if delta_pred is not None:
                all_delta_preds.append(delta_pred.detach().cpu())
            if delta_true is not None:
                all_delta_true.append(delta_true.detach().cpu())
            if kernel_pred is not None:
                all_kernel_preds.append(kernel_pred.detach().cpu())
            if kernel_true is not None:
                all_kernel_true.append(kernel_true.detach().cpu())

    # 计算评估指标
    avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')

    if all_design_preds:
        design_preds = torch.cat(all_design_preds, dim=0)
        design_true = torch.cat(all_design_true, dim=0)

        def calculate_r2(y_true, y_pred):
            ss_res = torch.sum((y_true - y_pred) ** 2)
            ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
            return (1 - ss_res / (ss_tot + 1e-8)).item()

        def calculate_mape(y_true, y_pred):
            y_true = y_true.view(-1)
            y_pred = y_pred.view(-1)
            mask = y_true != 0
            if mask.sum().item() == 0:
                return float('nan')
            mape = torch.mean(torch.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            return mape.item()

        design_mae = F.l1_loss(design_preds.squeeze(), design_true.squeeze()).item()
        design_rmse = torch.sqrt(F.mse_loss(design_preds.squeeze(), design_true.squeeze())).item()
        design_r2 = calculate_r2(design_true.squeeze(), design_preds.squeeze())
        design_mape = calculate_mape(design_true.squeeze(), design_preds.squeeze())

        metrics_dict = {
            'avg_loss': avg_loss,
            'design_mae': design_mae,
            'design_rmse': design_rmse,
            'design_r2': design_r2,
            'design_mape': design_mape
        }

        # 仅当目标资源存在时计算 ulti-RMSE
        if target_metric in AVAILABLE_RESOURCES:
            available_resource_num = AVAILABLE_RESOURCES[target_metric]
            if not isinstance(available_resource_num, (int, float)) or available_resource_num <= 0:
                raise ValueError(f"AVAILABLE_RESOURCES['{target_metric}'] 必须为正数，当前为: {available_resource_num}")
            metrics_dict['design_ulti_rmse'] = design_rmse / available_resource_num

        # 回传预测序列用于后续可视化
        metrics_dict.update({
            'design_preds': design_preds.squeeze(),
            'design_true': design_true.squeeze()
        })

        if all_delta_preds and all_delta_true:
            delta_preds = torch.cat(all_delta_preds, dim=0)
            delta_true = torch.cat(all_delta_true, dim=0)
            metrics_dict.update({
                'delta_mae': F.l1_loss(delta_preds.squeeze(), delta_true.squeeze()).item(),
                'delta_rmse': torch.sqrt(F.mse_loss(delta_preds.squeeze(), delta_true.squeeze())).item(),
                'delta_r2': calculate_r2(delta_true.squeeze(), delta_preds.squeeze()),
                'delta_mape': calculate_mape(delta_true.squeeze(), delta_preds.squeeze())
            })
            if target_metric in AVAILABLE_RESOURCES:
                metrics_dict['delta_ulti_rmse'] = metrics_dict['delta_rmse'] / AVAILABLE_RESOURCES[target_metric]
            metrics_dict.update({
                'delta_preds': delta_preds.squeeze(),
                'delta_true': delta_true.squeeze()
            })

        if all_kernel_preds and all_kernel_true:
            kernel_preds = torch.cat(all_kernel_preds, dim=0)
            kernel_true = torch.cat(all_kernel_true, dim=0)
            metrics_dict.update({
                'kernel_mae': F.l1_loss(kernel_preds.squeeze(), kernel_true.squeeze()).item(),
                'kernel_rmse': torch.sqrt(F.mse_loss(kernel_preds.squeeze(), kernel_true.squeeze())).item(),
                'kernel_r2': calculate_r2(kernel_true.squeeze(), kernel_preds.squeeze()),
                'kernel_mape': calculate_mape(kernel_true.squeeze(), kernel_preds.squeeze())
            })
            if target_metric in AVAILABLE_RESOURCES:
                metrics_dict['kernel_ulti_rmse'] = metrics_dict['kernel_rmse'] / AVAILABLE_RESOURCES[target_metric]
            metrics_dict.update({
                'kernel_preds': kernel_preds.squeeze(),
                'kernel_true': kernel_true.squeeze()
            })

        return metrics_dict

    return {
        'avg_loss': avg_loss,
        'design_mae': 0,
        'design_rmse': 0,
        'design_r2': 0,
        'design_mape': 0,
        'design_ulti_rmse': 0,
        'kernel_mae': 0,
        'kernel_rmse': 0,
        'kernel_r2': 0,
        'kernel_mape': 0,
        'kernel_ulti_rmse': 0
    }


def _create_prediction_plots(test_metrics: Dict, target_metric: str, output_dir: str):
    """将设计指标预测散点直接作为 ECharts 数据上传到 SwanLab。"""
    design_true = test_metrics['design_true'].cpu().numpy().tolist()
    design_preds = test_metrics['design_preds'].cpu().numpy().tolist()

    min_val = min(min(design_true), min(design_preds))
    max_val = max(max(design_true), max(design_preds))

    chart = (
        Scatter()
        .add_xaxis(design_true)
        .add_yaxis(
            series_name="Pred vs True",
            y_axis=design_preds,
            symbol_size=6,
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title=f"Prediction Scatter ({target_metric.upper()})"),
            xaxis_opts=opts.AxisOpts(
                name=f"True {target_metric.upper()}",
                min_=min_val,
                max_=max_val,
                type_="value",
            ),
            yaxis_opts=opts.AxisOpts(
                name=f"Pred {target_metric.upper()}",
                min_=min_val,
                max_=max_val,
                type_="value",
            ),
            tooltip_opts=opts.TooltipOpts(trigger="item"),
        )
    )

    # 直接上传散点数据（前端可交互）
    swanlab.log({"prediction_scatter": Echarts(chart)})


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
                        default=None,
                        help='Design数据根目录')
    parser.add_argument('--ood_design_base_dir', type=str,
                        default=None,
                        help='OOD Design数据根目录（可选，不存在则跳过 OOD 评估）')
    
    # 模型参数
    parser.add_argument('--target_metric', type=str, default=None,
                        choices=['dsp', 'lut', 'ff', 'latency'],
                        help='目标预测指标')
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=3, help='GNN层数')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    parser.add_argument('--gnn_type', type=str, default='gcn',
                        choices=['gcn', 'gin', 'rgcn', 'fast_rgcn', 'pna'],
                        help='GNN架构类型')
    parser.add_argument('--kernel_baseline', type=str, default='learned',
                        choices=['learned', 'oracle'],
                        help='kernel基线生成方式: learned=通过kernel头预测, oracle=直接使用真实kernel指标')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--device', type=int, default=0, help='GPU设备ID')
    parser.add_argument('--loss_type', type=str, default='mae',
                        choices=['mae', 'mse', 'smoothl1'],
                        help='训练与评估时使用的损失函数')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='梯度累积步数（>1 可提高有效批大小）')
    parser.add_argument('--loader_workers', type=int, default=8, help='DataLoader worker 数量（>0 启用多进程加载）')
    parser.add_argument('--prefetch_factor', type=int, default=4, help='DataLoader 预取批数（num_workers>0 时有效）')
    parser.add_argument('--persistent_workers', type=str, default='true', choices=['true', 'false'], help='是否持久化 DataLoader workers')
    parser.add_argument('--loader_start_method', type=str, default='spawn',
                        choices=['spawn', 'fork'],
                        help='DataLoader worker 启动方式（spawn 更安全，fork 更快但可能与 CUDA 线程冲突）')
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
    parser.add_argument('--use_code_feature', type=str, default='false', choices=['true', 'false'], help='是否启用设计源码的 LLM 嵌入特征')
    parser.add_argument('--code_model_path', type=str, default=None, help='LLM 模型路径（启用代码特征时必填）')
    parser.add_argument('--code_cache_root', type=str, default=default_cache_root, help='代码嵌入缓存根目录')
    parser.add_argument('--code_pooling', type=str, default='last_token', choices=['last_token', 'mean'], help='代码嵌入的 pooling 策略')
    parser.add_argument('--code_max_length', type=int, default=1024, help='代码嵌入的最大 token 长度')
    parser.add_argument('--code_normalize', type=str, default='true', choices=['true', 'false'], help='是否对代码嵌入做 L2 归一化')
    parser.add_argument('--code_batch_size', type=int, default=8, help='预编码代码嵌入的批大小')
    
    # 调试参数
    parser.add_argument('--max_pairs', type=int, default=None, help='最大配对数量（调试用）')
    
    # 模式参数
    parser.add_argument('--differential', type=str, default='true', 
                        choices=['true', 'false'],
                        help='是否使用差分学习模式：true=差分学习(kernel+design)，false=直接预测(仅design)')
    
    # 训练策略参数
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Linear warmup 的 epoch 数（0 表示关闭）')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='学习率下限（避免过小导致训练停滞）')
    parser.add_argument('--apply_hard_filter', type=str, default='true', choices=['true', 'false'],
                        help='是否过滤无效设计（基于稳健分位 p05-p95）')
    parser.add_argument('--normalize_targets', type=str, default='true', choices=['true', 'false'],
                        help='是否对目标值/差值使用稳健归一化（median/IQR）用于计算loss')
    
    # 层次化图开关
    parser.add_argument('--hierarchical', type=str, default='off', choices=['on', 'off'],
                        help='是否启用多层次结构构图（hierarchical graph），on/off')
    # Region 信息开关
    parser.add_argument('--region', type=str, default='off', choices=['on', 'off'],
                        help='是否启用 region 信息（构图元数据中记录），on/off')

    # 并行构建图cache参数
    parser.add_argument('--max_workers', type=int, default=32,
                        help='数据处理并行线程/进程数（<=0 表示自动选择）')
    
    # 模型保存选项
    parser.add_argument('--save_final_model', action='store_true',
                        help='是否保存最终epoch的模型（默认只保存best model，学术论文通常使用best model）')
    
    args = parser.parse_args()
    args.loss_type = args.loss_type.lower()
    loss_fn = resolve_loss_function(args.loss_type)
    use_code_feature = (args.use_code_feature.lower() == 'true')
    code_normalize = (args.code_normalize.lower() == 'true')
    if use_code_feature and not args.code_model_path:
        raise ValueError("use_code_feature=true 时必须提供 --code_model_path")

    _configure_mp(args.loader_start_method)
    
    # 设置设备
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"e2e_delta_{args.target_metric}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    log_status(f"[Step1] 输出目录已创建: {output_dir}")
    
    # 初始化SwanLab
    swanlab.init(
        project="HLS-Differential-Learning",
        experiment_name=f"E2E_Delta_{args.target_metric}_{timestamp}",
        config=vars(args),
        logdir=output_dir
    )
    log_status("[Step2] SwanLab 初始化完成，开始数据处理流水线")
    
    # ==================== 数据处理 ====================
    
    processor = E2EDifferentialProcessor(
        kernel_base_dir=args.kernel_base_dir,
        design_base_dir=args.design_base_dir,
        output_dir=output_dir,
        cache_root=args.cache_root,
        rebuild_cache=args.rebuild_cache,
        hierarchical=(args.hierarchical.lower() == 'on'),
        region=(args.region.lower() == 'on'),
        max_workers=(None if args.max_workers is None or args.max_workers <= 0 else args.max_workers),
        use_code_feature=use_code_feature,
        code_model_path=args.code_model_path,
        code_cache_root=args.code_cache_root,
        code_pooling=args.code_pooling,
        code_max_length=args.code_max_length,
        code_normalize=code_normalize,
        code_batch_size=args.code_batch_size
    )
    
    # 收集配对数据
    log_status("[Step3] 开始构建/加载ID图缓存并生成kernel-design配对")
    pairs = processor.collect_all_data()
    log_status(f"[Step3.5] ID 图缓存与配对完成，获得 {len(pairs)} 条记录")
    
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
            max_workers=(None if args.max_workers is None or args.max_workers <= 0 else args.max_workers),
            use_code_feature=use_code_feature,
            code_model_path=args.code_model_path,
            code_cache_root=args.code_cache_root,
            code_pooling=args.code_pooling,
            code_max_length=args.code_max_length,
            code_normalize=code_normalize,
            code_batch_size=args.code_batch_size
        )
        log_status("[Step5] 开始构建/加载OOD图缓存并生成kernel-design配对")
        ood_pairs = ood_processor.collect_all_data()
        log_status(f"[Step5] OOD 图缓存与配对完成，获得 {len(ood_pairs)} 条记录")
    else:
        print(f"提示: OOD 路径不可用或未提供，将跳过 OOD 评估: {args.ood_design_base_dir}")
        log_status("[Step5] 未提供 OOD 路径，跳过 OOD 评估阶段")
    
    # 记录数据集信息
    swanlab.log({"dataset/total_pairs": len(pairs)})
    if ood_pairs:
        swanlab.log({"dataset/ood_pairs": len(ood_pairs)})
    # 记录是否使用层次化图
    swanlab.log({"config/hierarchical": (args.hierarchical.lower() == 'on')})
    # 记录是否使用 region 信息
    swanlab.log({"config/region": (args.region.lower() == 'on')})
    swanlab.log({
        "config/kernel_baseline_mode": 0 if args.kernel_baseline.lower() == 'learned' else 1
    })
    swanlab.log({"config/loss_type": swanlab.Text(args.loss_type)})
    # 记录流水线旁证的总体占比（设计图）
    # 限制配对数量（调试用）
    if args.max_pairs and len(pairs) > args.max_pairs:
        pairs = pairs[:args.max_pairs]
        swanlab.log({"dataset/limited_pairs": args.max_pairs})
        log_status(f"[Step3.6] 调试模式：截断配对数至 {args.max_pairs}")

    # 统计原始分布并保存
    metric_idx = {'dsp': 0, 'lut': 1, 'ff': 2, 'latency': 3}[args.target_metric]
    delta_key = f'{args.target_metric}_delta' if args.target_metric != 'latency' else 'latency_delta'
    raw_stats = {k: _compute_basic_stats(v) for k, v in _extract_metric_values(pairs, metric_idx, delta_key).items()}
    stats_path = os.path.join(output_dir, 'target_metric_stats.json')
    with open(stats_path, 'w') as sf:
        json.dump({"raw": raw_stats}, sf, indent=2)
    log_status(f"[Step4] 已统计原始分布并写入 {stats_path}")
    swanlab.log({f"stats/raw/{k}/p05": v.get("p05", float('nan')) for k, v in raw_stats.items() if v})
    swanlab.log({f"stats/raw/{k}/p95": v.get("p95", float('nan')) for k, v in raw_stats.items() if v})

    # p05-p95 硬过滤（保持 10% avail 过滤基础上）
    if args.apply_hard_filter.lower() == 'true':
        thresholds = {k: (v["p05"], v["p95"]) for k, v in raw_stats.items() if v}
        before = len(pairs)
        filtered_pairs: List[Dict] = []
        for pair in pairs:
            vals = _extract_metric_values([pair], metric_idx, delta_key)
            keep = True
            for key, (low, high) in thresholds.items():
                value = vals[key][0]
                if value < low or value > high:
                    keep = False
                    break
            if keep:
                filtered_pairs.append(pair)
        pairs = filtered_pairs
        dropped = before - len(pairs)
        log_status(f"[Step4.1] 过滤无效设计（p05-p95），移除 {dropped} 条，剩余 {len(pairs)} 条")
        swanlab.log({
            "dataset/hard_filter_dropped": dropped,
            "dataset/after_hard_filter": len(pairs)
        })
    if not pairs:
        raise ValueError("硬过滤后无有效样本，请调整过滤阈值或关闭 --apply_hard_filter")
    
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
        }
        pairs_info.append(info)
    
    pairs_path = os.path.join(output_dir, 'kernel_design_pairs.json')
    with open(pairs_path, 'w') as f:
        json.dump(pairs_info, f, indent=2)
    
    # 不再上传表格形式的配对信息，保留 JSON 文件以便线下查看
    
    # 更新过滤后分布
    filtered_stats = {k: _compute_basic_stats(v) for k, v in _extract_metric_values(pairs, metric_idx, delta_key).items()}
    try:
        with open(stats_path, 'r') as sf:
            prev_stats = json.load(sf)
    except Exception:
        prev_stats = {}
    prev_stats["filtered"] = filtered_stats
    with open(stats_path, 'w') as sf:
        json.dump(prev_stats, sf, indent=2)
    log_status(f"[Step4.2] 过滤后分布已追加至 {stats_path}")

    # 创建数据集
    dataset = E2EDifferentialDataset(pairs, args.target_metric)
    ood_dataset = E2EDifferentialDataset(ood_pairs, args.target_metric) if ood_pairs else None
    
    # 数据集划分 - 8:1:1随机划分
    log_status("[Step6] 开始构建数据集并执行 8:1:1 划分")
    total_size = len(dataset)
    if total_size == 0:
        raise ValueError("预处理后数据集为空，无法继续训练。请放宽过滤阈值。")
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
    log_status(f"[Step6] 数据集划分完成: train={len(train_dataset)}, valid={len(valid_dataset)}, test={len(test_dataset)}")
    
    # 记录数据划分信息
    swanlab.log({
        "dataset/train_size": len(train_dataset),
        "dataset/valid_size": len(valid_dataset), 
        "dataset/test_size": len(test_dataset)
    })
    
    # 不再以表格形式记录数据集划分；仅保留标量日志

    # 基于训练集拟合稳健归一化器
    normalizers: Optional[Dict[str, RobustScaler]] = None
    if args.normalize_targets.lower() == 'true':
        train_pairs = [pairs[int(i)] for i in train_indices.tolist()]
        train_values = _extract_metric_values(train_pairs, metric_idx, delta_key)
        if any(len(v) == 0 for v in train_values.values()):
            raise ValueError("训练集目标值为空，无法拟合归一化器。请检查过滤配置。")
        normalizers = {k: _fit_robust_scaler(v) for k, v in train_values.items()}
        norm_stats = {k: v.to_dict() for k, v in normalizers.items()}
        try:
            with open(stats_path, 'r') as sf:
                prev_stats = json.load(sf)
        except Exception:
            prev_stats = {}
        prev_stats["normalizer"] = norm_stats
        with open(stats_path, 'w') as sf:
            json.dump(prev_stats, sf, indent=2)
        log_status(f"[Step6.1] 已基于训练集拟合稳健归一化器并写入 {stats_path}")
        swanlab.log({f"normalizer/{k}/scale": v.scale for k, v in normalizers.items()})
    
    # 创建数据加载器
    loader_workers = max(0, int(args.loader_workers))
    persistent = (args.persistent_workers.lower() == 'true')
    loader_common_kwargs = {
        'collate_fn': differential_collate_fn,
        'num_workers': loader_workers,
        'pin_memory': (args.pin_memory.lower() == 'true'),
        'persistent_workers': (persistent if loader_workers > 0 else False)
    }
    # 为 DataLoader 指定 worker 上下文，确保 Ctrl+C 能可靠终止子进程
    if loader_workers > 0:
        try:
            loader_common_kwargs['multiprocessing_context'] = args.loader_start_method
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
    log_status("[Step7] DataLoader 构建完成，准备创建模型")
    
    # ==================== 创建模型 ====================
    
    # 获取特征维度
    sample_pair = dataset[0]
    node_dim = sample_pair['kernel_graph'].x.size(1)
    code_dim = None
    if args.use_code_feature.lower() == 'true':
        sample_code = sample_pair.get('design_code')
        if sample_code is not None:
            code_dim = sample_code.shape[-1]
        else:
            raise ValueError("use_code_feature=True 但样本缺少 design_code，请检查源码或缓存构建。")
    edge_dim = 0
    if sample_pair['kernel_graph'].edge_attr is not None:
        edge_dim = sample_pair['kernel_graph'].edge_attr.size(1)
    elif sample_pair['design_graph'].edge_attr is not None:
        edge_dim = sample_pair['design_graph'].edge_attr.size(1)
    
    pna_deg = None
    if args.gnn_type.lower() == 'pna':
        def _compute_pna_degree_histogram(subset: torch.utils.data.Subset) -> torch.Tensor:
            hist = torch.zeros(1, dtype=torch.long)
            if isinstance(subset, torch.utils.data.Subset):
                base_dataset = subset.dataset
                if isinstance(subset.indices, torch.Tensor):
                    index_iterable = subset.indices.tolist()
                else:
                    index_iterable = list(subset.indices)
            else:
                base_dataset = subset
                index_iterable = list(range(len(subset)))
            for raw_idx in index_iterable:
                idx = int(raw_idx)
                sample = base_dataset[idx]
                for graph in (sample['kernel_graph'], sample['design_graph']):
                    if not hasattr(graph, 'edge_index') or graph.edge_index is None:
                        continue
                    deg = degree(graph.edge_index[1], num_nodes=graph.num_nodes, dtype=torch.long)
                    if deg.numel() == 0:
                        continue
                    max_deg = int(deg.max().item())
                    if hist.numel() <= max_deg:
                        hist = torch.cat([hist, torch.zeros(max_deg - hist.numel() + 1, dtype=torch.long)])
                    hist[:max_deg + 1] += torch.bincount(deg, minlength=max_deg + 1)
            if hist.sum() == 0:
                hist[0] = 1
            return hist

        pna_deg = _compute_pna_degree_histogram(train_dataset)
    
    # 创建模型
    differential_mode = args.differential.lower() == 'true'
    model = SimpleDifferentialGNN(
        node_dim=node_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        differential=differential_mode,
        dropout=args.dropout,
        target_metric=args.target_metric,
        gnn_type=args.gnn_type,
        kernel_baseline=args.kernel_baseline if differential_mode else 'learned',
        pna_deg=pna_deg,
        edge_dim=edge_dim,
        use_code_feature=(args.use_code_feature.lower() == 'true'),
        code_dim=code_dim
    ).to(device)
    log_status("[Step8] 模型初始化完成，开始训练循环准备")
    
    # 优化器和调度器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=30)
    
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
    
    log_status(f"[Step9] 进入训练阶段，共 {args.epochs} 个 epoch")
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
        print(f"[Epoch {epoch}/{args.epochs}] 开始训练...", flush=True)
        train_loss = train_epoch(
            model, device, train_loader, optimizer,
            loss_fn=loss_fn,
            grad_accum_steps=args.grad_accum_steps,
            normalizers=normalizers,
            use_tqdm=True,
            progress_desc=f"Train {epoch}/{args.epochs}"
        )
        train_losses.append(train_loss)
        
        # 验证
        valid_metrics = evaluate_model(
            model, valid_loader, device, args.target_metric,
            loss_fn=loss_fn,
            normalizers=normalizers
        )
        valid_loss = valid_metrics['avg_loss']
        valid_losses.append(valid_loss)
        valid_metrics_history.append(valid_metrics)
        
        # ID 测试
        id_test_metrics = evaluate_model(
            model, test_loader, device, args.target_metric,
            loss_fn=loss_fn,
            normalizers=normalizers
        )
        id_test_loss = id_test_metrics['avg_loss']
        test_losses.append(id_test_loss)
        test_metrics_history.append(id_test_metrics)
        
        # OOD 测试（可选）
        ood_test_metrics = None
        if ood_test_loader is not None:
            ood_test_metrics = evaluate_model(
                model, ood_test_loader, device, args.target_metric,
                loss_fn=loss_fn,
                normalizers=normalizers
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
        log_payload = {
            "epoch": epoch,
            "train/loss": train_loss,
            "valid/loss": valid_loss,
            "id_test/loss": id_test_loss,
            "optimizer/lr": current_lr
        }

        for split_name, split_metrics in [("valid", valid_metrics), ("id_test", id_test_metrics)]:
            log_payload[f"{split_name}/design_mae"] = split_metrics.get('design_mae', 0)
            log_payload[f"{split_name}/design_rmse"] = split_metrics.get('design_rmse', 0)
            log_payload[f"{split_name}/design_mape"] = split_metrics.get('design_mape', 0)
            log_payload[f"{split_name}/design_r2"] = split_metrics.get('design_r2', 0)
            log_payload[f"{split_name}/design_ulti_rmse"] = split_metrics.get('design_ulti_rmse', 0)

            log_payload[f"{split_name}/kernel_mae"] = split_metrics.get('kernel_mae', 0)
            log_payload[f"{split_name}/kernel_rmse"] = split_metrics.get('kernel_rmse', 0)
            log_payload[f"{split_name}/kernel_mape"] = split_metrics.get('kernel_mape', 0)
            log_payload[f"{split_name}/kernel_r2"] = split_metrics.get('kernel_r2', 0)
            log_payload[f"{split_name}/kernel_ulti_rmse"] = split_metrics.get('kernel_ulti_rmse', 0)

            if 'delta_mae' in split_metrics:
                log_payload[f"{split_name}/delta_mae"] = split_metrics.get('delta_mae', 0)
                log_payload[f"{split_name}/delta_rmse"] = split_metrics.get('delta_rmse', 0)
                log_payload[f"{split_name}/delta_mape"] = split_metrics.get('delta_mape', 0)
                log_payload[f"{split_name}/delta_r2"] = split_metrics.get('delta_r2', 0)
                log_payload[f"{split_name}/delta_ulti_rmse"] = split_metrics.get('delta_ulti_rmse', 0)

        if ood_test_metrics is not None:
            log_payload.update({
                "ood_test/loss": ood_test_metrics.get('avg_loss', 0),
                "ood_test/design_mae": ood_test_metrics.get('design_mae', 0),
                "ood_test/design_rmse": ood_test_metrics.get('design_rmse', 0),
                "ood_test/design_mape": ood_test_metrics.get('design_mape', 0),
                "ood_test/design_r2": ood_test_metrics.get('design_r2', 0),
                "ood_test/design_ulti_rmse": ood_test_metrics.get('design_ulti_rmse', 0),
                "ood_test/kernel_mae": ood_test_metrics.get('kernel_mae', 0),
                "ood_test/kernel_rmse": ood_test_metrics.get('kernel_rmse', 0),
                "ood_test/kernel_mape": ood_test_metrics.get('kernel_mape', 0),
                "ood_test/kernel_r2": ood_test_metrics.get('kernel_r2', 0),
                "ood_test/kernel_ulti_rmse": ood_test_metrics.get('kernel_ulti_rmse', 0)
            })

            if 'delta_mae' in ood_test_metrics:
                log_payload.update({
                    "ood_test/delta_mae": ood_test_metrics.get('delta_mae', 0),
                    "ood_test/delta_rmse": ood_test_metrics.get('delta_rmse', 0),
                    "ood_test/delta_mape": ood_test_metrics.get('delta_mape', 0),
                    "ood_test/delta_r2": ood_test_metrics.get('delta_r2', 0),
                    "ood_test/delta_ulti_rmse": ood_test_metrics.get('delta_ulti_rmse', 0)
                })

        swanlab.log(log_payload)
        summary_msg = (
            f"[Epoch {epoch}/{args.epochs}] 训练完成 | "
            f"train_loss={train_loss:.6f}, valid_loss={valid_loss:.6f}, "
            f"id_test_loss={id_test_loss:.6f}, lr={current_lr:.6f}"
        )
        if ood_test_metrics is not None:
            summary_msg += f", ood_loss={ood_test_metrics.get('avg_loss', 0):.6f}"
        log_status(f"[Step9] {summary_msg}")
        
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
            "final/id_design_mae": best_test.get('design_mae', 0),
            "final/id_design_rmse": best_test.get('design_rmse', 0),
            "final/id_design_mape": best_test.get('design_mape', 0),
            "final/id_design_ulti_rmse": best_test.get('design_ulti_rmse', 0),
            "final/id_design_r2": best_test.get('design_r2', 0),
            "final/id_kernel_mae": best_test.get('kernel_mae', 0),
            "final/id_kernel_rmse": best_test.get('kernel_rmse', 0),
            "final/id_kernel_mape": best_test.get('kernel_mape', 0),
            "final/id_kernel_ulti_rmse": best_test.get('kernel_ulti_rmse', 0),
            "final/id_kernel_r2": best_test.get('kernel_r2', 0),
            **({
                "final/id_delta_mae": best_test.get('delta_mae', 0),
                "final/id_delta_rmse": best_test.get('delta_rmse', 0),
                "final/id_delta_mape": best_test.get('delta_mape', 0),
                "final/id_delta_ulti_rmse": best_test.get('delta_ulti_rmse', 0),
                "final/id_delta_r2": best_test.get('delta_r2', 0)
            } if 'delta_mae' in best_test else {})
        })
        
        # 如果 OOD 可用，也记录最佳 epoch 对应的 OOD 指标（同一 epoch 下）
        if 'ood_test_loader' in locals() and ood_test_loader is not None:
            best_ood_metrics = evaluate_model(
                model, ood_test_loader, device, args.target_metric,
                loss_fn=loss_fn
            )
            swanlab.log({
                "final/ood_design_mae": best_ood_metrics.get('design_mae', 0),
                "final/ood_design_rmse": best_ood_metrics.get('design_rmse', 0),
                "final/ood_design_mape": best_ood_metrics.get('design_mape', 0),
                "final/ood_design_ulti_rmse": best_ood_metrics.get('design_ulti_rmse', 0),
                "final/ood_design_r2": best_ood_metrics.get('design_r2', 0),
                "final/ood_kernel_mae": best_ood_metrics.get('kernel_mae', 0),
                "final/ood_kernel_rmse": best_ood_metrics.get('kernel_rmse', 0),
                "final/ood_kernel_mape": best_ood_metrics.get('kernel_mape', 0),
                "final/ood_kernel_ulti_rmse": best_ood_metrics.get('kernel_ulti_rmse', 0),
                "final/ood_kernel_r2": best_ood_metrics.get('kernel_r2', 0),
                **({
                    "final/ood_delta_mae": best_ood_metrics.get('delta_mae', 0),
                    "final/ood_delta_rmse": best_ood_metrics.get('delta_rmse', 0),
                    "final/ood_delta_mape": best_ood_metrics.get('delta_mape', 0),
                    "final/ood_delta_ulti_rmse": best_ood_metrics.get('delta_ulti_rmse', 0),
                    "final/ood_delta_r2": best_ood_metrics.get('delta_r2', 0)
                } if 'delta_mae' in best_ood_metrics else {})
            })
        
        # 创建最佳epoch的预测vs真实散点图（仅 ID test ）
        if 'design_preds' in best_test and 'design_true' in best_test:
            _create_prediction_plots(best_test, args.target_metric, output_dir)
        
        print(f"训练完成! 最佳模型 ({args.target_metric.upper()}):")
        print(f"  Epoch: {best_epoch}")
        print(f"  [ID] design MAE: {best_test.get('design_mae', 0):.6f}")
        print(f"  [ID] design RMSE: {best_test.get('design_rmse', 0):.6f}")
        print(f"  [ID] design MAPE: {best_test.get('design_mape', 0):.2f}%")
        print(f"  [ID] design ulti-RMSE: {best_test.get('design_ulti_rmse', 0):.8f}")
        print(f"  [ID] design R²: {best_test.get('design_r2', 0):.4f}")
        print(f"  [ID] kernel MAE: {best_test.get('kernel_mae', 0):.6f}")
        print(f"  [ID] kernel RMSE: {best_test.get('kernel_rmse', 0):.6f}")
        print(f"  [ID] kernel MAPE: {best_test.get('kernel_mape', 0):.2f}%")
        print(f"  [ID] kernel ulti-RMSE: {best_test.get('kernel_ulti_rmse', 0):.8f}")
        print(f"  [ID] kernel R²: {best_test.get('kernel_r2', 0):.4f}")
        if 'delta_mae' in best_test:
            print(f"  [ID] delta MAE: {best_test.get('delta_mae', 0):.6f}")
            print(f"  [ID] delta RMSE: {best_test.get('delta_rmse', 0):.6f}")
            print(f"  [ID] delta MAPE: {best_test.get('delta_mape', 0):.2f}%")
            print(f"  [ID] delta ulti-RMSE: {best_test.get('delta_ulti_rmse', 0):.8f}")
            print(f"  [ID] delta R²: {best_test.get('delta_r2', 0):.4f}")
        if 'ood_test_loader' in locals() and ood_test_loader is not None:
            if best_ood_metrics is not None:
                print(f"  [OOD] design MAE: {best_ood_metrics.get('design_mae', 0):.6f}")
                print(f"  [OOD] design RMSE: {best_ood_metrics.get('design_rmse', 0):.6f}")
                print(f"  [OOD] design MAPE: {best_ood_metrics.get('design_mape', 0):.2f}%")
                print(f"  [OOD] design ulti-RMSE: {best_ood_metrics.get('design_ulti_rmse', 0):.8f}")
                print(f"  [OOD] design R²: {best_ood_metrics.get('design_r2', 0):.4f}")
                print(f"  [OOD] kernel MAE: {best_ood_metrics.get('kernel_mae', 0):.6f}")
                print(f"  [OOD] kernel RMSE: {best_ood_metrics.get('kernel_rmse', 0):.6f}")
                print(f"  [OOD] kernel MAPE: {best_ood_metrics.get('kernel_mape', 0):.2f}%")
                print(f"  [OOD] kernel ulti-RMSE: {best_ood_metrics.get('kernel_ulti_rmse', 0):.8f}")
                print(f"  [OOD] kernel R²: {best_ood_metrics.get('kernel_r2', 0):.4f}")
                if 'delta_mae' in best_ood_metrics:
                    print(f"  [OOD] delta MAE: {best_ood_metrics.get('delta_mae', 0):.6f}")
                    print(f"  [OOD] delta RMSE: {best_ood_metrics.get('delta_rmse', 0):.6f}")
                    print(f"  [OOD] delta MAPE: {best_ood_metrics.get('delta_mape', 0):.2f}%")
                    print(f"  [OOD] delta ulti-RMSE: {best_ood_metrics.get('delta_ulti_rmse', 0):.8f}")
                    print(f"  [OOD] delta R²: {best_ood_metrics.get('delta_r2', 0):.4f}")
    
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
    
    # 合并 Loss 与 MAE（因为 loss == MAE）: 展示 train(valid loss=mae), valid/test design_mae 三条曲线
    if valid_metrics_history:
        design_mae_valid = [m.get('design_mae', 0) for m in valid_metrics_history]
        design_mae_id_test = [m.get('design_mae', 0) for m in test_metrics_history]
        design_mae_ood_test = [m.get('design_mae', 0) for m in ood_metrics_history] if ood_metrics_history else []
        if differential_mode:
            kernel_mae_valid = [m.get('kernel_mae', 0) for m in valid_metrics_history]
            kernel_mae_id_test = [m.get('kernel_mae', 0) for m in test_metrics_history]
            kernel_mae_ood_test = [m.get('kernel_mae', 0) for m in ood_metrics_history] if ood_metrics_history else []
    else:
        design_mae_valid, design_mae_id_test, design_mae_ood_test = [], [], []
        kernel_mae_valid, kernel_mae_id_test, kernel_mae_ood_test = [], [], []

    axes[0, 0].plot(epochs, train_losses, color='blue', label='Train (Loss=MAE)', linewidth=2)
    if design_mae_valid:
        axes[0, 0].plot(epochs, design_mae_valid, color='orange', label='Valid MAE', linewidth=2)
    if test_metrics_history:
        axes[0, 0].plot(epochs, design_mae_id_test, color='purple', label='ID Test MAE', linewidth=2)
    if design_mae_ood_test:
        axes[0, 0].plot(epochs, design_mae_ood_test, color='red', label='OOD Test MAE', linewidth=2)

    if differential_mode and kernel_mae_valid:
        axes[0, 0].plot(epochs, kernel_mae_valid, linestyle='--', color='orange', alpha=0.6, label='Valid Kernel MAE')
        axes[0, 0].plot(epochs, kernel_mae_id_test, linestyle='--', color='purple', alpha=0.6, label='ID Kernel MAE')
        if kernel_mae_ood_test:
            axes[0, 0].plot(epochs, kernel_mae_ood_test, linestyle='--', color='red', alpha=0.6, label='OOD Kernel MAE')
    
    axes[0, 0].axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label='Best Epoch')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MAE (Loss)')
    axes[0, 0].set_title('MAE (Loss) - Train / Valid / ID Test / OOD Test')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 设计RMSE (放在 (0,1))，如存在差值信息则使用虚线辅助对比
    if valid_metrics_history:
        design_rmse_valid = [m.get('design_rmse', 0) for m in valid_metrics_history]
        design_rmse_id_test = [m.get('design_rmse', 0) for m in test_metrics_history]
        design_rmse_ood_test = [m.get('design_rmse', 0) for m in ood_metrics_history] if ood_metrics_history else []
        axes[0, 1].plot(epochs, design_rmse_valid, 'cyan', label='Valid Design RMSE', linewidth=2)
        axes[0, 1].plot(epochs, design_rmse_id_test, 'magenta', label='ID Test Design RMSE', linewidth=2)
        if design_rmse_ood_test:
            axes[0, 1].plot(epochs, design_rmse_ood_test, color='darkred', label='OOD Test Design RMSE', linewidth=2)

        if differential_mode:
            # 差值RMSE采用虚线展示（若存在）
            delta_rmse_valid = [m.get('delta_rmse', None) for m in valid_metrics_history]
            if any(v is not None for v in delta_rmse_valid):
                delta_rmse_valid = [m.get('delta_rmse', float('nan')) for m in valid_metrics_history]
                delta_rmse_id_test = [m.get('delta_rmse', float('nan')) for m in test_metrics_history]
                delta_rmse_ood_test = [m.get('delta_rmse', float('nan')) for m in ood_metrics_history] if ood_metrics_history else []
                axes[0, 1].plot(epochs, delta_rmse_valid, linestyle='--', color='cyan', alpha=0.6, label='Valid Delta RMSE')
                axes[0, 1].plot(epochs, delta_rmse_id_test, linestyle='--', color='magenta', alpha=0.6, label='ID Test Delta RMSE')
                if delta_rmse_ood_test:
                    axes[0, 1].plot(epochs, delta_rmse_ood_test, linestyle='--', color='darkred', alpha=0.6, label='OOD Test Delta RMSE')

            kernel_rmse_valid = [m.get('kernel_rmse', float('nan')) for m in valid_metrics_history]
            kernel_rmse_id_test = [m.get('kernel_rmse', float('nan')) for m in test_metrics_history]
            kernel_rmse_ood_test = [m.get('kernel_rmse', float('nan')) for m in ood_metrics_history] if ood_metrics_history else []
            axes[0, 1].plot(epochs, kernel_rmse_valid, linestyle='-.', color='cyan', alpha=0.6, label='Valid Kernel RMSE')
            axes[0, 1].plot(epochs, kernel_rmse_id_test, linestyle='-.', color='magenta', alpha=0.6, label='ID Kernel RMSE')
            if kernel_rmse_ood_test:
                axes[0, 1].plot(epochs, kernel_rmse_ood_test, linestyle='-.', color='darkred', alpha=0.6, label='OOD Kernel RMSE')

        axes[0, 1].axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_title('Design RMSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # 设计R2曲线 (放在 (1,0))，如存在差值信息则用虚线叠加
    if valid_metrics_history:
        design_r2_valid = [m.get('design_r2', 0) for m in valid_metrics_history]
        design_r2_id_test = [m.get('design_r2', 0) for m in test_metrics_history]
        design_r2_ood_test = [m.get('design_r2', 0) for m in ood_metrics_history] if ood_metrics_history else []
        axes[1, 0].plot(epochs, design_r2_valid, 'brown', label='Valid Design R²', linewidth=2)
        axes[1, 0].plot(epochs, design_r2_id_test, 'pink', label='ID Test Design R²', linewidth=2)
        if design_r2_ood_test:
            axes[1, 0].plot(epochs, design_r2_ood_test, color='firebrick', label='OOD Test Design R²', linewidth=2)

        if differential_mode:
            delta_r2_valid = [m.get('delta_r2', None) for m in valid_metrics_history]
            if any(v is not None for v in delta_r2_valid):
                delta_r2_valid = [m.get('delta_r2', float('nan')) for m in valid_metrics_history]
                delta_r2_id_test = [m.get('delta_r2', float('nan')) for m in test_metrics_history]
                delta_r2_ood_test = [m.get('delta_r2', float('nan')) for m in ood_metrics_history] if ood_metrics_history else []
                axes[1, 0].plot(epochs, delta_r2_valid, linestyle='--', color='brown', alpha=0.6, label='Valid Delta R²')
                axes[1, 0].plot(epochs, delta_r2_id_test, linestyle='--', color='pink', alpha=0.6, label='ID Test Delta R²')
                if delta_r2_ood_test:
                    axes[1, 0].plot(epochs, delta_r2_ood_test, linestyle='--', color='firebrick', alpha=0.6, label='OOD Test Delta R²')

            kernel_r2_valid = [m.get('kernel_r2', float('nan')) for m in valid_metrics_history]
            kernel_r2_id_test = [m.get('kernel_r2', float('nan')) for m in test_metrics_history]
            kernel_r2_ood_test = [m.get('kernel_r2', float('nan')) for m in ood_metrics_history] if ood_metrics_history else []
            axes[1, 0].plot(epochs, kernel_r2_valid, linestyle='-.', color='brown', alpha=0.6, label='Valid Kernel R²')
            axes[1, 0].plot(epochs, kernel_r2_id_test, linestyle='-.', color='pink', alpha=0.6, label='ID Kernel R²')
            if kernel_r2_ood_test:
                axes[1, 0].plot(epochs, kernel_r2_ood_test, linestyle='-.', color='firebrick', alpha=0.6, label='OOD Kernel R²')

        axes[1, 0].axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].set_title('Design R²')
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
    log_status("[Step10] 训练阶段完成，曲线图已保存并上传")

    # 可选：保存最终epoch的模型（用于对比分析，但论文通常使用best model）
    # 在学术论文中，通常使用验证集上的最佳模型更公平和标准
    if args.save_final_model:
        final_model_path = os.path.join(output_dir, f'final_e2e_delta_{args.target_metric}_model.pt')
        torch.save({
            'epoch': args.epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'valid_loss': valid_loss,
            'id_test_metrics': id_test_metrics,
            'ood_test_metrics': ood_test_metrics,
            'args': vars(args)
        }, final_model_path)
        print(f"最终模型已保存: {final_model_path}")
    
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
