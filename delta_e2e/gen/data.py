import os
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
import re

try:
    from delta_e2e.train_e2e import E2EDifferentialProcessor
except ModuleNotFoundError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from train_e2e import E2EDifferentialProcessor


def _safe_read(path: str) -> str:
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception:
        return ""


def extract_code_features(code_root: str) -> torch.Tensor:
    """Extract heuristic design-code statistics for conditional guidance."""
    exts = ('.c', '.cpp', '.cc', '.h', '.hpp')
    file_paths: List[str] = []
    try:
        for root, _, files in os.walk(code_root):
            for fname in files:
                if fname.lower().endswith(exts):
                    file_paths.append(os.path.join(root, fname))
    except Exception:
        pass

    num_files = len(file_paths)
    total_lines = 0
    pragma_total = 0
    pragma_counts = {k: 0 for k in ['PIPELINE', 'UNROLL', 'ARRAY_PARTITION', 'DATAFLOW', 'INLINE', 'INTERFACE']}
    func_count = 0
    loop_count = 0
    array_count = 0
    comment_lines = 0
    avg_len_acc = 0.0
    avg_len_cnt = 0
    num_includes = 0
    num_macros = 0

    pragma_patterns = {
        'PIPELINE': r'#pragma\s+HLS\s+PIPELINE',
        'UNROLL': r'#pragma\s+HLS\s+UNROLL',
        'ARRAY_PARTITION': r'#pragma\s+HLS\s+ARRAY_PARTITION',
        'DATAFLOW': r'#pragma\s+HLS\s+DATAFLOW',
        'INLINE': r'#pragma\s+HLS\s+INLINE',
        'INTERFACE': r'#pragma\s+HLS\s+INTERFACE'
    }

    func_regex = re.compile(r'\b[A-Za-z_][A-Za-z_0-9]*\s+\**\s*[A-Za-z_][A-Za-z_0-9]*\s*\(', re.MULTILINE)
    loop_regex = re.compile(r'\bfor\s*\(|\bwhile\s*\(')

    for fp in file_paths:
        content = _safe_read(fp)
        if not content:
            continue
        lines = content.splitlines()
        total_lines += len(lines)
        for ln in lines:
            if '//' in ln or '/*' in ln:
                comment_lines += 1
            avg_len_acc += float(len(ln))
            avg_len_cnt += 1
            if '#include' in ln:
                num_includes += 1
            if '#define' in ln:
                num_macros += 1
            array_count += ln.count('[')
        for k, pat in pragma_patterns.items():
            pragma_counts[k] += len(re.findall(pat, content, flags=re.IGNORECASE))
        pragma_total += sum(len(re.findall(pat, content, flags=re.IGNORECASE)) for pat in pragma_patterns.values())
        func_count += len(func_regex.findall(content))
        loop_count += len(loop_regex.findall(content))

    avg_len = (avg_len_acc / max(1, avg_len_cnt)) if avg_len_cnt > 0 else 0.0

    def lg(x: float) -> float:
        try:
            return float(torch.log1p(torch.tensor(float(x))).item())
        except Exception:
            return 0.0

    vec = [
        lg(total_lines),
        lg(num_files),
        lg(pragma_total),
        lg(pragma_counts['PIPELINE']),
        lg(pragma_counts['UNROLL']),
        lg(pragma_counts['ARRAY_PARTITION']),
        lg(pragma_counts['DATAFLOW']),
        lg(pragma_counts['INLINE']),
        lg(pragma_counts['INTERFACE']),
        lg(func_count),
        lg(loop_count),
        lg(array_count),
        lg(comment_lines),
        float(avg_len),
        lg(num_includes),
        lg(num_macros),
    ]
    return torch.tensor(vec, dtype=torch.float32)


class GraphAttributeDiffusionDataset(Dataset):
    """Full graph diffusion dataset (nodes + dense edges)."""

    def __init__(
        self,
        kernel_base_dir: str,
        design_base_dir: str,
        cache_root: str,
        rebuild_cache: bool = False,
        max_pairs: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.kernel_base_dir = kernel_base_dir
        self.design_base_dir = design_base_dir
        self.cache_root = cache_root
        self.rebuild_cache = rebuild_cache
        self.max_pairs = max_pairs
        self.seed = int(seed)

        self._records: List[Dict] = []
        self._feature_dim: int = 0
        self._edge_channels: int = 0
        self._code_feature_dim: int = 16
        self._code_cache: Dict[str, torch.Tensor] = {}
        self._load_and_prepare()

    def _load_and_prepare(self) -> None:
        processor = E2EDifferentialProcessor(
            kernel_base_dir=self.kernel_base_dir,
            design_base_dir=self.design_base_dir,
            output_dir=os.path.join(self.cache_root, "_gen_tmp"),
            cache_root=self.cache_root,
            rebuild_cache=self.rebuild_cache,
            hierarchical=False,
            max_workers=None,
        )
        records = processor.collect_all_data(materialize=False)
        if not records:
            self._records = []
            self._feature_dim = 0
            self._edge_channels = 0
            return

        g = torch.Generator()
        g.manual_seed(self.seed)
        perm = torch.randperm(len(records), generator=g).tolist()

        max_pairs = None
        if self.max_pairs is not None and int(self.max_pairs) > 0:
            max_pairs = int(self.max_pairs)
        if max_pairs is not None:
            perm = perm[:max_pairs]

        self._records = [records[i] for i in perm]

        # 预取一个样本以确定特征维度
        probe = self._load_pair_payload(self._records[0])
        design_graph = probe['design_graph']
        self._feature_dim = int(design_graph.x.size(1))
        edge_target, _ = self._build_dense_edges(design_graph)
        self._edge_channels = int(edge_target.size(-1))
        base_path = probe['design_info'].get('base_path') if probe.get('design_info') else None
        if base_path and os.path.exists(base_path):
            self._code_feature_dim = int(extract_code_features(base_path).numel())
        else:
            self._code_feature_dim = max(1, self._code_feature_dim)

    def _build_dense_edges(self, design_graph) -> Tuple[torch.Tensor, torch.Tensor]:
        num_nodes = int(design_graph.x.size(0))
        adjacency = torch.zeros((num_nodes, num_nodes, 1), dtype=torch.float32)
        edge_attr = getattr(design_graph, "edge_attr", None)
        attr_dim = int(edge_attr.size(1)) if edge_attr is not None else 0
        dense_attr = torch.zeros((num_nodes, num_nodes, attr_dim), dtype=torch.float32) if attr_dim > 0 else None

        if design_graph.edge_index is not None and design_graph.edge_index.numel() > 0:
            src_list = design_graph.edge_index[0].tolist()
            dst_list = design_graph.edge_index[1].tolist()
            for s, d in zip(src_list, dst_list):
                adjacency[s, d, 0] = 1.0
            if dense_attr is not None:
                for (s, d), attr in zip(design_graph.edge_index.t().tolist(), edge_attr.tolist()):
                    dense_attr[s, d] = torch.tensor(attr, dtype=torch.float32)

        edge_target = adjacency if dense_attr is None else torch.cat([adjacency, dense_attr], dim=-1)
        edge_mask = torch.ones((num_nodes, num_nodes), dtype=torch.bool)
        return edge_target, edge_mask

    @property
    def feature_dim(self) -> int:
        return int(self._feature_dim)

    @property
    def code_feature_dim(self) -> int:
        return int(self._code_feature_dim)

    @property
    def edge_channels(self) -> int:
        return int(self._edge_channels)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self._records[idx]
        payload = self._load_pair_payload(record)

        kg = payload['kernel_graph']
        dg = payload['design_graph']

        x_target = dg.x.detach().clone().float()
        kernel_global = kg.x.detach().clone().float().mean(dim=0, keepdim=True)
        x_cond = kernel_global.expand(x_target.size(0), -1).contiguous()

        design_base_path = payload.get('design_info', {}).get('base_path', record.get('design_base_path'))
        code_vec = self._lookup_code_features(design_base_path)
        if code_vec.numel() != self._code_feature_dim:
            code_vec = torch.zeros(self._code_feature_dim, dtype=torch.float32)
        code_cond = code_vec.unsqueeze(0).expand(x_target.size(0), -1).contiguous()

        edge_target, edge_mask = self._build_dense_edges(dg)

        return {
            "pair_id": payload.get("pair_id", record.get("pair_id", f"pair_{idx}")),
            "x_target": x_target,
            "x_cond": x_cond,
            "code_cond": code_cond,
            "edge_target": edge_target,
            "mask": torch.ones(x_target.size(0), dtype=torch.bool),
            "edge_mask": edge_mask,
            "kernel_graph": kg,
            "design_graph": dg,
            "pragma_count": torch.tensor([payload.get('pragma_info', {}).get('pragma_count', record.get('pragma_count', 0))], dtype=torch.long),
            "meta": {
                "design_base_path": design_base_path,
                "kernel_base_path": payload.get('kernel_info', {}).get('base_path', record.get('kernel_base_path')),
            },
        }

    def _load_pair_payload(self, record: Dict) -> Dict:
        pair_path = record.get('file')
        if pair_path is None:
            raise FileNotFoundError("Pair record does not contain file path")
        payload = torch.load(pair_path, map_location='cpu')
        return payload

    def _lookup_code_features(self, design_base_path: Optional[str]) -> torch.Tensor:
        if not design_base_path or not os.path.exists(design_base_path):
            return torch.zeros(self._code_feature_dim, dtype=torch.float32)
        cached = self._code_cache.get(design_base_path)
        if cached is not None:
            return cached
        code_vec = extract_code_features(design_base_path)
        self._code_cache[design_base_path] = code_vec
        return code_vec


def pad_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad variable-length node and edge tensors to a common shape."""
    if len(batch) == 0:
        return {
            "x_target": torch.empty(0, 0, 0),
            "x_cond": torch.empty(0, 0, 0),
            "code_cond": torch.empty(0, 0, 0),
            "mask": torch.empty(0, 0, dtype=torch.bool),
            "edge_target": torch.empty(0, 0, 0, 0),
            "edge_mask": torch.empty(0, 0, 0, dtype=torch.bool),
            "pair_ids": [],
            "kernel_graphs": [],
            "design_graphs": [],
            "metas": [],
        }

    feature_dim = batch[0]["x_target"].size(1)
    code_dim = batch[0]["code_cond"].size(1)
    edge_channels = batch[0]["edge_target"].size(-1)
    lengths = [b["x_target"].size(0) for b in batch]
    max_len = max(lengths)

    x_target_padded = []
    x_cond_padded = []
    code_cond_padded = []
    mask_padded = []
    edge_target_padded = []
    edge_mask_padded = []
    pair_ids: List[str] = []
    kernel_graphs = []
    design_graphs = []
    pragma_counts = []
    metas = []

    for item in batch:
        n = item["x_target"].size(0)
        padded_target = torch.zeros((max_len, feature_dim), dtype=item["x_target"].dtype)
        padded_target[:n] = item["x_target"]
        x_target_padded.append(padded_target)

        padded_cond = torch.zeros((max_len, feature_dim), dtype=item["x_cond"].dtype)
        padded_cond[:n] = item["x_cond"]
        x_cond_padded.append(padded_cond)

        padded_code = torch.zeros((max_len, code_dim), dtype=item["code_cond"].dtype)
        padded_code[:n] = item["code_cond"]
        code_cond_padded.append(padded_code)

        padded_mask = torch.zeros(max_len, dtype=torch.bool)
        padded_mask[:n] = item["mask"]
        mask_padded.append(padded_mask)

        padded_edge = torch.zeros((max_len, max_len, edge_channels), dtype=item["edge_target"].dtype)
        padded_edge[:n, :n] = item["edge_target"]
        edge_target_padded.append(padded_edge)

        padded_edge_mask = torch.zeros((max_len, max_len), dtype=torch.bool)
        padded_edge_mask[:n, :n] = item["edge_mask"]
        edge_mask_padded.append(padded_edge_mask)

        pair_ids.append(item["pair_id"])
        kernel_graphs.append(item["kernel_graph"])
        design_graphs.append(item["design_graph"])
        pragma_counts.append(item.get("pragma_count"))
        metas.append(item.get("meta", {}))

    return {
        "x_target": torch.stack(x_target_padded, dim=0),
        "x_cond": torch.stack(x_cond_padded, dim=0),
        "code_cond": torch.stack(code_cond_padded, dim=0),
        "mask": torch.stack(mask_padded, dim=0),
        "edge_target": torch.stack(edge_target_padded, dim=0),
        "edge_mask": torch.stack(edge_mask_padded, dim=0),
        "pair_ids": pair_ids,
        "kernel_graphs": kernel_graphs,
        "design_graphs": design_graphs,
        "pragma_counts": pragma_counts,
        "metas": metas,
    }
