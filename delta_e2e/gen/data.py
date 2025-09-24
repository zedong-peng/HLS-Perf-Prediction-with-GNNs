import os
import json
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset
import re

from delta_e2e.train_e2e import E2EDifferentialProcessor


def _safe_read(path: str) -> str:
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception:
        return ""


def extract_code_features(code_root: str) -> torch.Tensor:
    """Extract a small, fixed-size vector of kernel code statistics as condition.

    This is a lightweight PoC: counts heuristics over C/C++ files.
    Returns a float tensor of shape [16].
    """
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

    # Build vector and apply mild log scaling to counts
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
    """Dataset that provides paired node features (kernel -> design) for attribute diffusion.

    This PoC broadcasts a pooled kernel condition vector to the design's node count,
    and adds a kernel code statistics vector as an extra condition.
    """

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

        self._pairs: List[Dict] = []
        self._examples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]] = []
        self._feature_dim: int = 0
        self._code_feature_dim: int = 16
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
        pairs = processor.collect_all_data()
        if not pairs:
            self._pairs = []
            self._examples = []
            return

        # Deterministic shuffle
        g = torch.Generator()
        g.manual_seed(self.seed)
        perm = torch.randperm(len(pairs), generator=g).tolist()
        if self.max_pairs is not None:
            perm = perm[: int(self.max_pairs)]

        filtered: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]] = []
        for idx in perm:
            p = pairs[idx]
            kg = p["kernel_graph"]
            dg = p["design_graph"]
            if not hasattr(kg, "x") or not hasattr(dg, "x"):
                continue
            if kg.x is None or dg.x is None:
                continue
            if kg.x.dim() != 2 or dg.x.dim() != 2:
                continue
            if kg.x.size(1) != dg.x.size(1):
                continue
            # Prepare target design node features
            x_target = dg.x.detach().clone().float()  # [N_d, F]
            # Pooled kernel condition vector -> broadcast to design nodes
            kernel_global = kg.x.detach().clone().float().mean(dim=0, keepdim=True)  # [1, F]
            x_cond = kernel_global.expand(x_target.size(0), -1).contiguous()  # [N_d, F]
            # Kernel code condition vector -> broadcast per-node
            kernel_base_path = p.get('kernel_info', {}).get('base_path', None)
            if kernel_base_path is None:
                code_vec = torch.zeros(self._code_feature_dim, dtype=torch.float32)
            else:
                code_vec = extract_code_features(kernel_base_path)
                if code_vec.numel() != self._code_feature_dim:
                    code_vec = torch.zeros(self._code_feature_dim, dtype=torch.float32)
            code_cond = code_vec.unsqueeze(0).expand(x_target.size(0), -1).contiguous()  # [N_d, C]
            filtered.append((x_target, x_cond, code_cond, p.get("pair_id", f"pair_{idx}")))

        self._pairs = pairs
        self._examples = filtered
        self._feature_dim = int(filtered[0][0].size(1)) if filtered else 0

    @property
    def feature_dim(self) -> int:
        return int(self._feature_dim)

    @property
    def code_feature_dim(self) -> int:
        return int(self._code_feature_dim)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x_target, x_cond, code_cond, pair_id = self._examples[idx]
        mask = torch.ones(x_target.size(0), dtype=torch.bool)
        return {
            "x_target": x_target,      # [N_d, F]
            "x_cond": x_cond,          # [N_d, F]
            "code_cond": code_cond,    # [N_d, C]
            "mask": mask,              # [N_d]
            "pair_id": pair_id,
        }


def pad_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad variable-length node sequences in a batch to the same length."""
    if len(batch) == 0:
        return {
            "x_target": torch.empty(0, 0, 0),
            "x_cond": torch.empty(0, 0, 0),
            "code_cond": torch.empty(0, 0, 0),
            "mask": torch.empty(0, 0, dtype=torch.bool),
            "pair_ids": [],
        }

    feature_dim = batch[0]["x_target"].size(1)
    code_dim = batch[0]["code_cond"].size(1)
    lengths = [b["x_target"].size(0) for b in batch]
    max_len = max(lengths)

    x_target_padded = []
    x_cond_padded = []
    code_cond_padded = []
    mask_padded = []
    pair_ids: List[str] = []

    for item in batch:
        n = item["x_target"].size(0)
        pad_n = max_len - n
        if pad_n > 0:
            pad_target = torch.zeros(pad_n, feature_dim, dtype=item["x_target"].dtype)
            pad_cond = torch.zeros(pad_n, feature_dim, dtype=item["x_cond"].dtype)
            pad_code = torch.zeros(pad_n, code_dim, dtype=item["code_cond"].dtype)
            pad_mask = torch.zeros(pad_n, dtype=torch.bool)
            x_target_padded.append(torch.cat([item["x_target"], pad_target], dim=0))
            x_cond_padded.append(torch.cat([item["x_cond"], pad_cond], dim=0))
            code_cond_padded.append(torch.cat([item["code_cond"], pad_code], dim=0))
            mask_padded.append(torch.cat([item["mask"], pad_mask], dim=0))
        else:
            x_target_padded.append(item["x_target"])  # already max_len
            x_cond_padded.append(item["x_cond"])      # already max_len
            code_cond_padded.append(item["code_cond"])# already max_len
            mask_padded.append(item["mask"])          # already max_len
        pair_ids.append(item["pair_id"])

    return {
        "x_target": torch.stack(x_target_padded, dim=0),   # [B, max_N, F]
        "x_cond": torch.stack(x_cond_padded, dim=0),       # [B, max_N, F]
        "code_cond": torch.stack(code_cond_padded, dim=0), # [B, max_N, C]
        "mask": torch.stack(mask_padded, dim=0),           # [B, max_N]
        "pair_ids": pair_ids,
    } 