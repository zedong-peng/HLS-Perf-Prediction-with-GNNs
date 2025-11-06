"""Dataset utilities for ProgSG reproduction on ForgeHLS.

This module reuses the graph parsing logic from delta_e2e to convert a design
folder (containing csynth.xml and .adb graph dumps) into PyTorch Geometric
`Data` objects together with QoR labels and pragma metadata.
"""

from __future__ import annotations

import json
import os
import sys
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import torch
from torch_geometric.data import Data
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Import graph parsing helpers from delta_e2e
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DELTA_UTILS_DIR = REPO_ROOT / "delta_e2e"
if not DELTA_UTILS_DIR.exists():
    raise ImportError(f"Cannot locate delta_e2e utilities at {DELTA_UTILS_DIR}")
if str(DELTA_UTILS_DIR) not in sys.path:
    sys.path.append(str(DELTA_UTILS_DIR))

from utils import (  # type: ignore
    parse_xml_into_graph_single,
    node_to_feature_vector,
    edge_to_feature_vector,
)


DEFAULT_CODE_EXTENSIONS: Tuple[str, ...] = (".c", ".cc", ".cpp", ".cxx", ".ino", ".cu")


@dataclass
class DesignSample:
    """Container for a single design graph and metadata."""

    graph: Data
    metrics: Dict[str, float]
    pragma_count: int
    source_name: str
    algo_name: str
    design_id: str
    extra_meta: Dict[str, object]
    code_text: str = ""
    code_inputs: Optional[Dict[str, torch.Tensor]] = None
    code_path: Optional[Path] = None
    code_sha1: Optional[str] = None


class DesignGraphProcessor:
    """Convert ForgeHLS design folders into PyG graphs with QoR targets."""

    def __init__(
        self,
        design_base_dir: Path,
        output_dir: Path,
        cache_root: Path,
        rebuild_cache: bool = False,
        hierarchical: bool = False,
        region: bool = False,
        max_workers: Optional[int] = None,
        include_code: bool = True,
        code_extensions: Iterable[str] = DEFAULT_CODE_EXTENSIONS,
        skip_if_no_code: bool = True,
        max_code_chars: Optional[int] = None,
    ) -> None:
        self.design_base_dir = design_base_dir
        self.output_dir = output_dir
        self.cache_root = cache_root
        self.rebuild_cache = rebuild_cache
        self.hierarchical = hierarchical
        self.region = region
        self.max_workers = max_workers or min(32, os.cpu_count() or 8)
        self.include_code = include_code
        self.skip_if_no_code = skip_if_no_code
        self.max_code_chars = max_code_chars
        ext_list: List[str] = []
        for ext in code_extensions:
            normalized = ext if ext.startswith(".") else f".{ext}"
            for variant in {normalized, normalized.lower(), normalized.upper()}:
                if variant not in ext_list:
                    ext_list.append(variant)
        if not ext_list:
            ext_list = list(DEFAULT_CODE_EXTENSIONS)
        self.code_extensions = tuple(ext_list)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)

        cache_key_src = "|".join(
            [
                str(self.design_base_dir.resolve()),
                f"hier={int(self.hierarchical)}",
                f"region={int(self.region)}",
                f"featv=20241005.0",
                f"code={int(self.include_code)}",
                f"codeext={'-'.join(sorted(self.code_extensions))}",
            ]
        )
        cache_digest = hashlib.md5(cache_key_src.encode("utf-8")).hexdigest()[:12]
        self.cache_dir = self.cache_root / f"design_cache_{cache_digest}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "design_graphs.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_designs(
        self,
        max_designs: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[DesignSample]:
        """Return a list of processed design samples, using cache if possible."""

        use_cache = max_designs is None

        if use_cache and self.cache_file.exists() and not self.rebuild_cache:
            return self._load_cache()

        design_dirs = self._enumerate_design_dirs()
        if not design_dirs:
            return []

        if max_designs is not None and max_designs > 0 and max_designs < len(design_dirs):
            rng = random.Random(seed)
            rng.shuffle(design_dirs)
            design_dirs = design_dirs[:max_designs]

        samples: List[DesignSample] = []

        def _process(task: Tuple[Path, str, str, str]) -> Optional[DesignSample]:
            design_dir, source_name, algo_name, design_id = task
            try:
                return self._collect_single_design(
                    design_dir, source_name, algo_name, design_id
                )
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(_process, task): task for task in design_dirs
            }
            with tqdm(total=len(future_to_task), desc="Parsing designs", ncols=100) as pbar:
                success = 0
                for future in as_completed(future_to_task):
                    result = future.result()
                    if result:
                        samples.append(result)
                        success += 1
                    pbar.update(1)
                    pbar.set_postfix(success=success, failed=pbar.n - success)

        if use_cache and samples:
            self._save_cache(samples)

        return samples

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enumerate_design_dirs(self) -> List[Tuple[Path, str, str, str]]:
        design_tasks: List[Tuple[Path, str, str, str]] = []
        if not self.design_base_dir.exists():
            return design_tasks

        for source_dir in sorted(self.design_base_dir.iterdir()):
            if not source_dir.is_dir():
                continue
            source_name = source_dir.name
            for algo_dir in sorted(source_dir.iterdir()):
                if not algo_dir.is_dir():
                    continue
                algo_name = algo_dir.name
                for design_dir in sorted(algo_dir.iterdir()):
                    if not design_dir.is_dir():
                        continue
                    if not design_dir.name.startswith("design_"):
                        continue
                    design_id = design_dir.name
                    design_tasks.append((design_dir, source_name, algo_name, design_id))
        return design_tasks

    def _collect_single_design(
        self,
        design_dir: Path,
        source_name: str,
        algo_name: str,
        design_id: str,
    ) -> Optional[DesignSample]:
        csynth_path = next(
            design_dir.glob("**/csynth.xml"),
            None,
        )
        if csynth_path is None:
            return None

        metrics = self._parse_csynth_xml(csynth_path)
        if metrics is None:
            return None

        graph_files = [
            p
            for p in design_dir.glob("**/*.adb")
            if p.is_file() and p.name.count(".") <= 1
        ]
        if not graph_files:
            return None

        graph = self._process_graphs(graph_files)
        if graph is None:
            return None

        graph.y = torch.tensor(
            [
                metrics["DSP"],
                metrics["LUT"],
                metrics["FF"],
                metrics["best_latency"],
            ],
            dtype=torch.float,
        ).unsqueeze(0)

        pragma_info = self._extract_pragma_info(design_dir)
        pragma_count = int(pragma_info.get("pragma_count", 0))

        # Attach lightweight tensor attributes for batching.
        graph.pragma_count = torch.tensor([pragma_count], dtype=torch.float)

        def _tensor_attr(name: str) -> float:
            value = getattr(graph, name, torch.tensor([0.0]))
            if isinstance(value, torch.Tensor):
                return float(value.view(-1)[0].item())
            return float(value)

        extra_meta: Dict[str, object] = {
            "has_pipeline": int(_tensor_attr("has_pipeline")),
            "pipeline_region_count": int(_tensor_attr("pipeline_region_count")),
            "avg_ii": _tensor_attr("avg_ii"),
            "max_pipe_depth": int(_tensor_attr("max_pipe_depth")),
        }

        code_text = ""
        code_path: Optional[Path] = None
        code_sha1: Optional[str] = None
        if self.include_code:
            code_path = self._find_source_code_file(design_dir)
            if code_path is None:
                if self.skip_if_no_code:
                    return None
            else:
                code_text = self._read_source_code(code_path)
                if self.max_code_chars is not None and self.max_code_chars > 0:
                    code_text = code_text[: self.max_code_chars]
                normalized = code_text.encode("utf-8", errors="ignore")
                code_sha1 = hashlib.sha1(normalized).hexdigest()

        sample = DesignSample(
            graph=graph,
            metrics=metrics,
            pragma_count=pragma_count,
            source_name=source_name,
            algo_name=algo_name,
            design_id=design_id,
            extra_meta=extra_meta,
            code_text=code_text,
            code_path=code_path,
            code_sha1=code_sha1,
        )

        if code_path is not None:
            try:
                extra_meta["code_rel_path"] = str(code_path.relative_to(design_dir))
            except ValueError:
                extra_meta["code_rel_path"] = str(code_path)
        if code_sha1 is not None:
            extra_meta["code_sha1"] = code_sha1

        return sample

    @staticmethod
    def _parse_csynth_xml(xml_path: Path) -> Optional[Dict[str, float]]:
        import xml.etree.ElementTree as ET

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            latency = root.find(
                ".//PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency"
            )
            dsp = root.find(".//AreaEstimates/Resources/DSP")
            lut = root.find(".//AreaEstimates/Resources/LUT")
            ff = root.find(".//AreaEstimates/Resources/FF")
            if (
                latency is None
                or latency.text in {None, "undef"}
                or dsp is None
                or lut is None
                or ff is None
            ):
                return None
            return {
                "best_latency": float(latency.text),
                "DSP": float(dsp.text),
                "LUT": float(lut.text),
                "FF": float(ff.text),
            }
        except Exception:
            return None

    def _process_graphs(self, graph_files: Sequence[Path]) -> Optional[Data]:
        graphs: List[Data] = []
        for adb_path in graph_files:
            g = self._process_graph_file(adb_path)
            if g is not None:
                graphs.append(g)
        if not graphs:
            return None
        if len(graphs) == 1:
            return graphs[0]
        # Merge multiple graphs by disjoint union.
        from torch_geometric.data import Batch

        batch = Batch.from_data_list(graphs)
        merged = Data(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
        )
        for attr in [
            "has_pipeline",
            "pipeline_region_count",
            "avg_ii",
            "max_pipe_depth",
            "pipeline_components_present",
            "pipeline_signals_present",
        ]:
            values = [float(getattr(g, attr, torch.tensor([0.0])).item()) for g in graphs]
            merged_value = torch.tensor([sum(values)], dtype=torch.float)
            setattr(merged, attr, merged_value)
        return merged

    def _process_graph_file(self, adb_path: Path) -> Optional[Data]:
        try:
            g_nx = self._parse_graph(adb_path)
        except Exception:
            return None
        if g_nx is None or g_nx.number_of_nodes() == 0:
            return None

        nodes = list(g_nx.nodes())
        node_map = {node: idx for idx, node in enumerate(nodes)}

        node_features = [
            node_to_feature_vector(g_nx.nodes[node]) for node in nodes
        ]
        edge_index = []
        edge_features = []
        for src, dst in g_nx.edges():
            if src not in node_map or dst not in node_map:
                continue
            edge_index.append([node_map[src], node_map[dst]])
            edge_features.append(edge_to_feature_vector(g_nx.edges[(src, dst)]))

        if not edge_index:
            return None

        data = Data()
        data.x = torch.tensor(node_features, dtype=torch.long)
        data.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        data.edge_attr = torch.tensor(edge_features, dtype=torch.long)

        # Propagate known pipeline metadata (default to zero if absent).
        pipeline_attrs = [
            "has_pipeline",
            "pipeline_region_count",
            "avg_ii",
            "max_pipe_depth",
            "pipeline_components_present",
            "pipeline_signals_present",
        ]
        for attr in pipeline_attrs:
            value = g_nx.graph.get(attr, 0)
            setattr(data, attr, torch.tensor([value], dtype=torch.float))

        return data

    def _parse_graph(self, adb_path: Path):
        hierarchical = bool(self.hierarchical)
        region = bool(self.region)
        try:
            return parse_xml_into_graph_single(
                str(adb_path), hierarchical=hierarchical, region=region
            )
        except TypeError:
            return parse_xml_into_graph_single(str(adb_path), hierarchical=hierarchical)

    # ------------------------------------------------------------------
    # Source code helpers
    # ------------------------------------------------------------------

    def _iter_source_files(self, design_path: Path) -> Iterable[Path]:
        seen = set()
        for ext in self.code_extensions:
            pattern = f"**/*{ext}"
            for candidate in design_path.glob(pattern):
                if not candidate.is_file():
                    continue
                resolved = candidate.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                yield candidate

    def _find_source_code_file(self, design_path: Path) -> Optional[Path]:
        candidates = list(self._iter_source_files(design_path))
        if not candidates:
            return None

        def candidate_key(path: Path) -> Tuple[int, int, str]:
            try:
                rel = path.relative_to(design_path)
                depth = len(rel.parts)
            except ValueError:
                depth = 999
            return (depth, len(path.name), path.name)

        candidates.sort(key=candidate_key)

        top_function = self._read_top_function_name(design_path)
        if top_function:
            for cand in candidates:
                if cand.stem == top_function:
                    return cand

        return candidates[0]

    @staticmethod
    def _read_top_function_name(design_path: Path) -> Optional[str]:
        top_file = design_path / "top_function_name.txt"
        if not top_file.exists():
            return None
        try:
            content = top_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None
        for line in content.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
        return None

    @staticmethod
    def _read_source_code(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""

    def _extract_pragma_info(self, design_path: Path) -> Dict[str, int]:
        pragma_info = {
            "pragma_count": 0,
            "pipeline_pragmas": 0,
            "unroll_pragmas": 0,
            "array_partition_pragmas": 0,
            "inline_pragmas": 0,
            "dataflow_pragmas": 0,
            "resource_pragmas": 0,
            "latency_pragmas": 0,
            "dependence_pragmas": 0,
            "other_pragmas": 0,
        }

        pragma_keywords = {
            "pipeline": "pipeline_pragmas",
            "unroll": "unroll_pragmas",
            "array_partition": "array_partition_pragmas",
            "inline": "inline_pragmas",
            "dataflow": "dataflow_pragmas",
            "resource": "resource_pragmas",
            "latency": "latency_pragmas",
            "dependence": "dependence_pragmas",
        }
        for source_file in self._iter_source_files(design_path):
            try:
                content = source_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for line in content.splitlines():
                if "#pragma" in line:
                    pragma_info["pragma_count"] += 1
                    lowered = line.lower()
                    matched = False
                    for keyword, key in pragma_keywords.items():
                        if keyword in lowered:
                            pragma_info[key] += 1
                            matched = True
                            break
                    if not matched:
                        pragma_info["other_pragmas"] += 1

        return pragma_info

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _save_cache(self, samples: Sequence[DesignSample]) -> None:
        serialisable = []
        for sample in samples:
            graph = sample.graph
            serialisable.append(
                {
                    "metrics": sample.metrics,
                    "pragma_count": sample.pragma_count,
                    "source_name": sample.source_name,
                    "algo_name": sample.algo_name,
                    "design_id": sample.design_id,
                    "extra_meta": sample.extra_meta,
                    "code_text": sample.code_text,
                    "code_sha1": sample.code_sha1,
                    "code_path": str(sample.code_path) if sample.code_path is not None else None,
                    "graph": {
                        "x": graph.x.tolist(),
                        "edge_index": graph.edge_index.tolist(),
                        "edge_attr": graph.edge_attr.tolist()
                        if graph.edge_attr is not None
                        else None,
                    },
                }
            )

        with self.cache_file.open("w", encoding="utf-8") as fh:
            json.dump(serialisable, fh)

    def _load_cache(self) -> List[DesignSample]:
        samples: List[DesignSample] = []
        with self.cache_file.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        for item in tqdm(payload, desc="Loading cached designs", ncols=100):
            graph_dict = item["graph"]
            graph = Data()
            graph.x = torch.tensor(graph_dict["x"], dtype=torch.long)
            graph.edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
            if graph_dict.get("edge_attr") is not None:
                graph.edge_attr = torch.tensor(graph_dict["edge_attr"], dtype=torch.long)
            else:
                graph.edge_attr = None

            metrics = item["metrics"]
            graph.y = torch.tensor(
                [
                    metrics["DSP"],
                    metrics["LUT"],
                    metrics["FF"],
                    metrics["best_latency"],
                ],
                dtype=torch.float,
            ).unsqueeze(0)
            pragma_count = int(item["pragma_count"])
            graph.pragma_count = torch.tensor([pragma_count], dtype=torch.float)

            # Restore pipeline/context scalar attributes from extra_meta when available
            extra_meta = item.get("extra_meta", {})
            def _restore_scalar(name: str) -> None:
                if name in extra_meta:
                    value = float(extra_meta[name])
                    setattr(graph, name, torch.tensor([value], dtype=torch.float))
            for key in [
                "has_pipeline",
                "pipeline_region_count",
                "avg_ii",
                "max_pipe_depth",
            ]:
                _restore_scalar(key)

            samples.append(
                DesignSample(
                    graph=graph,
                    metrics={k: float(v) for k, v in metrics.items()},
                    pragma_count=pragma_count,
                    source_name=item["source_name"],
                    algo_name=item["algo_name"],
                    design_id=item["design_id"],
                    extra_meta=extra_meta,
                    code_text=item.get("code_text", ""),
                    code_path=Path(item["code_path"]) if item.get("code_path") else None,
                    code_sha1=item.get("code_sha1"),
                )
            )
        return samples


def assign_sample_indices(samples: Sequence[DesignSample]) -> None:
    """Add a `sample_id` tensor attribute to each graph for tracking."""

    for idx, sample in enumerate(samples):
        sample.graph.sample_id = torch.tensor([idx], dtype=torch.long)


class CodeFeatureProvider(Protocol):
    def prepare_sample(self, sample: DesignSample, encode: bool = True) -> Dict[str, torch.Tensor]:
        ...


def attach_code_features(
    samples: Sequence[DesignSample],
    provider: CodeFeatureProvider,
    encode: bool = True,
) -> None:
    """Populate `code_inputs` for each sample using the given provider."""

    for sample in samples:
        code_text = sample.code_text
        code_path = sample.code_path

        if not code_text and code_path is not None:
            try:
                if code_path.is_file():
                    code_text = code_path.read_text(encoding="utf-8", errors="ignore")
                else:
                    code_path = None
            except OSError:
                code_path = None

        if not code_text:
            sample.code_path = None
            continue

        if not sample.code_text:
            sample.code_text = code_text

        sample.code_path = code_path

        payload = provider.prepare_sample(sample, encode=encode)
        copied: Dict[str, torch.Tensor] = {}
        for key, value in payload.items():
            if isinstance(value, torch.Tensor):
                copied[key] = value.clone()
        sample.code_inputs = copied
