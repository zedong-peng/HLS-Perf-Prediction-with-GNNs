from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, T5EncoderModel

from .config import CodeT5Config


TARGET_KEYS = ["lut", "ff", "dsp"]


@dataclass
class CodeSample:
    pooled: torch.Tensor
    target: torch.Tensor
    metadata: Dict[str, str]


class CodeT5Dataset(Dataset):
    def __init__(self, samples: List[CodeSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            "pooled": sample.pooled,
            "target": sample.target,
        }

    def metadata(self, idx: int) -> Dict[str, str]:
        return self.samples[idx].metadata


def _find_csynth(design_path: Path) -> Optional[Path]:
    for candidate in design_path.glob("project/**/csynth.xml"):
        if candidate.is_file():
            return candidate
    return None


def _parse_metrics(csynth_xml: Path) -> Dict[str, float]:
    import xml.etree.ElementTree as ET

    tree = ET.parse(csynth_xml)
    root = tree.getroot()

    def grab(xpath: str) -> float:
        node = root.find(xpath)
        if node is None or node.text is None:
            raise ValueError(f"Missing node {xpath}")
        return float(node.text)

    return {
        "lut": grab(".//AreaEstimates/Resources/LUT"),
        "ff": grab(".//AreaEstimates/Resources/FF"),
        "dsp": grab(".//AreaEstimates/Resources/DSP"),
        "latency": grab(".//PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency"),
    }


def _locate_source(design_path: Path) -> Optional[Path]:
    for ext in (".c", ".cpp", ".cu", ".h", ".hpp"):
        matches = sorted(design_path.glob(f"**/*{ext}"))
        if matches:
            return matches[0]
    return None


def _iter_designs(design_root: Path, max_designs: Optional[int]) -> Iterable[Tuple[Path, Dict[str, float]]]:
    count = 0
    for source_dir in sorted(design_root.iterdir()):
        if not source_dir.is_dir():
            continue
        for algo_dir in sorted(source_dir.iterdir()):
            if not algo_dir.is_dir():
                continue
            for design_dir in sorted(algo_dir.iterdir()):
                if not design_dir.is_dir() or not design_dir.name.startswith("design_"):
                    continue
                csynth = _find_csynth(design_dir)
                if csynth is None:
                    continue
                try:
                    metrics = _parse_metrics(csynth)
                except Exception:
                    continue
                yield design_dir, metrics
                count += 1
                if max_designs is not None and count >= max_designs:
                    return


class CodeEncoder:
    def __init__(self, config: CodeT5Config) -> None:
        device = config.device if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name,
            use_fast=True,
            truncation=True,
            padding="max_length",
            model_max_length=config.max_tokens,
        )
        self.encoder = T5EncoderModel.from_pretrained(config.encoder_name)
        self.encoder.to(self.device)
        self.encoder.eval()
        self.max_tokens = config.max_tokens

    def encode(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_tokens,
        )
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=tokens["input_ids"].to(self.device),
                attention_mask=tokens["attention_mask"].to(self.device),
            )
        hidden = outputs.last_hidden_state.squeeze(0).cpu()
        mask = tokens["attention_mask"].squeeze(0).to(torch.float32).unsqueeze(1)
        pooled = (hidden * mask).sum(dim=0)
        denom = mask.sum().clamp(min=1.0)
        return (pooled / denom).to(torch.float32)


def _build_samples(config: CodeT5Config, design_root: str, dataset_name: str, max_designs: Optional[int]) -> List[CodeSample]:
    cache_path = config.cache_path(dataset_name)
    if cache_path.exists() and not config.rebuild_cache:
        payload = torch.load(cache_path, map_location="cpu")
        return [
            CodeSample(
                pooled=item["pooled"],
                target=item["target"],
                metadata=item["metadata"],
            )
            for item in payload
        ]

    design_dir = Path(design_root)
    encoder = CodeEncoder(config)
    samples: List[CodeSample] = []

    for design_path, metrics in _iter_designs(design_dir, max_designs):
        source_file = _locate_source(design_path)
        if source_file is None:
            continue
        code_text = source_file.read_text(encoding="utf-8", errors="ignore")
        pooled = encoder.encode(code_text)
        target = torch.tensor([metrics[key] for key in TARGET_KEYS], dtype=torch.float32)
        metadata = {
            "source": design_path.parent.parent.name,
            "algo": design_path.parent.name,
            "design_id": design_path.name,
        }
        samples.append(CodeSample(pooled=pooled, target=target, metadata=metadata))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        [
            {"pooled": sample.pooled, "target": sample.target, "metadata": sample.metadata}
            for sample in samples
        ],
        cache_path,
    )
    return samples


def split_indices(total: int, train_ratio: float, val_ratio: float, seed: int) -> Dict[str, torch.Tensor]:
    if total == 0:
        empty = torch.empty(0, dtype=torch.long)
        return {"train": empty, "valid": empty, "test": empty}
    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(total, generator=gen)
    train_sz = int(total * train_ratio)
    val_sz = int(total * val_ratio)
    test_sz = total - train_sz - val_sz
    if test_sz <= 0:
        test_sz = max(1, total - train_sz - val_sz)
        val_sz = max(0, val_sz - (test_sz - (total - train_sz - val_sz)))
    train_idx = perm[:train_sz]
    val_idx = perm[train_sz:train_sz + val_sz]
    test_idx = perm[train_sz + val_sz:]
    return {"train": train_idx, "valid": val_idx, "test": test_idx}


def create_dataloaders(
    config: CodeT5Config,
) -> Tuple[CodeT5Dataset, Dict[str, torch.Tensor], Dict[str, DataLoader], Optional[CodeT5Dataset], Optional[DataLoader]]:
    samples = _build_samples(config, config.design_root, config.dataset_name, config.max_designs)
    dataset = CodeT5Dataset(samples)

    indices = split_indices(len(dataset), config.train_ratio, config.val_ratio, config.seed)

    def _make_loader(idx: torch.Tensor, shuffle: bool) -> Optional[DataLoader]:
        if idx.numel() == 0:
            return None
        subset = torch.utils.data.Subset(dataset, idx.tolist())
        return DataLoader(
            subset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    loaders = {
        "train": _make_loader(indices["train"], shuffle=True),
        "valid": _make_loader(indices["valid"], shuffle=False),
        "test": _make_loader(indices["test"], shuffle=False),
    }

    if loaders["train"] is None:
        raise ValueError("Training split is empty; adjust ratios or dataset size")

    ood_dataset: Optional[CodeT5Dataset] = None
    ood_loader: Optional[DataLoader] = None
    if config.ood_design_root:
        dataset_name = config.ood_dataset_name or Path(config.ood_design_root).name
        ood_samples = _build_samples(config, config.ood_design_root, dataset_name, config.max_ood_designs)
        if ood_samples:
            ood_dataset = CodeT5Dataset(ood_samples)
            ood_loader = DataLoader(
                ood_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=torch.cuda.is_available(),
            )

    return dataset, indices, loaders, ood_dataset, ood_loader
