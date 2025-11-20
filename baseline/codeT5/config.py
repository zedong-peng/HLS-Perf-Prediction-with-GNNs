from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CodeT5Config:
    metric_name: str
    design_root: str
    ood_design_root: str | None = None
    cache_dir: Path = field(default_factory=lambda: Path("baseline/codeT5/cache"))
    output_dir: Path = field(default_factory=lambda: Path("baseline/codeT5/artifacts"))
    dataset_name: str = "forgehls_lite_100designs"
    ood_dataset_name: str | None = None
    rebuild_cache: bool = False
    epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    patience: int = 20
    min_delta: float = 1e-4
    seed: int = 42
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    max_designs: int | None = None
    max_ood_designs: int | None = None
    num_workers: int = 0
    device: str = "cuda"
    hidden_dim: int = 512
    mlp_hidden: int = 256
    dropout: float = 0.1
    tokenizer_name: str = "Salesforce/codet5-small"
    encoder_name: str = "Salesforce/codet5-small"
    max_tokens: int = 256
    store_token_embeddings: bool = False
    store_pooled: bool = True
    no_swanlab: bool = False
    swan_project: str = "CodeT5-Baseline"
    swan_prefix: str = "codet5"

    def artifact_dir(self, run_id: str) -> Path:
        path = self.output_dir / self.metric_name / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def cache_path(self, dataset_name: str | None = None) -> Path:
        name = dataset_name or self.dataset_name
        cache_root = Path(self.cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        return cache_root / f"{name}.pt"
