"""Source code tokenisation and encoding utilities for ProgSG multimodal runs.

This module streamlines creating CodeT5 features for ForgeHLS designs. It
handles tokenisation, optional encoder forward passes, and persistent caching
so repeated experiments do not repeatedly re-encode the same kernels.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from transformers import AutoTokenizer, T5EncoderModel

from dataset import DesignSample


@dataclass(frozen=True)
class CodeEncoderConfig:
    """Configuration for source-code modality processing."""

    model_name_or_path: str = "Salesforce/codet5-small"
    max_length: int = 256
    padding: str = "max_length"
    truncation: bool = True
    local_files_only: bool = False
    dtype: torch.dtype = torch.float32
    cache_namespace: str = "codet5_small_v1"
    store_token_embeddings: bool = True
    store_pooled: bool = True

    def digest(self) -> str:
        payload = {
            "model": self.model_name_or_path,
            "max_length": self.max_length,
            "padding": self.padding,
            "trunc": int(self.truncation),
            "dtype": str(self.dtype),
            "ns": self.cache_namespace,
            "store_tokens": int(self.store_token_embeddings),
            "store_pooled": int(self.store_pooled),
        }
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.md5(raw).hexdigest()


class SourceCodeFeatureCache:
    """Cache manager that materialises CodeT5 features for design samples."""

    def __init__(
        self,
        cache_root: Path,
        config: Optional[CodeEncoderConfig] = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config or CodeEncoderConfig()
        self.cache_root = Path(cache_root)
        self.device = torch.device(device)

        digest = self.config.digest()
        self.cache_dir = self.cache_root / f"code_features_{digest}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._tokenizer = None
        self._encoder: Optional[nn.Module] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name_or_path,
                use_fast=True,
                padding=self.config.padding,
                truncation=self.config.truncation,
                model_max_length=self.config.max_length,
                local_files_only=self.config.local_files_only,
            )
        return self._tokenizer

    @property
    def encoder(self) -> T5EncoderModel:
        if self._encoder is None:
            encoder = T5EncoderModel.from_pretrained(
                self.config.model_name_or_path,
                local_files_only=self.config.local_files_only,
            )
            encoder.to(self.device)
            encoder.eval()
            self._encoder = encoder
        return self._encoder  # type: ignore[return-value]

    def prepare_sample(self, sample: DesignSample, encode: bool = True) -> Dict[str, torch.Tensor]:
        """Ensure features for the given sample are cached and return them."""

        if sample.code_text == "" and sample.code_path is None:
            raise ValueError("Sample does not contain source code text to encode")

        cache_path = self._feature_path(sample)
        if cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu")
            return self._cast_payload(payload)

        payload = self._build_payload(sample, encode)
        torch.save(payload, cache_path)
        return payload

    def _build_payload(self, sample: DesignSample, encode: bool) -> Dict[str, torch.Tensor]:
        code_text = sample.code_text
        if not code_text and sample.code_path is not None:
            code_text = sample.code_path.read_text(encoding="utf-8", errors="ignore")
        if not code_text:
            raise ValueError("No code text available after attempting to read file")

        token_output = self.tokenizer(
            code_text,
            return_tensors="pt",
            padding=self.config.padding,
            truncation=self.config.truncation,
            max_length=self.config.max_length,
        )
        input_ids = token_output["input_ids"].squeeze(0)
        attention_mask = token_output["attention_mask"].squeeze(0)

        payload: Dict[str, torch.Tensor] = {
            "input_ids": input_ids.to(torch.long),
            "attention_mask": attention_mask.to(torch.long),
        }

        if encode:
            encoder = self.encoder
            with torch.no_grad():
                outputs = encoder(
                    input_ids=input_ids.unsqueeze(0).to(self.device),
                    attention_mask=attention_mask.unsqueeze(0).to(self.device),
                )
            hidden = outputs.last_hidden_state.squeeze(0).to(torch.float32)
            if hidden.dtype != self.config.dtype:
                hidden = hidden.to(self.config.dtype)
            if self.config.store_token_embeddings:
                payload["token_embeddings"] = hidden.cpu()
            if self.config.store_pooled:
                mask = attention_mask.unsqueeze(-1).to(hidden.device)
                mask = mask.to(hidden.dtype)
                denom = mask.sum(dim=0).clamp_min(1.0)
                pooled = (hidden * mask).sum(dim=0) / denom
                payload["pooled_embedding"] = pooled.cpu()

        return payload

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _feature_path(self, sample: DesignSample) -> Path:
        if sample.code_sha1:
            name = sample.code_sha1
        else:
            hasher = hashlib.sha1()
            content = sample.code_text.encode("utf-8", errors="ignore")
            hasher.update(content)
            name = hasher.hexdigest()
        return self.cache_dir / f"{name}.pt"

    @staticmethod
    def _cast_payload(payload: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for key, value in list(payload.items()):
            if isinstance(value, torch.Tensor):
                payload[key] = value.clone().detach()
        return payload


__all__ = [
    "CodeEncoderConfig",
    "SourceCodeFeatureCache",
]
