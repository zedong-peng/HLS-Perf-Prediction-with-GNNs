import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional, Union


class LLMEmbedder:
    """
    Thin wrapper to turn design code into a fixed-length embedding using a local LLM.

    Usage:
        embedder = LLMEmbedder("/home/user/zedongpeng/workspace/GiT/zedong/Code-Verification/Qwen/Qwen2.5-Coder-1.5B-Instruct")
        vectors = embedder.encode(["#include <...>\\nint main() {...}"])
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        max_length: int = 2048,
        pooling: str = "last_token",
        normalize: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = dtype or (torch.float16 if self.device.startswith("cuda") else torch.float32)
        self.max_length = max_length
        self.pooling = pooling
        self.normalize = normalize

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
        ).to(self.device)
        self.model.eval()

    def encode(self, texts: Union[List[str], str]) -> torch.Tensor:
        """Return a tensor of shape (batch, hidden_size) with pooled embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded, output_hidden_states=True, return_dict=True)

        hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden_size)
        embeddings = self._pool(hidden_states, encoded["attention_mask"])
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def encode_to_numpy(self, texts: Union[List[str], str]):
        """Helper for downstream code that prefers numpy arrays."""
        return self.encode(texts).detach().cpu().numpy()

    def _pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool token representations into a single vector."""
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1)
            summed = (hidden_states * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            return summed / counts

        # default: use the last non-padding token
        last_indices = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, last_indices]
