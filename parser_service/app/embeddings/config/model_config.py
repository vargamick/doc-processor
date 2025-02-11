from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch


@dataclass
class ModelConfig:
    """Configuration for embedding models"""

    model_name: str = "all-MiniLM-L6-v2"  # Default model
    max_seq_length: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    normalize_embeddings: bool = True
    pooling_strategy: str = "mean"  # mean, max, cls
    cache_dir: Optional[str] = None
    model_kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.model_kwargs is None:
            self.model_kwargs = {}
