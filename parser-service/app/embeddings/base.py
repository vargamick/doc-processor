from typing import List, Optional, Dict, Any, Union, Sequence, cast, TypeVar, overload
import torch
from sentence_transformers import SentenceTransformer
import redis
import json
import hashlib
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel


# Type aliases for better readability
T = TypeVar("T", bound=Union[float, np.float32, np.float64])
FloatArray = NDArray[np.float32]
FloatTensor = torch.Tensor
SingleTensor = Union[FloatTensor, FloatArray]
TensorList = List[FloatTensor]
ModelOutput = Union[SingleTensor, TensorList]
BatchOutput = Union[SingleTensor, TensorList]


class EmbeddingConfig(BaseModel):
    """Configuration for the embedding processor"""

    model_name: str = (
        "all-MiniLM-L6-v2"  # Default model, good balance of speed and quality
    )
    batch_size: int = 32
    max_seq_length: int = 256
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour cache TTL
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class EmbeddingProcessor:
    """Base class for generating and managing embeddings"""

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.model = self._initialize_model()
        self.cache = self._initialize_cache() if self.config.cache_enabled else None

    def _initialize_model(self) -> SentenceTransformer:
        """Initialize the sentence transformer model"""
        try:
            model = SentenceTransformer(self.config.model_name)
            model.to(self.config.device)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embedding model: {str(e)}")

    def _initialize_cache(self) -> Optional[redis.Redis]:
        """Initialize Redis cache connection"""
        try:
            import os

            return redis.Redis(
                host=os.getenv("REDIS_HOST", "redis"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                db=int(os.getenv("REDIS_DB", "0")),
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
        except Exception as e:
            print(f"Warning: Failed to initialize Redis cache: {str(e)}")
            return None

    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for the given text"""
        return f"emb:{hashlib.md5(text.encode()).hexdigest()}"

    @overload
    def _tensor_to_list(self, tensor: FloatTensor) -> List[float]: ...

    @overload
    def _tensor_to_list(self, tensor: List[FloatTensor]) -> List[List[float]]: ...

    def _tensor_to_list(
        self, tensor: Union[FloatTensor, List[FloatTensor]]
    ) -> Union[List[float], List[List[float]]]:
        """Convert PyTorch tensor or list of tensors to list format"""
        if isinstance(tensor, list):
            return [t.detach().cpu().numpy().tolist() for t in tensor]
        return tensor.detach().cpu().numpy().tolist()

    def _array_to_list(self, array: FloatArray) -> List[float]:
        """Convert numpy array to list of floats"""
        return array.tolist()

    def _ensure_device_compatibility(self, tensor: FloatTensor) -> FloatTensor:
        """Ensure tensor is on the correct device"""
        if tensor.device.type != self.config.device:
            return tensor.to(self.config.device)
        return tensor

    def _convert_single_output(self, output: ModelOutput) -> List[float]:
        """Convert single model output to list format"""
        try:
            if isinstance(output, torch.Tensor):
                return self._tensor_to_list(self._ensure_device_compatibility(output))
            if isinstance(output, np.ndarray):
                return self._array_to_list(output)
            if isinstance(output, list) and all(
                isinstance(t, torch.Tensor) for t in output
            ):
                # Handle case where model returns list of tensors for single input
                tensors = cast(List[FloatTensor], output)
                # Concatenate tensors if multiple are returned
                combined = torch.cat(tensors, dim=0)
                return self._tensor_to_list(combined)
            raise TypeError(f"Unsupported output type: {type(output)}")
        except Exception as e:
            raise RuntimeError(f"Failed to convert model output: {str(e)}")

    def _convert_batch_output(self, output: BatchOutput) -> List[List[float]]:
        """Convert batch model output to list format"""
        try:
            if isinstance(output, torch.Tensor):
                return cast(
                    List[List[float]],
                    self._tensor_to_list(self._ensure_device_compatibility(output)),
                )
            if isinstance(output, np.ndarray):
                return cast(List[List[float]], self._array_to_list(output))
            if isinstance(output, list) and all(
                isinstance(t, torch.Tensor) for t in output
            ):
                # Handle list of tensors
                return cast(List[List[float]], self._tensor_to_list(output))
            raise TypeError(f"Unsupported batch output type: {type(output)}")
        except Exception as e:
            raise RuntimeError(f"Failed to convert batch output: {str(e)}")

    def _get_from_cache(self, text: str) -> Optional[List[float]]:
        """Retrieve embeddings from cache if available"""
        if not self.cache:
            return None

        try:
            cached_value = self.cache.get(self._get_cache_key(text))
            if isinstance(cached_value, str):
                return cast(List[float], json.loads(cached_value))
        except Exception as e:
            print(f"Warning: Cache retrieval failed: {str(e)}")
        return None

    def _store_in_cache(self, text: str, embedding: List[float]) -> None:
        """Store embeddings in cache"""
        if not self.cache:
            return

        try:
            self.cache.setex(
                self._get_cache_key(text), self.config.cache_ttl, json.dumps(embedding)
            )
        except Exception as e:
            print(f"Warning: Cache storage failed: {str(e)}")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        # Try cache first
        cached = self._get_from_cache(text)
        if cached:
            return cached

        # Generate new embedding
        try:
            embedding = self.model.encode(text)
            embedding_list = self._convert_single_output(embedding)

            # Store in cache
            self._store_in_cache(text, embedding_list)

            return embedding_list
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {str(e)}")

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        # Initialize results list
        results: List[List[float]] = []
        texts_to_embed: List[str] = []
        cache_indices: Dict[int, str] = {}

        # Check cache for each text
        for i, text in enumerate(texts):
            cached = self._get_from_cache(text)
            if cached:
                results.append(cached)
            else:
                texts_to_embed.append(text)
                cache_indices[len(texts_to_embed) - 1] = text

        # Generate embeddings for texts not in cache
        if texts_to_embed:
            try:
                batch_embeddings = self.model.encode(
                    texts_to_embed,
                    batch_size=self.config.batch_size,
                    show_progress_bar=False,
                )

                # Process and cache new embeddings
                batch_results = self._convert_batch_output(batch_embeddings)
                for i, embedding_list in enumerate(batch_results):
                    original_text = cache_indices[i]
                    self._store_in_cache(original_text, embedding_list)
                    results.append(embedding_list)

            except Exception as e:
                raise RuntimeError(f"Failed to generate batch embeddings: {str(e)}")

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration"""
        return {
            "model_name": self.config.model_name,
            "max_seq_length": self.config.max_seq_length,
            "device": self.config.device,
            "cache_enabled": self.config.cache_enabled,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "batch_size": self.config.batch_size,
        }
