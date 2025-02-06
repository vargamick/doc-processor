from typing import (
    List,
    Optional,
    Dict,
    Any,
    Union,
    Sequence,
    cast,
    TypeVar,
    overload,
    Type,
    Callable,
)
import os
import torch
from sentence_transformers import SentenceTransformer
import redis
import json
import hashlib
import numpy as np
import logging
from numpy.typing import NDArray
from pydantic import BaseModel
from .cache import EmbeddingCache, CacheConfig
from .batch import DynamicBatchProcessor, DynamicBatchConfig, BatchPriority

# Configure logging
logger = logging.getLogger(__name__)

# Callback type for progress updates
ProgressCallback = Callable[[float, int, int], None]

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

    # Dynamic batch processing settings
    dynamic_batching: bool = True
    batch_config: Optional[DynamicBatchConfig] = None


class EmbeddingProcessor:
    """Base class for generating and managing embeddings"""

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.model = self._initialize_model()
        self.cache = self._initialize_cache() if self.config.cache_enabled else None
        self.batch_processor = (
            DynamicBatchProcessor(self.config.batch_config)
            if self.config.dynamic_batching
            else None
        )

    def _initialize_model(self) -> SentenceTransformer:
        """Initialize the sentence transformer model"""
        try:
            model = SentenceTransformer(self.config.model_name)
            model.to(self.config.device)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embedding model: {str(e)}")

    def _initialize_cache(self) -> Optional["EmbeddingCache"]:
        """Initialize Redis cache"""
        try:
            config = CacheConfig(
                host=os.getenv("REDIS_HOST", "redis"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                db=int(os.getenv("REDIS_DB", "0")),
                pool_size=int(os.getenv("REDIS_POOL_SIZE", "10")),
                socket_timeout=int(os.getenv("REDIS_TIMEOUT", "5")),
                compression_threshold=int(
                    os.getenv("REDIS_COMPRESSION_THRESHOLD", "1024")
                ),
                max_item_size=int(os.getenv("REDIS_MAX_ITEM_SIZE", "1048576")),  # 1MB
                default_ttl=int(os.getenv("REDIS_TTL", "3600")),  # 1 hour
            )
            return EmbeddingCache(config)
        except Exception as e:
            logger.warning(f"Failed to initialize Redis cache: {str(e)}")
            return None

    @overload
    def _tensor_to_list(self, tensor: FloatTensor) -> List[float]: ...

    @overload
    def _tensor_to_list(self, tensor: List[FloatTensor]) -> List[List[float]]: ...

    def _tensor_to_list(
        self, tensor: Union[FloatTensor, List[FloatTensor]]
    ) -> Union[List[float], List[List[float]]]:
        """Convert PyTorch tensor or list of tensors to list format"""
        if isinstance(tensor, list):
            result = [t.detach().cpu().numpy().tolist() for t in tensor]
            return cast(List[List[float]], result)
        result = tensor.detach().cpu().numpy().tolist()
        return cast(List[float], result)

    def _array_to_list(self, array: FloatArray) -> List[float]:
        """Convert numpy array to list of floats"""
        result = array.tolist()
        return cast(List[float], result)

    def _ensure_device_compatibility(self, tensor: FloatTensor) -> FloatTensor:
        """Ensure tensor is on the correct device"""
        if tensor.device.type != self.config.device:
            return tensor.to(self.config.device)
        return tensor

    def _convert_single_output(self, output: ModelOutput) -> List[float]:
        """Convert single model output to list format"""
        try:
            if isinstance(output, torch.Tensor):
                result = self._tensor_to_list(self._ensure_device_compatibility(output))
                return cast(List[float], result)
            if isinstance(output, np.ndarray):
                return self._array_to_list(output)
            if isinstance(output, list) and all(
                isinstance(t, torch.Tensor) for t in output
            ):
                # Handle case where model returns list of tensors for single input
                tensors = cast(List[FloatTensor], output)
                # Concatenate tensors if multiple are returned
                combined = torch.cat(tensors, dim=0)
                result = self._tensor_to_list(combined)
                return cast(List[float], result)
            raise TypeError(f"Unsupported output type: {type(output)}")
        except Exception as e:
            raise RuntimeError(f"Failed to convert model output: {str(e)}")

    def _convert_batch_output(self, output: BatchOutput) -> List[List[float]]:
        """Convert batch model output to list format"""
        try:
            if isinstance(output, torch.Tensor):
                result = self._tensor_to_list(self._ensure_device_compatibility(output))
                return cast(List[List[float]], result)
            if isinstance(output, np.ndarray):
                result = self._array_to_list(output)
                return cast(List[List[float]], [result])
            if isinstance(output, list) and all(
                isinstance(t, torch.Tensor) for t in output
            ):
                # Handle list of tensors
                result = [self._tensor_to_list(t) for t in output]
                return cast(List[List[float]], result)
            raise TypeError(f"Unsupported batch output type: {type(output)}")
        except Exception as e:
            raise RuntimeError(f"Failed to convert batch output: {str(e)}")

    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for the given text"""
        return hashlib.md5(text.encode()).hexdigest()

    def _get_from_cache(self, text: str) -> Optional[List[float]]:
        """Retrieve embeddings from cache if available"""
        if not self.cache:
            return None

        try:
            result = self.cache.get(self._get_cache_key(text))
            if isinstance(result, list) and all(
                isinstance(x, (int, float)) for x in result
            ):
                return cast(List[float], result)
            return None
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {str(e)}")
            return None

    def _store_in_cache(self, text: str, embedding: List[float]) -> None:
        """Store embeddings in cache"""
        if not self.cache:
            return

        try:
            if isinstance(embedding, list) and all(
                isinstance(x, (int, float)) for x in embedding
            ):
                self.cache.set(self._get_cache_key(text), embedding)
        except Exception as e:
            logger.warning(f"Cache storage failed: {str(e)}")

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

    def generate_embeddings_batch(
        self,
        texts: List[str],
        priority: BatchPriority = BatchPriority.MEDIUM,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[List[float]]:
        """Generate embeddings for a batch of texts with priority and progress tracking"""
        if not texts:
            return []

        if not self.config.dynamic_batching or not self.batch_processor:
            return self._generate_embeddings_simple_batch(texts)

        # Add texts to batch processor
        for text in texts:
            self.batch_processor.add_item(text, priority)

        def process_batch(batch_texts: List[str]) -> List[List[float]]:
            # Check cache for each text
            results: List[List[float]] = []
            texts_to_embed: List[str] = []
            cache_indices: Dict[int, str] = {}

            for i, text in enumerate(batch_texts):
                cached = self._get_from_cache(text)
                if cached:
                    results.append(cached)
                else:
                    texts_to_embed.append(text)
                    cache_indices[len(texts_to_embed) - 1] = text

            if texts_to_embed:
                try:
                    batch_embeddings = self.model.encode(
                        texts_to_embed,
                        batch_size=self.config.batch_size,
                        show_progress_bar=False,
                    )

                    batch_results = self._convert_batch_output(batch_embeddings)
                    for i, embedding_list in enumerate(batch_results):
                        original_text = cache_indices[i]
                        self._store_in_cache(original_text, embedding_list)
                        results.append(embedding_list)

                except Exception as e:
                    raise RuntimeError(f"Failed to generate batch embeddings: {str(e)}")

            return results

        # Process queue with progress tracking
        total_texts = len(texts)
        if progress_callback:

            def wrapped_processor(batch: List[str]) -> List[List[float]]:
                result = process_batch(batch)
                processed = len(result)
                progress_callback(
                    (processed / total_texts) * 100, processed, total_texts
                )
                return result

            return self.batch_processor.process_queue(wrapped_processor)

        return self.batch_processor.process_queue(process_batch)

    def _generate_embeddings_simple_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using simple batching without priority or progress tracking"""
        results: List[List[float]] = []
        texts_to_embed: List[str] = []
        cache_indices: Dict[int, str] = {}

        for i, text in enumerate(texts):
            cached = self._get_from_cache(text)
            if cached:
                results.append(cached)
            else:
                texts_to_embed.append(text)
                cache_indices[len(texts_to_embed) - 1] = text

        if texts_to_embed:
            try:
                batch_embeddings = self.model.encode(
                    texts_to_embed,
                    batch_size=self.config.batch_size,
                    show_progress_bar=False,
                )

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
        info = {
            "model_name": self.config.model_name,
            "max_seq_length": self.config.max_seq_length,
            "device": self.config.device,
            "cache_enabled": self.config.cache_enabled,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "batch_size": self.config.batch_size,
            "dynamic_batching": self.config.dynamic_batching,
        }

        if self.batch_processor:
            info.update(self.batch_processor.get_stats())

        return info
