from typing import List, Optional, Dict, Any, Union, Sequence, cast
import torch
from sentence_transformers import SentenceTransformer
import logging
import numpy as np
from numpy.typing import NDArray

from .base import (
    IEmbeddingModel,
    ModelOutput,
    SingleEmbedding,
    BatchEmbeddings,
    FloatArray,
)
from ..config.model_config import ModelConfig

logger = logging.getLogger(__name__)


def ensure_float_list(data: Union[List[Any], NDArray[Any]]) -> List[float]:
    """Convert numpy array or list to list of floats"""
    if isinstance(data, np.ndarray):
        return [float(x) for x in data.flatten()]
    return [float(x) for x in data]


def ensure_float_list_batch(
    data: Union[List[List[Any]], NDArray[Any]]
) -> List[List[float]]:
    """Convert numpy array or nested list to list of float lists"""
    if isinstance(data, np.ndarray):
        return [[float(x) for x in row] for row in data]
    return [[float(x) for x in row] for row in data]


class SentenceTransformerModel(IEmbeddingModel):
    """Implementation of IEmbeddingModel using sentence-transformers"""

    def _initialize_model(self) -> SentenceTransformer:
        """Initialize the sentence transformer model"""
        try:
            model = SentenceTransformer(
                self.config.model_name,
                cache_folder=self.config.cache_dir,
                **(self.config.model_kwargs or {}),
            )
            model.max_seq_length = self.config.max_seq_length
            if self.config.device != "cpu":
                model.to(self.config.device)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embedding model: {str(e)}")

    def _convert_output(
        self, output: ModelOutput, is_batch: bool = False
    ) -> Union[SingleEmbedding, BatchEmbeddings]:
        """Convert model output to appropriate embedding format"""
        try:
            if isinstance(output, torch.Tensor):
                result = output.detach().cpu().numpy()
                if is_batch:
                    if len(result.shape) != 2:
                        raise TypeError("Expected batch output to be 2D tensor")
                    return ensure_float_list_batch(result)
                else:
                    if len(result.shape) != 1:
                        raise TypeError("Expected single output to be 1D tensor")
                    return ensure_float_list(result)

            if isinstance(output, list) and all(
                isinstance(t, torch.Tensor) for t in output
            ):
                # Handle case where model returns list of tensors
                combined = torch.cat(output, dim=0)
                result = combined.detach().cpu().numpy()
                if is_batch:
                    if len(result.shape) != 2:
                        result = np.expand_dims(result, axis=1)
                    return ensure_float_list_batch(result)
                else:
                    if len(result.shape) != 1:
                        result = result.squeeze()
                    return ensure_float_list(result)

            raise TypeError(f"Unsupported output type: {type(output)}")
        except Exception as e:
            raise RuntimeError(f"Failed to convert model output: {str(e)}")

    def encode(self, text: str) -> SingleEmbedding:
        """Generate embedding for a single text"""
        try:
            embedding = self._model.encode(
                text,
                normalize_embeddings=self.config.normalize_embeddings,
                convert_to_tensor=True,
            )
            result = self._convert_output(embedding, is_batch=False)
            if not isinstance(result, list):
                raise TypeError("Expected single embedding result")
            return cast(SingleEmbedding, result)
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {str(e)}")

    def encode_batch(
        self, texts: Sequence[str], batch_size: Optional[int] = None
    ) -> BatchEmbeddings:
        """Generate embeddings for a batch of texts"""
        if not texts:
            return []

        try:
            batch_size = batch_size or self.config.batch_size
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=self.config.normalize_embeddings,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            result = self._convert_output(embeddings, is_batch=True)
            if not isinstance(result, list) or not all(
                isinstance(x, list) for x in result
            ):
                raise TypeError("Expected batch embedding result")
            return cast(BatchEmbeddings, result)
        except Exception as e:
            raise RuntimeError(f"Failed to generate batch embeddings: {str(e)}")

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings generated by this model"""
        model = cast(SentenceTransformer, self._model)
        dim = model.get_sentence_embedding_dimension()
        if dim is None:
            raise RuntimeError("Failed to get embedding dimension")
        return dim

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model configuration"""
        return {
            "model_name": self.config.model_name,
            "max_seq_length": self.config.max_seq_length,
            "device": self.config.device,
            "embedding_dimension": self.get_embedding_dimension(),
            "batch_size": self.config.batch_size,
            "normalize_embeddings": self.config.normalize_embeddings,
            "pooling_strategy": self.config.pooling_strategy,
        }

    def to_device(self, device: str) -> None:
        """Move model to specified device"""
        if device != self.config.device:
            self._model.to(device)
            self.config.device = device
