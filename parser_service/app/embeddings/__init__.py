"""Document embedding system

This package provides functionality for generating and managing document embeddings.
It includes model implementations, configuration, and factory classes for creating
embedding models.

Example usage:
    from parser_service.app.embeddings import EmbeddingModelFactory, ModelConfig

    # Create a model with default configuration
    model = EmbeddingModelFactory.create_model()

    # Generate embeddings
    text = "Example document text"
    embedding = model.encode(text)

    # Generate batch embeddings
    texts = ["First document", "Second document"]
    embeddings = model.encode_batch(texts)
"""

from .models import (
    IEmbeddingModel,
    SingleEmbedding,
    BatchEmbeddings,
    EmbeddingModelFactory,
    SentenceTransformerModel,
)
from .config.model_config import ModelConfig

__all__ = [
    "IEmbeddingModel",
    "SingleEmbedding",
    "BatchEmbeddings",
    "EmbeddingModelFactory",
    "SentenceTransformerModel",
    "ModelConfig",
]
