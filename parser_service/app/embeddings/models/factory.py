from typing import Dict, Type, Optional
import logging

from .base import IEmbeddingModel
from .sentence_transformer import SentenceTransformerModel
from ..config.model_config import ModelConfig

logger = logging.getLogger(__name__)


class EmbeddingModelFactory:
    """Factory for creating embedding model instances"""

    _model_registry: Dict[str, Type[IEmbeddingModel]] = {
        "sentence-transformer": SentenceTransformerModel,
    }

    @classmethod
    def register_model(cls, name: str, model_class: Type[IEmbeddingModel]) -> None:
        """Register a new model type"""
        if name in cls._model_registry:
            logger.warning(f"Overwriting existing model type: {name}")
        cls._model_registry[name] = model_class

    @classmethod
    def create_model(
        cls,
        model_type: str = "sentence-transformer",
        config: Optional[ModelConfig] = None,
    ) -> IEmbeddingModel:
        """Create a new embedding model instance"""
        if model_type not in cls._model_registry:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available types: {list(cls._model_registry.keys())}"
            )

        model_class = cls._model_registry[model_type]
        model_config = config or ModelConfig()

        try:
            return model_class(model_config)
        except Exception as e:
            raise RuntimeError(f"Failed to create model {model_type}: {str(e)}")

    @classmethod
    def available_models(cls) -> Dict[str, Type[IEmbeddingModel]]:
        """Get dictionary of available model types"""
        return cls._model_registry.copy()
