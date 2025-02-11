from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.covariance import EllipticEnvelope
import logging
import torch
from sentence_transformers import SentenceTransformer
from .base import EmbeddingProcessor

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for embedding validation"""

    similarity_threshold: float = (
        0.7  # Minimum cosine similarity for semantic validation
    )
    dimension_tolerance: float = 1e-6  # Tolerance for dimension validation
    outlier_threshold: float = 0.1  # Fraction of points to consider as outliers
    min_samples: int = 100  # Minimum samples needed for statistical validation
    reference_texts: List[str] = field(
        default_factory=list
    )  # Reference texts for semantic validation
    periodic_validation_interval: int = 1000  # Number of embeddings between validations


class EmbeddingValidator:
    """Validates embedding quality and consistency"""

    def __init__(
        self, processor: EmbeddingProcessor, config: Optional[ValidationConfig] = None
    ):
        self.processor = processor
        self.config = config or ValidationConfig()
        self.expected_dim = self.processor.model.get_sentence_embedding_dimension()
        self.validation_count = 0
        self._initialize_reference_embeddings()

    def _initialize_reference_embeddings(self) -> None:
        """Initialize reference embeddings for semantic validation"""
        if not self.config.reference_texts:
            # Default reference texts for general semantic validation
            self.config.reference_texts = [
                "The quick brown fox jumps over the lazy dog",
                "A simple test sentence to validate embeddings",
                "This is a reference text for semantic similarity",
            ]

        self.reference_embeddings = np.array(
            [
                self.processor.generate_embedding(text)
                for text in self.config.reference_texts
            ]
        )

    def validate_dimensions(
        self, embeddings: List[List[float]]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate embedding dimensions"""
        if not embeddings:
            return False, {"error": "No embeddings provided"}

        try:
            dims = [len(emb) for emb in embeddings]
            consistent = all(d == self.expected_dim for d in dims)
            stats = {
                "expected_dimension": self.expected_dim,
                "actual_dimensions": dims,
                "consistent": consistent,
                "dimension_errors": [
                    i for i, d in enumerate(dims) if d != self.expected_dim
                ],
            }
            return consistent, stats
        except Exception as e:
            logger.error(f"Dimension validation error: {str(e)}")
            return False, {"error": str(e)}

    def validate_semantic_similarity(
        self, embeddings: List[List[float]]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate semantic coherence using cosine similarity"""
        if not embeddings:
            return False, {"error": "No embeddings provided"}

        try:
            embeddings_array = np.array(embeddings)
            ref_similarities = cosine_similarity(
                embeddings_array, self.reference_embeddings
            )

            # Check if embeddings maintain expected semantic relationships
            max_similarities = np.max(ref_similarities, axis=1)
            valid = np.mean(max_similarities) >= self.config.similarity_threshold

            stats = {
                "mean_similarity": float(np.mean(max_similarities)),
                "min_similarity": float(np.min(max_similarities)),
                "max_similarity": float(np.max(max_similarities)),
                "threshold": self.config.similarity_threshold,
                "valid": valid,
            }
            return valid, stats
        except Exception as e:
            logger.error(f"Semantic validation error: {str(e)}")
            return False, {"error": str(e)}

    def detect_outliers(
        self, embeddings: List[List[float]]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Detect outlier embeddings using robust covariance estimation"""
        if len(embeddings) < self.config.min_samples:
            return True, {"warning": "Not enough samples for outlier detection"}

        try:
            embeddings_array = np.array(embeddings)
            outlier_detector = EllipticEnvelope(
                contamination=self.config.outlier_threshold, random_state=42
            )

            # Fit and predict outliers
            labels = outlier_detector.fit_predict(embeddings_array)
            outlier_indices = np.where(labels == -1)[0]

            # Calculate outlier statistics
            outlier_ratio = len(outlier_indices) / len(embeddings)
            valid = outlier_ratio <= self.config.outlier_threshold

            stats = {
                "outlier_indices": outlier_indices.tolist(),
                "outlier_ratio": float(outlier_ratio),
                "threshold": self.config.outlier_threshold,
                "valid": valid,
            }
            return valid, stats
        except Exception as e:
            logger.error(f"Outlier detection error: {str(e)}")
            return False, {"error": str(e)}

    def validate_batch(
        self, embeddings: List[List[float]]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Perform comprehensive validation on a batch of embeddings"""
        self.validation_count += len(embeddings)

        # Collect all validation results
        dim_valid, dim_stats = self.validate_dimensions(embeddings)
        sem_valid, sem_stats = self.validate_semantic_similarity(embeddings)
        out_valid, out_stats = self.detect_outliers(embeddings)

        # Combine validation results
        valid = all([dim_valid, sem_valid, out_valid])

        results = {
            "valid": valid,
            "validation_count": self.validation_count,
            "dimension_validation": dim_stats,
            "semantic_validation": sem_stats,
            "outlier_detection": out_stats,
        }

        if not valid:
            logger.warning(f"Validation failed: {results}")

        return valid, results

    def should_validate(self) -> bool:
        """Check if periodic validation should be performed"""
        return (
            self.validation_count % self.config.periodic_validation_interval == 0
            or self.validation_count == 0
        )

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get current validation statistics"""
        return {
            "validation_count": self.validation_count,
            "expected_dimension": self.expected_dim,
            "similarity_threshold": self.config.similarity_threshold,
            "outlier_threshold": self.config.outlier_threshold,
            "periodic_validation_interval": self.config.periodic_validation_interval,
        }
