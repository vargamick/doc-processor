import unittest
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List, cast
import torch
import numpy as np
from datetime import datetime

from ..models.document import (
    ProcessedDocument,
    DocumentMetadata,
    DocumentChunk,
    ChunkEmbedding,
)
from ..embeddings.base import EmbeddingConfig
from .base import BaseDocumentProcessor


class TestProcessor(BaseDocumentProcessor):
    """Test implementation of BaseDocumentProcessor"""

    def _extract_text(self, content: bytes) -> str:
        return content.decode("utf-8")

    def _extract_metadata(self, content: bytes) -> Dict[str, Any]:
        return {
            "title": "Test Document",
            "author": "Test Author",
            "created_date": datetime.now(),
            "modified_date": datetime.now(),
            "page_count": 1,
        }


class TestBaseDocumentProcessor(unittest.TestCase):
    def setUp(self):
        self.test_content = (
            b"This is a test document.\nIt has multiple lines.\nAnd some content."
        )
        self.embedding_config = EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",
            batch_size=2,
            cache_enabled=False,  # Disable cache for testing
            device="cpu",
        )
        self.mock_embedding = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        self.mock_batch_embeddings = torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32
        )

    def test_process_without_embeddings(self):
        """Test document processing without embeddings enabled"""
        processor = TestProcessor(chunk_size=50, chunk_overlap=10)
        result = processor.process(self.test_content, "test.txt")

        self.assertEqual(result.processing_status, "completed")
        self.assertEqual(result.embedding_status, "skipped")
        self.assertIsNone(result.embedding_error)
        self.assertTrue(len(result.chunks) > 0)
        self.assertIsNone(result.chunks[0].embedding)
        self.assertEqual(result.chunks[0].embedding_status, "pending")

    @patch("sentence_transformers.SentenceTransformer")
    def test_process_with_embeddings(self, mock_transformer):
        """Test document processing with embeddings enabled"""
        # Setup mock
        mock_model = MagicMock()
        mock_model.encode.return_value = self.mock_batch_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_transformer.return_value = mock_model

        processor = TestProcessor(
            chunk_size=50, chunk_overlap=10, embedding_config=self.embedding_config
        )
        result = processor.process(self.test_content, "test.txt")

        self.assertEqual(result.processing_status, "completed")
        self.assertEqual(result.embedding_status, "completed")
        self.assertIsNone(result.embedding_error)
        self.assertTrue(len(result.chunks) > 0)

        # Check first chunk's embedding
        first_chunk = result.chunks[0]
        self.assertIsNotNone(first_chunk.embedding)
        if first_chunk.embedding:  # Type guard for embedding
            self.assertEqual(first_chunk.embedding_status, "completed")
            self.assertEqual(len(first_chunk.embedding.vector), 3)
            self.assertEqual(first_chunk.embedding.dimension, 3)
            self.assertEqual(first_chunk.embedding.model, "all-MiniLM-L6-v2")

    @patch("sentence_transformers.SentenceTransformer")
    def test_embedding_error_handling(self, mock_transformer):
        """Test error handling during embedding generation"""
        # Setup mock to raise an error
        mock_model = MagicMock()
        mock_model.encode.side_effect = RuntimeError("Embedding generation failed")
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_transformer.return_value = mock_model

        processor = TestProcessor(
            chunk_size=50, chunk_overlap=10, embedding_config=self.embedding_config
        )
        result = processor.process(self.test_content, "test.txt")

        self.assertEqual(result.processing_status, "completed")
        self.assertEqual(result.embedding_status, "failed")
        self.assertIsNotNone(result.embedding_error)
        if result.embedding_error:  # Type guard for error message
            self.assertIn("Embedding generation failed", result.embedding_error)

    @patch("sentence_transformers.SentenceTransformer")
    def test_partial_embedding_failure(self, mock_transformer):
        """Test handling of partial failures in batch embedding generation"""
        # Setup mock for partial success
        mock_model = MagicMock()
        mock_model.encode.return_value = self.mock_batch_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_transformer.return_value = mock_model

        # Create a processor with small chunk size to generate multiple chunks
        processor = TestProcessor(
            chunk_size=20, chunk_overlap=5, embedding_config=self.embedding_config
        )

        # Process document
        result = processor.process(self.test_content, "test.txt")

        self.assertEqual(result.processing_status, "completed")
        self.assertIn(result.embedding_status, ["completed", "partially_completed"])

        # Verify chunk embeddings
        completed_chunks = sum(
            1 for chunk in result.chunks if chunk.embedding_status == "completed"
        )
        self.assertTrue(completed_chunks > 0)

    @patch("sentence_transformers.SentenceTransformer")
    def test_metadata_with_embeddings(self, mock_transformer):
        """Test that embedding metadata is properly included"""
        # Setup mock
        mock_model = MagicMock()
        mock_model.encode.return_value = self.mock_batch_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_transformer.return_value = mock_model

        processor = TestProcessor(
            chunk_size=50, chunk_overlap=10, embedding_config=self.embedding_config
        )
        result = processor.process(self.test_content, "test.txt")

        # Check metadata
        self.assertEqual(result.metadata.embedding_model, "all-MiniLM-L6-v2")
        self.assertEqual(result.metadata.embedding_dimension, 3)

    def test_document_processing_error(self):
        """Test error handling during document processing"""

        # Create a processor that will fail during text extraction
        class FailingProcessor(TestProcessor):
            def _extract_text(self, content: bytes) -> str:
                raise RuntimeError("Text extraction failed")

        processor = FailingProcessor(
            chunk_size=50, chunk_overlap=10, embedding_config=self.embedding_config
        )
        result = processor.process(self.test_content, "test.txt")

        self.assertEqual(result.processing_status, "failed")
        self.assertEqual(result.embedding_status, "failed")
        if result.error:  # Type guard for error message
            self.assertIn("Text extraction failed", result.error)
        self.assertEqual(len(result.chunks), 0)


if __name__ == "__main__":
    unittest.main()
