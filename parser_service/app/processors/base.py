"""Base document processor implementation"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime

from ..models.document import Document
from ..embeddings import EmbeddingModelFactory, ModelConfig

logger = logging.getLogger(__name__)


class BaseDocumentProcessor(ABC):
    """Base class for document processors"""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_config: Optional[ModelConfig] = None,
    ):
        """Initialize processor with configuration"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._embedding_model = EmbeddingModelFactory.create_model(
            config=embedding_config
        )

    @abstractmethod
    def _extract_text(self, file_path: str) -> str:
        """Extract raw text from document"""
        pass

    @abstractmethod
    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from document"""
        pass

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        if not text:
            return []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            # Try to end at a sentence or paragraph boundary
            if end < text_len:
                for boundary in [".\n", ".\r\n", ". ", "\n\n", "\r\n\r\n"]:
                    boundary_pos = text.rfind(boundary, start, end)
                    if boundary_pos > start:
                        end = boundary_pos + len(boundary)
                        break

            chunks.append(text[start:end].strip())
            start = end - self.chunk_overlap

        return [chunk for chunk in chunks if chunk]  # Remove empty chunks

    def _generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks"""
        if not chunks:
            return []

        try:
            return self._embedding_model.encode_batch(chunks)
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise

    def process(self, file_path: str) -> Tuple[Document, List[List[float]]]:
        """Process document and generate embeddings

        Returns:
            Tuple containing:
            - Document object with extracted text and metadata
            - List of embeddings for text chunks
        """
        try:
            # Extract content
            text = self._extract_text(file_path)
            metadata = self._extract_metadata(file_path)

            # Create chunks
            chunks = self._chunk_text(text)
            if not chunks:
                raise ValueError("No text content extracted from document")

            # Generate embeddings
            embeddings = self._generate_embeddings(chunks)

            # Create document
            doc = Document(
                text=text,
                chunks=chunks,
                metadata=metadata,
                processed_at=datetime.utcnow(),
            )

            return doc, embeddings

        except Exception as e:
            logger.error(f"Failed to process document {file_path}: {str(e)}")
            raise
