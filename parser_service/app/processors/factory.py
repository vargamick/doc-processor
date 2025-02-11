"""Factory for creating document processors"""

from typing import Dict, Type, Optional, Set
import logging

from .base import BaseDocumentProcessor
from .pdf_processor import PDFProcessor
from .docx_processor import DOCXProcessor
from ..embeddings import ModelConfig

logger = logging.getLogger(__name__)


class ProcessorFactory:
    """Factory for creating document processors"""

    _processor_registry: Dict[str, Type[BaseDocumentProcessor]] = {
        "pdf": PDFProcessor,
        "docx": DOCXProcessor,
    }

    @classmethod
    def register_processor(
        cls, file_type: str, processor_class: Type[BaseDocumentProcessor]
    ) -> None:
        """Register a new processor type"""
        if file_type in cls._processor_registry:
            logger.warning(f"Overwriting existing processor for {file_type}")
        cls._processor_registry[file_type] = processor_class

    @classmethod
    def get_processor(
        cls,
        file_type: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        embedding_config: Optional[ModelConfig] = None,
    ) -> Optional[BaseDocumentProcessor]:
        """Get processor instance for file type"""
        processor_class = cls._processor_registry.get(file_type.lower())
        if not processor_class:
            logger.warning(f"No processor registered for file type: {file_type}")
            return None

        try:
            return processor_class(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embedding_config=embedding_config,
            )
        except Exception as e:
            logger.error(f"Failed to create processor for {file_type}: {str(e)}")
            return None

    @classmethod
    def supported_types(cls) -> Set[str]:
        """Get set of supported file types"""
        return set(cls._processor_registry.keys())
