from typing import Optional
from .base import BaseDocumentProcessor
from .pdf_processor import PDFProcessor
from .docx_processor import DOCXProcessor


class ProcessorFactory:
    """Factory for creating document processors based on file type"""

    _processors = {
        "pdf": PDFProcessor,
        "docx": DOCXProcessor,
    }

    @classmethod
    def get_processor(
        cls, file_type: str, chunk_size: int = 1000, chunk_overlap: int = 100
    ) -> Optional[BaseDocumentProcessor]:
        """Get appropriate processor for file type"""
        processor_class = cls._processors.get(file_type.lower())
        if processor_class:
            return processor_class(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return None

    @classmethod
    def supported_types(cls) -> list[str]:
        """Get list of supported file types"""
        return list(cls._processors.keys())
