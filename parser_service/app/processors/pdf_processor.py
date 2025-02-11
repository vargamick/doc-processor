"""PDF document processor implementation"""

from typing import Dict, Any, Optional
import logging
from datetime import datetime
import fitz  # PyMuPDF

from .base import BaseDocumentProcessor

logger = logging.getLogger(__name__)


class PDFProcessor(BaseDocumentProcessor):
    """Processor for PDF documents"""

    def _extract_text(self, file_path: str) -> str:
        """Extract text content from PDF document"""
        try:
            with fitz.open(file_path) as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
                return text.strip()
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {file_path}: {str(e)}")
            raise

    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from PDF document"""
        try:
            with fitz.open(file_path) as doc:
                metadata = doc.metadata
                if metadata is None:
                    metadata = {}

                # Convert PDF-specific metadata to common format
                return {
                    "title": metadata.get("title", ""),
                    "author": metadata.get("author", ""),
                    "created_at": self._parse_pdf_date(
                        metadata.get("creationDate", "")
                    ),
                    "modified_at": self._parse_pdf_date(metadata.get("modDate", "")),
                    "page_count": doc.page_count,
                    "producer": metadata.get("producer", ""),
                    "creator": metadata.get("creator", ""),
                    "format": "PDF",
                    "file_path": file_path,
                }
        except Exception as e:
            logger.error(f"Failed to extract metadata from PDF {file_path}: {str(e)}")
            raise

    def _parse_pdf_date(self, date_str: str) -> Optional[datetime]:
        """Parse PDF date format to datetime"""
        if not date_str:
            return None

        try:
            # Handle D: prefix in PDF dates
            if date_str.startswith("D:"):
                date_str = date_str[2:]

            # Basic format: YYYYMMDDHHmmSS
            if len(date_str) >= 14:
                year = int(date_str[0:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                hour = int(date_str[8:10])
                minute = int(date_str[10:12])
                second = int(date_str[12:14])
                return datetime(year, month, day, hour, minute, second)
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse PDF date {date_str}: {str(e)}")

        return None
