from typing import Dict, Any
from pdf2image import convert_from_bytes
import pytesseract
import io
from PyPDF2 import PdfReader
from datetime import datetime
from .base import BaseDocumentProcessor


class PDFProcessor(BaseDocumentProcessor):
    """Processor for PDF documents"""

    def _extract_text(self, content: bytes) -> str:
        """Extract text from PDF using OCR and direct extraction"""
        extracted_text = ""

        # First try direct text extraction
        pdf_reader = PdfReader(io.BytesIO(content))
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text.strip():
                extracted_text += page_text + "\n"

        # If no text was extracted, use OCR
        if not extracted_text.strip():
            # Convert PDF to images
            images = convert_from_bytes(content)

            # Extract text from each image using OCR
            for image in images:
                page_text = pytesseract.image_to_string(image)
                extracted_text += page_text + "\n"

        return extracted_text

    def _extract_metadata(self, content: bytes) -> Dict[str, Any]:
        """Extract metadata from PDF document"""
        metadata = {}

        try:
            pdf_reader = PdfReader(io.BytesIO(content))
            pdf_info = pdf_reader.metadata

            if pdf_info:
                # Extract title
                metadata["title"] = pdf_info.get("/Title", "")

                # Extract author
                metadata["author"] = pdf_info.get("/Author", "")

                # Extract dates
                creation_date = pdf_info.get("/CreationDate", "")
                if creation_date and creation_date.startswith("D:"):
                    # Parse PDF date format (D:YYYYMMDDHHmmSS)
                    date_str = creation_date[2:14]  # Extract YYYYMMDDHHMM
                    try:
                        metadata["created_date"] = datetime.strptime(
                            date_str, "%Y%m%d%H%M"
                        )
                    except ValueError:
                        pass

                mod_date = pdf_info.get("/ModDate", "")
                if mod_date and mod_date.startswith("D:"):
                    date_str = mod_date[2:14]
                    try:
                        metadata["modified_date"] = datetime.strptime(
                            date_str, "%Y%m%d%H%M"
                        )
                    except ValueError:
                        pass

            # Get page count
            metadata["page_count"] = len(pdf_reader.pages)

        except Exception as e:
            # Log error but continue processing
            print(f"Error extracting PDF metadata: {str(e)}")

        return metadata
