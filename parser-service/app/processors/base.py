from abc import ABC, abstractmethod
from typing import BinaryIO, Dict, Any, List, Optional, Tuple
from ..models.document import (
    ProcessedDocument,
    DocumentMetadata,
    DocumentChunk,
    ChunkEmbedding,
    EmbeddingStats,
)
from ..embeddings.base import EmbeddingProcessor, EmbeddingConfig
from ..utils.text_processing import (
    clean_text,
    chunk_text,
    count_words,
    detect_language,
    extract_keywords,
    measure_processing_time,
)
import os
from datetime import datetime
import time


class BaseDocumentProcessor(ABC):
    """Base class for document processors"""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        embedding_config: Optional[EmbeddingConfig] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_processor = (
            EmbeddingProcessor(embedding_config) if embedding_config else None
        )

    @abstractmethod
    def _extract_text(self, content: bytes) -> str:
        """Extract text from document content"""
        pass

    @abstractmethod
    def _extract_metadata(self, content: bytes) -> Dict[str, Any]:
        """Extract metadata from document content"""
        pass

    def _generate_embeddings(
        self, chunks: List[DocumentChunk]
    ) -> Tuple[List[DocumentChunk], EmbeddingStats]:
        """Generate embeddings for document chunks"""
        if not self.embedding_processor:
            return chunks, EmbeddingStats(
                total_chunks=len(chunks),
                processed_chunks=0,
                cached_chunks=0,
                failed_chunks=0,
                average_time_per_chunk=0.0,
                total_processing_time=0.0,
                cache_hit_rate=0.0,
                model_name="none",
                dimension=0,
            )

        start_time = time.time()
        processed_chunks = 0
        cached_chunks = 0
        failed_chunks = 0

        # Get model info for metadata
        model_info = self.embedding_processor.get_model_info()

        # Process chunks in batches
        texts = [chunk.content for chunk in chunks]
        try:
            embeddings = self.embedding_processor.generate_embeddings_batch(texts)

            # Update chunks with embeddings
            for chunk, embedding_vector in zip(chunks, embeddings):
                try:
                    chunk.embedding = ChunkEmbedding(
                        vector=embedding_vector,
                        model=model_info["model_name"],
                        dimension=model_info["embedding_dimension"],
                    )
                    chunk.embedding_status = "completed"
                    processed_chunks += 1
                except Exception as e:
                    chunk.embedding_status = "failed"
                    chunk.embedding_error = str(e)
                    failed_chunks += 1

        except Exception as e:
            # Handle batch processing failure
            for chunk in chunks:
                chunk.embedding_status = "failed"
                chunk.embedding_error = str(e)
            failed_chunks = len(chunks)

        total_time = time.time() - start_time
        avg_time = total_time / len(chunks) if chunks else 0

        # Calculate statistics
        stats = EmbeddingStats(
            total_chunks=len(chunks),
            processed_chunks=processed_chunks,
            cached_chunks=cached_chunks,
            failed_chunks=failed_chunks,
            average_time_per_chunk=avg_time,
            total_processing_time=total_time,
            cache_hit_rate=cached_chunks / len(chunks) if chunks else 0,
            model_name=model_info["model_name"],
            dimension=model_info["embedding_dimension"],
        )

        return chunks, stats

    @measure_processing_time
    def process(self, content: bytes, filename: str) -> ProcessedDocument:
        """Process a document and return structured results"""
        try:
            # Extract text and metadata
            raw_text = self._extract_text(content)
            doc_metadata = self._extract_metadata(content)

            # Clean and process text
            cleaned_text = clean_text(raw_text)

            # Create chunks
            text_chunks = chunk_text(cleaned_text, self.chunk_size, self.chunk_overlap)
            chunks = [
                DocumentChunk(
                    content=chunk_text,
                    chunk_number=chunk_num,
                    metadata={"position": f"chunk_{chunk_num}"},
                )
                for chunk_text, chunk_num in text_chunks
            ]

            # Generate embeddings if processor is configured
            embedding_error = None
            if self.embedding_processor:
                try:
                    chunks, stats = self._generate_embeddings(chunks)
                    embedding_status = (
                        "completed"
                        if stats.failed_chunks == 0
                        else (
                            "partially_completed"
                            if stats.processed_chunks > 0
                            else "failed"
                        )
                    )
                except Exception as e:
                    embedding_status = "failed"
                    embedding_error = str(e)
            else:
                embedding_status = "skipped"

            # Extract additional metadata
            word_count = count_words(cleaned_text)
            language = detect_language(cleaned_text)
            keywords = extract_keywords(cleaned_text)

            # Create document metadata
            metadata = DocumentMetadata(
                title=doc_metadata.get("title", filename),
                author=doc_metadata.get("author"),
                created_date=doc_metadata.get("created_date"),
                modified_date=doc_metadata.get("modified_date"),
                page_count=doc_metadata.get("page_count", 1),
                word_count=word_count,
                language=language,
                keywords=keywords,
                file_type=filename.split(".")[-1].lower(),
                file_size=len(content),
                embedding_model=(
                    self.embedding_processor.get_model_info()["model_name"]
                    if self.embedding_processor
                    else None
                ),
                embedding_dimension=(
                    self.embedding_processor.get_model_info()["embedding_dimension"]
                    if self.embedding_processor
                    else None
                ),
            )

            # Save processed text
            os.makedirs("data", exist_ok=True)
            output_path = os.path.join("data", f"{filename}.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            return ProcessedDocument(
                filename=filename,
                metadata=metadata,
                chunks=chunks,
                raw_text=cleaned_text,
                output_path=output_path,
                processing_status="completed",
                embedding_status=embedding_status,
                embedding_error=embedding_error,
            )

        except Exception as e:
            return ProcessedDocument(
                filename=filename,
                metadata=DocumentMetadata(
                    file_type=filename.split(".")[-1].lower(),
                    file_size=len(content),
                    processing_time=0.0,
                ),
                chunks=[],
                raw_text="",
                output_path="",
                error=str(e),
                processing_status="failed",
                embedding_status="failed",
                embedding_error=str(e),
            )
