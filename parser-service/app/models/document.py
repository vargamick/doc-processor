from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime


class DocumentMetadata(BaseModel):
    """Metadata extracted from documents"""

    title: Optional[str] = None
    author: Optional[str] = None
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    language: Optional[str] = None
    keywords: List[str] = []
    file_type: str
    file_size: int
    processing_time: float = 0.0
    embedding_model: Optional[str] = None
    embedding_dimension: Optional[int] = None


class ChunkEmbedding(BaseModel):
    """Represents an embedding vector for a document chunk"""

    vector: List[float]
    model: str
    dimension: int
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    cache_key: Optional[str] = None


class DocumentChunk(BaseModel):
    """Represents a chunk of processed document text"""

    content: str
    page_number: Optional[int] = None
    chunk_number: int
    metadata: Dict[str, str] = {}
    embedding: Optional[ChunkEmbedding] = None
    embedding_status: str = "pending"  # pending, processing, completed, failed
    embedding_error: Optional[str] = None


class ProcessedDocument(BaseModel):
    """Represents a fully processed document"""

    filename: str
    metadata: DocumentMetadata
    chunks: List[DocumentChunk]
    raw_text: str
    output_path: str
    error: Optional[str] = None
    processing_status: str = "completed"  # pending, processing, completed, failed
    embedding_status: str = "pending"  # pending, processing, completed, failed
    embedding_error: Optional[str] = None


class EmbeddingStats(BaseModel):
    """Statistics about embedding generation"""

    total_chunks: int
    processed_chunks: int
    cached_chunks: int
    failed_chunks: int
    average_time_per_chunk: float
    total_processing_time: float
    cache_hit_rate: float
    model_name: str
    dimension: int
