"""Document model for storing processed document information"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class Document:
    """Represents a processed document with text content and metadata"""

    text: str
    chunks: List[str]
    metadata: Dict[str, Any]
    processed_at: datetime = field(default_factory=datetime.utcnow)
    id: Optional[str] = None
    file_path: Optional[str] = None
    file_type: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    language: Optional[str] = None
    status: str = "pending"

    def __post_init__(self):
        """Extract common metadata fields if present"""
        if self.metadata:
            self.title = self.metadata.get("title", self.title)
            self.author = self.metadata.get("author", self.author)
            self.created_at = self.metadata.get("created_at", self.created_at)
            self.modified_at = self.metadata.get("modified_at", self.modified_at)
            self.page_count = self.metadata.get("page_count", self.page_count)
            self.word_count = self.metadata.get("word_count", self.word_count)
            self.language = self.metadata.get("language", self.language)

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary representation"""
        return {
            "id": self.id,
            "text": self.text,
            "chunks": self.chunks,
            "metadata": self.metadata,
            "file_path": self.file_path,
            "file_type": self.file_type,
            "title": self.title,
            "author": self.author,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "modified_at": self.modified_at.isoformat() if self.modified_at else None,
            "processed_at": self.processed_at.isoformat(),
            "page_count": self.page_count,
            "word_count": self.word_count,
            "language": self.language,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create document from dictionary representation"""
        # Convert ISO format strings back to datetime objects
        for field in ["created_at", "modified_at", "processed_at"]:
            if data.get(field):
                data[field] = datetime.fromisoformat(data[field])

        return cls(**data)
