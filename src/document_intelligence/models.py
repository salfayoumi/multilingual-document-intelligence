"""Domain models shared by ingestion, retrieval, answering, and APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    name: str
    text: str
    language: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Document text cannot be empty")


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    document_id: str
    source: str
    text: str
    language: str
    position: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchResult:
    chunk: Chunk
    score: float
    dense_score: float
    lexical_score: float
    rank: int


@dataclass(frozen=True, slots=True)
class Citation:
    number: int
    source: str
    chunk_id: str
    snippet: str
    score: float
    page: int | None = None


@dataclass(frozen=True, slots=True)
class Answer:
    query: str
    text: str
    citations: tuple[Citation, ...]
    confidence: float
    supported: bool
    language: str
    mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.text,
            "confidence": round(self.confidence, 4),
            "supported": self.supported,
            "language": self.language,
            "mode": self.mode,
            "citations": [
                {
                    "number": item.number,
                    "source": item.source,
                    "chunk_id": item.chunk_id,
                    "snippet": item.snippet,
                    "score": round(item.score, 4),
                    "page": item.page,
                }
                for item in self.citations
            ],
        }

