"""High-level indexing, retrieval, and question-answering engine."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .answering import Answerer, create_answerer
from .encoders import TextEncoder, create_encoder
from .ingestion import chunk_documents, load_document
from .models import Answer, Chunk, Document, SearchResult
from .retrieval import HybridRetriever


class DocumentIntelligence:
    def __init__(
        self,
        *,
        encoder: TextEncoder | None = None,
        answerer: Answerer | None = None,
        chunk_size: int = 900,
        overlap: int = 140,
    ) -> None:
        self.encoder = encoder or create_encoder()
        self.answerer = answerer or create_answerer()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.documents: dict[str, Document] = {}
        self.chunks: list[Chunk] = []
        self.retriever: HybridRetriever | None = None

    def index_documents(self, documents: Iterable[Document]) -> dict[str, object]:
        for document in documents:
            self.documents[document.id] = document
        self.chunks = chunk_documents(
            self.documents.values(), chunk_size=self.chunk_size, overlap=self.overlap
        )
        self.retriever = HybridRetriever(self.chunks, self.encoder) if self.chunks else None
        return self.stats()

    def add_documents(self, documents: Iterable[Document]) -> dict[str, object]:
        return self.index_documents([*self.documents.values(), *documents])

    def index_paths(self, paths: Iterable[str | Path]) -> dict[str, object]:
        return self.add_documents(load_document(path) for path in paths)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        dense_weight: float = 0.60,
        language: str | None = None,
    ) -> list[SearchResult]:
        if self.retriever is None:
            raise RuntimeError("Index at least one document before searching")
        return self.retriever.search(
            query, top_k=top_k, dense_weight=dense_weight, language=language
        )

    def ask(self, query: str, *, top_k: int = 5, dense_weight: float = 0.60) -> Answer:
        return self.answerer.answer(
            query, self.search(query, top_k=top_k, dense_weight=dense_weight)
        )

    def stats(self) -> dict[str, object]:
        languages: dict[str, int] = {}
        for document in self.documents.values():
            languages[document.language] = languages.get(document.language, 0) + 1
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "languages": languages,
            "encoder": self.encoder.name,
            "answer_mode": self.answerer.mode,
        }

    def clear(self) -> None:
        self.documents.clear()
        self.chunks.clear()
        self.retriever = None
