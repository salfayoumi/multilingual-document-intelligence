"""Multilingual, citation-first document intelligence."""

from .engine import DocumentIntelligence
from .ingestion import document_from_text, load_document
from .models import Answer, Chunk, Citation, Document, SearchResult

__all__ = [
    "Answer",
    "Chunk",
    "Citation",
    "Document",
    "DocumentIntelligence",
    "SearchResult",
    "document_from_text",
    "load_document",
]

__version__ = "0.1.0"

