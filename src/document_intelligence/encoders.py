"""Pluggable dense encoders with an offline deterministic fallback."""

from __future__ import annotations

import os
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class TextEncoder(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashingEncoder:
    """Fast, dependency-light character n-gram representation.

    This fallback makes the whole project runnable offline. Use the semantic
    encoder for cross-language retrieval.
    """

    name = "hashing-char-ngrams"

    def __init__(self, dimensions: int = 2048) -> None:
        self.vectorizer = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            n_features=dimensions,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.vectorizer.n_features), dtype=np.float32)
        return self.vectorizer.transform(texts).toarray().astype(np.float32)


class SentenceTransformerEncoder:
    name = "multilingual-sentence-transformer"

    def __init__(self, model_name: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Semantic retrieval requires: pip install -e '.[semantic]'"
            ) from exc
        selected_model = model_name or os.getenv(
            "DOCUMENT_AI_EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self.model_name = selected_model
        self.model = SentenceTransformer(selected_model)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)


def create_encoder(mode: str | None = None) -> TextEncoder:
    selected = (mode or os.getenv("DOCUMENT_AI_ENCODER", "hashing")).casefold()
    if selected in {"semantic", "multilingual", "sentence-transformer"}:
        return SentenceTransformerEncoder()
    if selected in {"hashing", "offline", "fast"}:
        return HashingEncoder()
    raise ValueError("Encoder must be 'hashing' or 'semantic'")

