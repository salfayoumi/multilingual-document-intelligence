"""Pluggable dense encoders with an offline deterministic fallback."""

from __future__ import annotations

import os
import warnings
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class TextEncoder(Protocol):
    name: str
    cross_language: bool

    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashingEncoder:
    """Fast, dependency-light character n-gram representation.

    This fallback makes the whole project runnable offline. Use the semantic
    encoder for cross-language retrieval.
    """

    name = "hashing-char-ngrams"
    cross_language = False

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


class FastEmbedEncoder:
    """Compact ONNX multilingual embeddings suitable for the hosted demo."""

    name = "fastembed-multilingual-minilm"
    cross_language = True

    def __init__(self, model_name: str | None = None) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError(
                "Semantic retrieval requires: pip install -e '.[semantic]'"
            ) from exc
        selected_model = model_name or os.getenv(
            "DOCUMENT_AI_EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self.model_name = selected_model
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"The model .* now uses mean pooling instead of CLS embedding.*",
                    category=UserWarning,
                )
                self.model = TextEmbedding(model_name=selected_model, threads=2)
        except Exception as exc:
            raise RuntimeError(f"Could not load multilingual embedding model: {exc}") from exc
        self.dimensions = self.model.embedding_size

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        vectors = list(self.model.embed(texts))
        return np.asarray(vectors, dtype=np.float32)


def create_encoder(mode: str | None = None) -> TextEncoder:
    selected = (mode or os.getenv("DOCUMENT_AI_ENCODER", "hashing")).casefold()
    if selected in {"semantic", "multilingual", "sentence-transformer"}:
        return FastEmbedEncoder()
    if selected in {"hashing", "offline", "fast"}:
        return HashingEncoder()
    raise ValueError("Encoder must be 'hashing' or 'semantic'")
