"""Hybrid dense and BM25 retrieval with reciprocal-rank fusion."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .encoders import TextEncoder
from .language import tokenize
from .models import Chunk, SearchResult


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.term_frequencies = [Counter(tokenize(chunk.text)) for chunk in chunks]
        self.lengths = [sum(counter.values()) for counter in self.term_frequencies]
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)
        document_frequency: Counter[str] = Counter()
        for counter in self.term_frequencies:
            document_frequency.update(counter.keys())
        count = len(chunks)
        self.idf = {
            term: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def score(self, query: str) -> np.ndarray:
        query_terms = tokenize(query)
        scores = np.zeros(len(self.term_frequencies), dtype=np.float32)
        for index, frequencies in enumerate(self.term_frequencies):
            length = self.lengths[index]
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / max(self.average_length, 1)
                )
                scores[index] += self.idf.get(term, 0.0) * (
                    frequency * (self.k1 + 1) / denominator
                )
        return scores


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], encoder: TextEncoder, *, fusion_k: int = 60) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required")
        self.chunks = chunks
        self.encoder = encoder
        self.fusion_k = fusion_k
        self.bm25 = BM25Index(chunks)
        self.vectors = _normalize_rows(encoder.encode([chunk.text for chunk in chunks]))

    @staticmethod
    def _ranks(scores: np.ndarray) -> np.ndarray:
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)
        return ranks

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        dense_weight: float = 0.60,
        language: str | None = None,
    ) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("Query cannot be empty")
        if not 0 <= dense_weight <= 1:
            raise ValueError("dense_weight must be between 0 and 1")

        query_vector = _normalize_rows(self.encoder.encode([query]))[0]
        dense_scores = self.vectors @ query_vector
        lexical_scores = self.bm25.score(query)
        dense_ranks = self._ranks(dense_scores)
        lexical_ranks = self._ranks(lexical_scores)

        fused = dense_weight / (self.fusion_k + dense_ranks)
        fused += (1 - dense_weight) / (self.fusion_k + lexical_ranks)
        if fused.max() > 0:
            fused = fused / fused.max()

        candidate_indices = np.argsort(-fused, kind="stable")
        results: list[SearchResult] = []
        for index in candidate_indices:
            chunk = self.chunks[int(index)]
            if language and chunk.language != language:
                continue
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=float(fused[index]),
                    dense_score=float(dense_scores[index]),
                    lexical_score=float(lexical_scores[index]),
                    rank=len(results) + 1,
                )
            )
            if len(results) >= top_k:
                break
        return results

