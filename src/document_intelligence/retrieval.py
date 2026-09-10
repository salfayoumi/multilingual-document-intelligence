"""Hybrid dense and BM25 retrieval with reciprocal-rank fusion."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .encoders import TextEncoder
from .language import content_tokens
from .models import Chunk, SearchResult


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.term_frequencies = [Counter(content_tokens(chunk.text)) for chunk in chunks]
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
        query_terms = content_tokens(query)
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

    def coverage(self, query: str) -> np.ndarray:
        """Return the share of unique query terms present in each passage."""

        query_terms = set(content_tokens(query))
        if not query_terms:
            return np.zeros(len(self.term_frequencies), dtype=np.float32)
        coverages = [
            len(query_terms & set(frequencies)) / len(query_terms)
            for frequencies in self.term_frequencies
        ]
        return np.asarray(
            coverages,
            dtype=np.float32,
        )


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

    def _reciprocal_ranks(self, scores: np.ndarray) -> np.ndarray:
        """Return RRF contributions without ranking zero-evidence candidates."""

        contributions = np.zeros(len(scores), dtype=np.float32)
        valid = np.flatnonzero(scores > 0)
        if not len(valid):
            return contributions
        order = valid[np.argsort(-scores[valid], kind="stable")]
        contributions[order] = 1 / (self.fusion_k + np.arange(1, len(order) + 1))
        return contributions

    def _evidence_scores(
        self,
        dense_scores: np.ndarray,
        lexical_scores: np.ndarray,
        lexical_coverage: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map raw retrieval signals to an absolute, non-rank confidence score."""

        dense_support = np.clip(dense_scores, 0.0, 1.0)
        lexical_strength = 1 - np.exp(-np.maximum(lexical_scores, 0.0) / 3.0)
        lexical_support = lexical_strength * np.sqrt(lexical_coverage)
        if getattr(self.encoder, "cross_language", False):
            evidence = dense_support
            ranking = 0.85 * dense_support + 0.15 * lexical_support
        else:
            evidence = np.maximum(dense_support, lexical_support)
            ranking = 0.55 * dense_support + 0.45 * lexical_support
        return evidence.astype(np.float32), ranking.astype(np.float32)

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
        lexical_coverage = self.bm25.coverage(query)
        fused = dense_weight * self._reciprocal_ranks(dense_scores)
        fused += (1 - dense_weight) * self._reciprocal_ranks(lexical_scores)
        evidence_scores, ranking_scores = self._evidence_scores(
            dense_scores, lexical_scores, lexical_coverage
        )

        # Calibrated hybrid evidence decides rank. RRF breaks ties between
        # candidates with similarly strong semantic and lexical support.
        candidate_indices = np.lexsort((-fused, -ranking_scores))
        results: list[SearchResult] = []
        for index in candidate_indices:
            chunk = self.chunks[int(index)]
            if language and chunk.language != language:
                continue
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=float(evidence_scores[index]),
                    dense_score=float(dense_scores[index]),
                    lexical_score=float(lexical_scores[index]),
                    rank=len(results) + 1,
                )
            )
            if len(results) >= top_k:
                break
        return results
