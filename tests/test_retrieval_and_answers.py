from __future__ import annotations

import unittest

import numpy as np

from document_intelligence import DocumentIntelligence, document_from_text
from document_intelligence.answering import ExtractiveAnswerer
from document_intelligence.encoders import HashingEncoder
from document_intelligence.models import Chunk, SearchResult


class ConceptEncoder:
    """Tiny deterministic encoder used to test cross-language retrieval logic."""

    name = "test-multilingual-concepts"
    cross_language = True

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            normalized = text.casefold()
            vectors.append(
                [
                    float(any(term in normalized for term in ("vibration", "titreşim", "اهتزاز"))),
                    float(any(term in normalized for term in ("filter", "filtre", "مرشح"))),
                ]
            )
        return np.asarray(vectors, dtype=np.float32)


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = DocumentIntelligence(encoder=HashingEncoder(), chunk_size=300, overlap=40)
        self.engine.index_documents(
            [
                document_from_text(
                    "motor.md",
                    "The emergency vibration threshold is 11.2 mm/s. "
                    "Stop the motor before inspection.",
                ),
                document_from_text(
                    "quality.md",
                    "Inspect five samples from every batch of two hundred manufactured parts.",
                ),
                document_from_text(
                    "safety_ar.md",
                    "يجب عزل مصدر الكهرباء قبل فتح غطاء المعدة وإجراء الفحص.",
                ),
            ]
        )

    def test_hybrid_search_ranks_exact_evidence_first(self) -> None:
        results = self.engine.search("What is the emergency vibration threshold?", top_k=2)
        self.assertEqual(results[0].chunk.source, "motor.md")
        self.assertGreater(results[0].score, results[1].score)

    def test_arabic_query_retrieves_arabic_source(self) -> None:
        results = self.engine.search("ماذا يجب أن نفعل قبل فتح غطاء المعدة؟", top_k=1)
        self.assertEqual(results[0].chunk.source, "safety_ar.md")

    def test_answer_contains_traceable_citations(self) -> None:
        answer = self.engine.ask("What is the emergency vibration threshold?")
        self.assertTrue(answer.supported)
        self.assertIn("[1]", answer.text)
        self.assertEqual(answer.citations[0].source, "motor.md")

    def test_extractive_answer_does_not_append_lower_ranked_topics(self) -> None:
        answer = self.engine.ask("What is the emergency vibration threshold?")
        self.assertEqual(len(answer.citations), 1)
        self.assertNotIn("manufactured parts", answer.text)

    def test_answer_refuses_when_no_evidence_matches(self) -> None:
        answer = self.engine.ask("Who won the World Cup in 2022?")
        self.assertFalse(answer.supported)
        self.assertEqual(answer.citations, ())
        self.assertEqual(answer.confidence, 0.0)

    def test_confidence_is_absolute_evidence_not_rank_normalization(self) -> None:
        answer = self.engine.ask("What is the emergency vibration threshold?")
        self.assertTrue(answer.supported)
        self.assertGreater(answer.confidence, 0.32)
        self.assertLess(answer.confidence, 1.0)

    def test_cross_language_query_can_retrieve_supported_evidence(self) -> None:
        engine = DocumentIntelligence(encoder=ConceptEncoder())
        engine.index_documents(
            [
                document_from_text(
                    "motor_en.md",
                    "The vibration alarm must be escalated after ten continuous seconds.",
                ),
                document_from_text(
                    "filter_tr.md",
                    "Ana hat filtresi her beş yüz çalışma saatinde değiştirilir.",
                ),
            ]
        )

        answer = engine.ask("متى يجب تصعيد إنذار الاهتزاز؟")
        self.assertTrue(answer.supported)
        self.assertEqual(answer.citations[0].source, "motor_en.md")

    def test_decisive_lexical_overlap_resolves_close_semantic_scores(self) -> None:
        query = "What temperature requires the equipment to be stopped?"
        results = [
            SearchResult(
                Chunk(
                    id="temperature",
                    document_id="maintenance",
                    source="maintenance_en.md",
                    text=(
                        "## Temperature response\n\nAt 90 °C, stop the equipment and "
                        "investigate before restarting."
                    ),
                    language="en",
                    position=0,
                ),
                score=0.615,
                dense_score=0.615,
                lexical_score=4.677,
                rank=1,
            ),
            SearchResult(
                Chunk(
                    id="unrelated",
                    document_id="safety",
                    source="safety_ar.md",
                    text="يجب عزل مصدر الكهرباء قبل فتح غطاء المعدة.",
                    language="ar",
                    position=0,
                ),
                score=0.608,
                dense_score=0.608,
                lexical_score=0.0,
                rank=2,
            ),
        ]

        answer = ExtractiveAnswerer().answer(query, results)

        self.assertTrue(answer.supported)
        self.assertEqual(answer.citations[0].source, "maintenance_en.md")

    def test_close_semantic_scores_without_lexical_evidence_remain_unsupported(self) -> None:
        results = [
            SearchResult(
                Chunk("first", "one", "one.md", "First possible topic.", "en", 0),
                score=0.61,
                dense_score=0.61,
                lexical_score=0.0,
                rank=1,
            ),
            SearchResult(
                Chunk("second", "two", "two.md", "Second possible topic.", "en", 0),
                score=0.59,
                dense_score=0.59,
                lexical_score=0.0,
                rank=2,
            ),
        ]

        answer = ExtractiveAnswerer().answer("ما الإجراء المطلوب؟", results)

        self.assertFalse(answer.supported)
        self.assertEqual(answer.citations, ())


if __name__ == "__main__":
    unittest.main()
