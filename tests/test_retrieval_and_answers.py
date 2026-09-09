from __future__ import annotations

import unittest

from document_intelligence import DocumentIntelligence, document_from_text
from document_intelligence.answering import ExtractiveAnswerer
from document_intelligence.encoders import HashingEncoder


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
        answerer = ExtractiveAnswerer(minimum_dense_score=0.95)
        answer = answerer.answer("Who won the championship?", self.engine.search("championship"))
        self.assertFalse(answer.supported)
        self.assertEqual(answer.citations, ())


if __name__ == "__main__":
    unittest.main()
