from __future__ import annotations

import unittest

from document_intelligence import DocumentIntelligence, document_from_text
from document_intelligence.encoders import HashingEncoder
from document_intelligence.evaluation import EvaluationCase, evaluate


class EvaluationTests(unittest.TestCase):
    def test_reports_retrieval_metrics(self) -> None:
        engine = DocumentIntelligence(encoder=HashingEncoder())
        engine.index_documents(
            [
                document_from_text("alpha.md", "Bearing vibration threshold is seven millimetres."),
                document_from_text("beta.md", "Quality samples are checked every morning."),
            ]
        )
        metrics = evaluate(
            engine,
            [EvaluationCase("bearing vibration", ("alpha.md",))],
            top_k=1,
        )
        self.assertEqual(metrics["hit_rate@1"], 1.0)
        self.assertEqual(metrics["mean_reciprocal_rank"], 1.0)
        self.assertGreaterEqual(metrics["mean_latency_ms"], 0)


if __name__ == "__main__":
    unittest.main()

