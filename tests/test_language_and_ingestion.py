from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from document_intelligence import DocumentIntelligence
from document_intelligence.ingestion import chunk_document, document_from_text, load_document
from document_intelligence.language import (
    content_tokens,
    detect_language,
    detect_languages,
    normalize_text,
    tokenize,
)


class LanguageTests(unittest.TestCase):
    def test_detects_supported_languages(self) -> None:
        self.assertEqual(detect_language("The maintenance system should stop the motor."), "en")
        self.assertEqual(
            detect_language("Bu sistem için bakım yapılması gereken süre nedir?"), "tr"
        )
        self.assertEqual(detect_language("Numune planı"), "tr")
        self.assertEqual(detect_language("يجب إيقاف المعدة قبل بدء أعمال الصيانة."), "ar")

    def test_normalizes_arabic_diacritics(self) -> None:
        self.assertEqual(normalize_text("السَّلَامُ"), "السلام")
        self.assertIn("bakım", tokenize("Bakım için kontrol"))

    def test_detects_every_language_in_mixed_text(self) -> None:
        text = (
            "The maintenance report explains the shutdown procedure.\n\n"
            "Bakım raporu sistemin durdurulma sürecini açıklar.\n\n"
            "يشرح تقرير الصيانة خطوات إيقاف النظام قبل الفحص."
        )
        self.assertEqual(detect_languages(text), ("en", "tr", "ar"))
        self.assertNotIn("the", content_tokens("Who won the World Cup in 2022?"))


class IngestionTests(unittest.TestCase):
    def test_text_file_loading_uses_original_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guide.txt"
            path.write_text("The system contains a documented safety procedure.", encoding="utf-8")
            document = load_document(path)
        self.assertEqual(document.name, "guide.txt")
        self.assertEqual(document.language, "en")

    def test_chunk_ids_are_deterministic_and_overlap(self) -> None:
        text = " ".join(f"sentence-{index}." for index in range(260))
        document = document_from_text("long.txt", text)
        first = chunk_document(document, chunk_size=300, overlap=50)
        second = chunk_document(document, chunk_size=300, overlap=50)
        self.assertGreater(len(first), 2)
        self.assertEqual([chunk.id for chunk in first], [chunk.id for chunk in second])
        self.assertLess(first[1].metadata["start_char"], first[0].metadata["end_char"])

    def test_mixed_document_creates_language_aware_passages(self) -> None:
        document = document_from_text(
            "mixed.md",
            """# Operations guide

The vibration alarm is escalated after ten seconds.

## Bakım notu

Filtre her beş yüz çalışma saatinde değiştirilir.

## تعليمات السلامة

يجب عزل مصدر الكهرباء قبل فتح غطاء المعدة.
""",
        )
        chunks = chunk_document(document)

        self.assertEqual(document.language, "mixed")
        self.assertEqual(set(document.languages), {"ar", "tr", "en"})
        self.assertEqual({chunk.language for chunk in chunks}, {"ar", "tr", "en"})

        engine = DocumentIntelligence()
        stats = engine.index_documents([document])
        self.assertEqual(set(stats["languages"]), {"ar", "tr", "en"})


if __name__ == "__main__":
    unittest.main()
