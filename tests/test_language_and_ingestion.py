from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from document_intelligence.ingestion import chunk_document, document_from_text, load_document
from document_intelligence.language import detect_language, normalize_text, tokenize


class LanguageTests(unittest.TestCase):
    def test_detects_supported_languages(self) -> None:
        self.assertEqual(detect_language("The maintenance system should stop the motor."), "en")
        self.assertEqual(
            detect_language("Bu sistem için bakım yapılması gereken süre nedir?"), "tr"
        )
        self.assertEqual(detect_language("يجب إيقاف المعدة قبل بدء أعمال الصيانة."), "ar")

    def test_normalizes_arabic_diacritics(self) -> None:
        self.assertEqual(normalize_text("السَّلَامُ"), "السلام")
        self.assertIn("bakım", tokenize("Bakım için kontrol"))


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


if __name__ == "__main__":
    unittest.main()
