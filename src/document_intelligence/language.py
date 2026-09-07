"""Lightweight multilingual text normalization and language hints."""

from __future__ import annotations

import re
import unicodedata

ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
TOKEN_PATTERN = re.compile(r"[\w\u0600-\u06FF]+", re.UNICODE)
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?؟؛])\s+|\n+")

TURKISH_HINTS = {
    "bir",
    "bu",
    "için",
    "ile",
    "ve",
    "veya",
    "olarak",
    "gereken",
    "bakım",
    "sistem",
}
ENGLISH_HINTS = {"the", "and", "for", "with", "from", "system", "maintenance", "should"}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = ARABIC_DIACRITICS.sub("", text).replace("ـ", "")
    text = text.casefold()
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_PATTERN.findall(normalize_text(text)) if len(token) > 1]


def detect_language(text: str) -> str:
    """Return an explainable language hint: ar, tr, en, or unknown.

    This is intentionally small and deterministic. It is a UI/indexing hint,
    not a replacement for a production language-identification model.
    """

    normalized = normalize_text(text)
    letters = [char for char in normalized if char.isalpha()]
    if not letters:
        return "unknown"

    arabic_count = sum("\u0600" <= char <= "\u06ff" for char in letters)
    if arabic_count / len(letters) >= 0.15:
        return "ar"

    tokens = set(tokenize(normalized))
    turkish_chars = sum(char in "çğıöşü" for char in normalized)
    tr_score = len(tokens & TURKISH_HINTS) + min(turkish_chars, 3)
    en_score = len(tokens & ENGLISH_HINTS)
    if tr_score >= 2 and tr_score > en_score:
        return "tr"
    if en_score or any("a" <= char <= "z" for char in letters):
        return "en"
    return "unknown"


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_BOUNDARY.split(text) if part.strip()]

