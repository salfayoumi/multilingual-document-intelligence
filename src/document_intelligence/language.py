"""Lightweight multilingual text normalization and language hints.

The project supports a deliberately small language set.  The detector is used
to describe documents and passages; retrieval quality comes from the encoder,
not from pretending this heuristic is a general-purpose language classifier.
"""

from __future__ import annotations

import re
import unicodedata

ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
TOKEN_PATTERN = re.compile(r"[^\W\d_]+|\d+(?:[.,]\d+)?", re.UNICODE)
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?؟؛])\s+|\n+")

SUPPORTED_LANGUAGES = ("ar", "tr", "en")

TURKISH_HINTS = {
    "ama",
    "bakım",
    "bir",
    "bu",
    "de",
    "da",
    "değil",
    "gibi",
    "göre",
    "her",
    "için",
    "ile",
    "ise",
    "kadar",
    "olan",
    "ve",
    "veya",
    "olarak",
    "gereken",
    "sistem",
    "sonra",
    "önce",
    "üretim",
    "yapılmalıdır",
}
ENGLISH_HINTS = {
    "a",
    "an",
    "and",
    "are",
    "before",
    "for",
    "from",
    "is",
    "maintenance",
    "of",
    "should",
    "system",
    "the",
    "to",
    "with",
}

STOPWORDS = {
    "en": {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "before",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "should",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
    },
    "tr": {
        "ama",
        "bir",
        "bu",
        "da",
        "de",
        "gibi",
        "göre",
        "hangi",
        "her",
        "için",
        "ile",
        "ise",
        "kadar",
        "mi",
        "mı",
        "mu",
        "mü",
        "nasıl",
        "ne",
        "neden",
        "ve",
        "veya",
    },
    "ar": {
        "أو",
        "إلى",
        "التي",
        "الذي",
        "عن",
        "على",
        "في",
        "قبل",
        "كيف",
        "ما",
        "ماذا",
        "متى",
        "من",
        "هل",
        "هو",
        "هي",
        "و",
    },
}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = ARABIC_DIACRITICS.sub("", text).replace("ـ", "")
    text = text.casefold()
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_PATTERN.findall(normalize_text(text)) if len(token) > 1]


def content_tokens(text: str) -> list[str]:
    """Return lexical-search tokens with common question words removed."""

    language = detect_language(text)
    stopwords = STOPWORDS.get(language, set())
    return [token for token in tokenize(text) if token not in stopwords]


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
    if tr_score > en_score and (turkish_chars >= 1 or tr_score >= 2):
        return "tr"
    if en_score or any("a" <= char <= "z" for char in letters):
        return "en"
    return "unknown"


def detect_languages(text: str) -> tuple[str, ...]:
    """Return every supported language found across a document or passage.

    Detection is performed on paragraph/sentence-sized units so an Arabic
    paragraph does not cause the English and Turkish parts of the same file to
    disappear behind one document-level label.
    """

    found: list[str] = []
    units = split_sentences(text) or [text]
    for unit in units:
        language = detect_language(unit)
        if language in SUPPORTED_LANGUAGES and language not in found:
            found.append(language)
    if found:
        return tuple(found)
    language = detect_language(text)
    return (language,) if language in SUPPORTED_LANGUAGES else ()


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_BOUNDARY.split(text) if part.strip()]
