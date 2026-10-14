"""Shared text normalization for lexical memory retrieval."""

from __future__ import annotations

import re
import unicodedata
from typing import Final

_ENGLISH_STOPWORDS: Final[set[str]] = {
    "a",
    "about",
    "an",
    "and",
    "any",
    "are",
    "assist",
    "be",
    "can",
    "could",
    "do",
    "for",
    "find",
    "get",
    "good",
    "help",
    "i",
    "in",
    "is",
    "it",
    "like",
    "looking",
    "me",
    "my",
    "need",
    "of",
    "on",
    "option",
    "please",
    "some",
    "that",
    "the",
    "this",
    "to",
    "upcoming",
    "want",
    "with",
    "would",
    "you",
    "your",
}

def expand_query_terms(text: str) -> list[str]:
    """Return data-independent lexical terms for backward compatibility.

    Cross-language and synonym matching belongs to the configured embedding
    provider. Keeping a product/domain vocabulary here makes retrieval depend
    on hand-maintained cases and can contaminate blind evaluation.
    """
    return tokenize_search_text(text)


def tokenize_search_text(text: str) -> list[str]:
    """Return stable search tokens with light English morphology handling."""
    normalized = unicodedata.normalize("NFKC", text)
    # Split camelCase/PascalCase before case-folding, then handle common
    # identifier separators uniformly. This is grammar-level normalization,
    # not a vocabulary or synonym table.
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", normalized).casefold()
    raw_ascii = re.findall(r"[a-z0-9_./\\'-]{2,}", normalized)
    expanded_ascii: list[str] = []
    for term in raw_ascii:
        expanded_ascii.append(term)
        expanded_ascii.extend(
            part for part in re.split(r"[_./\\-]+", term) if len(part) >= 2
        )
    ascii_terms = [_normalize_english(term) for term in expanded_ascii]
    ascii_terms = [
        term for term in ascii_terms if term and term not in _ENGLISH_STOPWORDS
    ]

    cjk_terms: list[str] = []
    for sequence in re.findall(r"[\u4e00-\u9fff]+", normalized):
        cjk_terms.extend(sequence)
        if len(sequence) >= 2:
            cjk_terms.extend(
                sequence[index : index + 2] for index in range(len(sequence) - 1)
            )
        if 2 <= len(sequence) <= 12:
            cjk_terms.append(sequence)

    terms = sorted(set(ascii_terms + cjk_terms))
    if terms:
        return terms
    return sorted({_normalize_english(term) for term in expanded_ascii if term})


def term_weight(term: str) -> float:
    """Weight informative long terms above individual CJK characters."""
    if len(term) == 1 and "\u4e00" <= term <= "\u9fff":
        return 0.2
    return 1.0 + min(len(term), 16) / 16.0


def _normalize_english(term: str) -> str:
    clean = term.strip("'-")
    if clean.endswith("'s") and len(clean) > 3:
        clean = clean[:-2]
    if not clean.isalpha():
        return clean
    if len(clean) > 4 and clean.endswith("ies"):
        return clean[:-3] + "y"
    if (
        len(clean) > 4
        and clean.endswith("s")
        and not clean.endswith(("ss", "us", "is"))
    ):
        return clean[:-1]
    return clean
