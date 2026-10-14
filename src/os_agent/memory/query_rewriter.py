"""Query planning for multi-route memory retrieval."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar

from os_agent.memory.retrieval_text import tokenize_search_text


@dataclass(frozen=True, slots=True)
class QueryPlan:
    """A bounded retrieval plan produced without looking at memory labels."""

    original: str
    variants: tuple[str, ...] = ()
    subqueries: tuple[str, ...] = ()

    @property
    def retrieval_queries(self) -> tuple[str, ...]:
        return _unique((self.original, *self.variants, *self.subqueries))


class QueryRewriter:
    """Normalize queries and compose externally supplied retrieval routes.

    Translation, HyDE, and LLM decomposition can supply variants or subqueries
    through the stable plan interface. The deterministic fallback only removes
    conversational filler and splits explicit clause lists; it has no domain or
    benchmark vocabulary.
    """

    MAX_ROUTES: ClassVar[int] = 8
    FILLERS_CN: ClassVar[tuple[str, ...]] = (
        r"帮我",
        r"请(?:你|您)?",
        r"(?:给|帮)我(?:看|查|找|搜)(?:一下|一下下)?",
        r"(?:麻烦|劳驾)",
        r"一下",
        r"看看",
        r"能不能",
        r"可不可以",
    )
    _LEADING_ACT = re.compile(
        r"^(?:(?:请问|请|能否|可以|告诉我|我想知道|想知道|帮我)\s*)+"
        r"|^(?:(?:please|could you|can you|would you|tell me)\s*)+",
        re.IGNORECASE,
    )
    _TRAILING_QUESTION = re.compile(r"[?？。.!！]+\s*$")
    _CLAUSE_SEPARATOR = re.compile(r"(?:\r?\n|[,，;；])+")
    _COORDINATOR = re.compile(r"(?:以及|同时|分别|并且|\b(?:and|also|respectively)\b)", re.IGNORECASE)

    def rewrite(self, user_input: str) -> str:
        """Return one conservative normalized query for compatibility."""
        text = unicodedata.normalize("NFKC", user_input.strip())
        text = " ".join(text.split())
        if not text:
            return text
        text = self._LEADING_ACT.sub("", text)
        # Keep filler removal bounded to leading conversational acts. A word
        # such as "and" or "一下" may be meaningful inside the memory value.
        for pattern in self.FILLERS_CN:
            text = re.sub(rf"^(?:{pattern})\s*", "", text, flags=re.IGNORECASE)
        if re.search(r"[\u4e00-\u9fff]", text):
            text = self._TRAILING_QUESTION.sub("", text)
        return " ".join(text.split()).strip() or user_input.strip()

    def plan(
        self,
        user_input: str,
        *,
        variants: Iterable[str] = (),
        subqueries: Iterable[str] = (),
        rewrite: bool = True,
    ) -> QueryPlan:
        original = " ".join(user_input.strip().split())
        if not original:
            return QueryPlan(original="")

        planned_variants = list(_clean(variants))
        normalized = self.rewrite(original) if rewrite else original
        if normalized.casefold() != original.casefold():
            planned_variants.insert(0, normalized)
        projection = self._keyword_projection(normalized)
        if projection and projection.casefold() != normalized.casefold():
            planned_variants.append(projection)
        planned_subqueries = list(_clean(subqueries))
        clauses = self._split_clauses(original)
        if len(clauses) > 1:
            planned_subqueries.extend(clauses)

        remaining = self.MAX_ROUTES - 1
        unique_variants = _unique(planned_variants)[:remaining]
        remaining -= len(unique_variants)
        unique_subqueries = tuple(
            query
            for query in _unique(planned_subqueries)
            if query.casefold() != original.casefold()
            and query.casefold() not in {item.casefold() for item in unique_variants}
        )[:remaining]
        return QueryPlan(
            original=original,
            variants=unique_variants,
            subqueries=unique_subqueries,
        )

    @classmethod
    def _split_clauses(cls, text: str) -> tuple[str, ...]:
        clauses = list(_clean(cls._CLAUSE_SEPARATOR.split(text)))
        if len(clauses) <= 1 and cls._COORDINATOR.search(text):
            clauses = list(_clean(cls._COORDINATOR.split(text)))
        clauses = [
            re.sub(
                r"^(?:以及|同时|分别|并且|\b(?:and|also|respectively)\b)\s*",
                "",
                clause,
                flags=re.IGNORECASE,
            )
            for clause in clauses
        ]
        return tuple(clause for clause in clauses if len(clause) >= 2)

    @staticmethod
    def _keyword_projection(text: str) -> str:
        """Create a bounded lexical route for script and morphology variance."""
        if not re.search(r"[\u4e00-\u9fff]", text):
            return ""
        tokens = tokenize_search_text(text)
        if len(tokens) < 2:
            return ""
        # Preserve complete CJK runs and searchable ASCII terms. The retrieval
        # index performs the finer-grained bigram expansion itself.
        runs = re.findall(r"[\u4e00-\u9fff]+", text)
        ascii_terms = [token for token in tokens if re.fullmatch(r"[a-z0-9_.-]+", token)]
        projected = " ".join(dict.fromkeys([*runs, *ascii_terms]))
        return projected if projected != text else ""


def _clean(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        clean
        for value in values
        if (clean := " ".join(str(value).strip().split()))
    )


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in _clean(values):
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return tuple(result)
