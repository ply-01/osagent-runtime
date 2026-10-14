"""Language-agnostic alignment helpers for memory facts.

The memory stores deliberately keep their logical keys stable. Natural language
often produces a different spelling for the same attribute, however, so merge
decisions need a small, explainable alignment layer before falling back to a
new key. This module uses Unicode normalization and token evidence only; it
does not contain benchmark entities, values, or sentence templates.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from os_agent.memory.retrieval_text import tokenize_search_text

if TYPE_CHECKING:
    from os_agent.memory.mem0_schemas import FactCard


_NON_WORD = re.compile(r"[^\w\u4e00-\u9fff]+", re.UNICODE)
_IDENTIFIER_SEPARATOR = re.compile(r"[_./\\-]+")
_NON_ALIGNING_INTENTS = {"query", "reported", "uncertain", "ignore"}


@dataclass(frozen=True, slots=True)
class FactAlignment:
    """Explain why two facts are considered the same logical slot."""

    entity_match: bool
    scope_match: bool
    attribute_score: float
    value_score: float
    score: float

    @property
    def same_value(self) -> bool:
        return self.value_score >= 0.88

    @property
    def can_merge(self) -> bool:
        # A changed value is precisely what makes a version conflict. Attribute
        # identity therefore gates the merge; value similarity only decides
        # whether to reinforce or supersede the existing fact.
        return self.entity_match and self.scope_match and self.attribute_score >= 0.68


def normalize_text(value: object) -> str:
    """Normalize mixed Chinese/English text before token comparison."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = _NON_WORD.sub(" ", text)
    return " ".join(text.split())


def normalize_identifier(value: object) -> str:
    """Normalize equivalent identifier separators without alias vocabulary."""
    return " ".join(_IDENTIFIER_SEPARATOR.sub(" ", normalize_text(value)).split())


def alignment(left: FactCard, right: FactCard) -> FactAlignment:
    """Return a conservative, inspectable fact alignment score."""
    left_entity, left_scope = _entity_scope(left)
    right_entity, right_scope = _entity_scope(right)
    entity_match = normalize_identifier(left_entity) == normalize_identifier(right_entity)
    scope_match = normalize_identifier(left_scope) == normalize_identifier(right_scope)
    attribute_score = attribute_similarity(left.attribute, right.attribute)
    value_score = value_similarity(left.value, right.value)
    # Attribute identity carries more weight than the observed value: two
    # different values should conflict, while similar labels alone should not.
    score = 0.62 * attribute_score + 0.38 * value_score
    return FactAlignment(
        entity_match=entity_match,
        scope_match=scope_match,
        attribute_score=attribute_score,
        value_score=value_score,
        score=score,
    )


def find_aligned_fact(
    incoming: FactCard,
    candidates: Iterable[FactCard],
    *,
    threshold: float = 0.68,
) -> tuple[FactCard, FactAlignment] | None:
    """Find the strongest active fact in the same entity/scope context."""
    if not _eligible(incoming):
        return None
    matches: list[tuple[float, str, FactCard, FactAlignment]] = []
    for candidate in candidates:
        if candidate.status != "active" or not _eligible(candidate):
            continue
        relation = alignment(incoming, candidate)
        if relation.can_merge and relation.attribute_score >= threshold:
            matches.append((relation.score, candidate.fact_id, candidate, relation))
    if not matches:
        return None
    matches.sort(key=lambda item: (-item[0], item[1]))
    _, _, fact, relation = matches[0]
    return fact, relation


def attribute_similarity(left: object, right: object) -> float:
    """Compare attribute labels with exact, segment, and token evidence."""
    left_text = normalize_text(left)
    right_text = normalize_text(right)
    if not left_text or not right_text:
        return 0.0
    if left_text == right_text:
        return 1.0
    left_identifier = normalize_identifier(left_text)
    right_identifier = normalize_identifier(right_text)
    if left_identifier == right_identifier:
        return 1.0
    left_tokens = _attribute_tokens(left_text)
    right_tokens = _attribute_tokens(right_text)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens & right_tokens
    if not overlap:
        ratio = SequenceMatcher(None, left_identifier, right_identifier).ratio()
        return ratio if ratio >= 0.88 else 0.0
    containment = len(overlap) / min(len(left_tokens), len(right_tokens))
    union = len(overlap) / len(left_tokens | right_tokens)
    return max(containment, union)


def value_similarity(left: object, right: object) -> float:
    """Compare values without making semantic equivalence assumptions."""
    left_text = normalize_text(left)
    right_text = normalize_text(right)
    if not left_text or not right_text:
        return 0.0
    if left_text == right_text:
        return 1.0
    left_tokens = set(tokenize_search_text(left_text))
    right_tokens = set(tokenize_search_text(right_text))
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens & right_tokens
    if not overlap:
        return 0.0
    return len(overlap) / min(len(left_tokens), len(right_tokens))


def _attribute_tokens(text: str) -> set[str]:
    # Keep dotted/underscored segments as well as their searchable forms.
    raw = set(tokenize_search_text(text))
    raw.update(part for part in re.split(r"[._\-/\s]+", text) if part)
    return {token for token in raw if len(token) > 1 or "\u4e00" <= token <= "\u9fff"}


def _entity_scope(fact: FactCard) -> tuple[str, str]:
    entity = str(fact.entity or "").strip()
    scope = str(fact.scope or "").strip()
    if "::" in entity:
        base_entity, embedded_scope = entity.split("::", 1)
        if not scope:
            entity, scope = base_entity, embedded_scope
        elif normalize_identifier(scope) == normalize_identifier(embedded_scope):
            entity = base_entity
    return entity, scope


def _eligible(fact: FactCard) -> bool:
    if "preference_observation" in fact.tags:
        # Observations are intentionally kept per session so the independent
        # evidence promotion stage can count them before consolidation.
        return False
    intent = normalize_text(fact.intent)
    certainty = normalize_text(fact.certainty)
    temporal = normalize_text(fact.temporal_status)
    if intent in _NON_ALIGNING_INTENTS or certainty in {"uncertain", "reported"}:
        return False
    return not (
        temporal in {"temporary", "reverted"} and intent not in {"update", "revert"}
    )
