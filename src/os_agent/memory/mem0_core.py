"""Mem0 Core layer: structured fact store with versioning and dedup.

Core Memory holds the "active working set" of facts — recently accessed,
high-confidence, currently valid information.  When a fact is updated,
the old version is superseded and moved to Archival.

This is analogous to Mem0's Core Memory: small, fast, structured.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from heapq import nsmallest
from math import log
from pathlib import Path

from os_agent.memory.mem0_schemas import FactCard, _utc_now_iso
from os_agent.memory.memory_config import MemoryConfig
from os_agent.memory.retrieval_text import (
    expand_query_terms,
    term_weight,
    tokenize_search_text,
)


class CoreMemoryStore:
    """Structured, versioned fact store with JSON backing file.

    Key operations:
      - get(entity, attribute) → FactCard | None
      - set(entity, attribute, value, ...) → FactCard (auto-bump version)
      - search(query) → list[FactCard]
      - list_by_entity(entity) → list[FactCard]
      - remove(entity, attribute) → marks as deleted
      - snapshot() → serialize all active facts

    The backing file ``memory/core_facts.json`` is a JSON list of fact
    dictionaries.  On load, only "active" facts are kept in memory;
    superseded/deleted facts are held in the file for history but not
    searched by default.
    """

    def __init__(
        self, root: Path | str = "memory", config: MemoryConfig | None = None
    ) -> None:
        self.root = Path(root)
        self.config = config or MemoryConfig.global_config()
        self.root.mkdir(parents=True, exist_ok=True)
        self._path = self.root / "core_facts.json"
        self._facts: dict[tuple[str, str], FactCard] = {}
        self._entity_index: dict[str, set[tuple[str, str]]] = {}
        self._fact_token_cache: dict[str, tuple[str, frozenset[str]]] = {}
        self._lexical_index_dirty = True
        self._lexical_postings: dict[str, frozenset[str]] = {}
        self._active_facts_by_id: dict[str, FactCard] = {}
        self._active_fact_order: dict[str, int] = {}
        self._load()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def get(self, entity: str, attribute: str) -> FactCard | None:
        """Return the active FactCard for (entity, attribute), or None."""
        key = (entity, attribute)
        fact = self._facts.get(key)
        if fact and fact.status == "active":
            fact.mark_accessed()
            self._save()
            return fact
        return None

    def set(
        self,
        entity: str,
        attribute: str,
        value: object,
        *,
        evidence: str = "",
        source_session_id: str = "",
        source_event_id: str = "",
        confidence: float | None = None,
        tags: list[str] | None = None,
        importance: float | None = None,
        scope: str = "",
        intent: str = "",
        certainty: str = "certain",
        temporal_status: str = "stable",
        custom_metadata: dict | None = None,
    ) -> FactCard:
        """Create or update a fact.  If it already exists and the value
        differs, the old version is superseded (archived) and the new one
        becomes active.  If the value is identical, confidence is boosted."""
        key = (entity, attribute)
        existing = self._facts.get(key)
        if confidence is None:
            confidence = float(self.config.core_default_confidence)
        if importance is None:
            importance = float(self.config.core_default_importance)

        if existing and existing.status == "active":
            if existing.value == value:
                # reinforce — bump confidence and access
                self._fact_token_cache.pop(existing.fact_id, None)
                existing.confidence = min(
                    existing.confidence + float(self.config.core_confidence_boost),
                    1.0,
                )
                existing.updated_at = _utc_now_iso()
                existing.mark_accessed()
                self._save()
                return existing

            # value changed → supersede old, create new
            # CRITICAL FIX: Mark the old fact as superseded BEFORE replacing it
            old_fact_id = existing.fact_id
            self._fact_token_cache.pop(old_fact_id, None)
            existing.status = "superseded"
            existing.temporal_status = "superseded"
            existing.tier = "archival"

            new = FactCard(
                entity=entity,
                attribute=attribute,
                value=value,
                evidence=evidence,
                source_session_id=source_session_id,
                source_event_id=source_event_id,
                confidence=confidence,
                version=existing.version + 1,
                tags=tags or [],
                importance=importance,
                scope=scope,
                intent=intent,
                certainty=certainty,
                temporal_status=temporal_status,
                supersedes=old_fact_id,
                custom_metadata=custom_metadata or {},
            )

            # Link the supersession relationship both ways
            existing.superseded_by = new.fact_id

            # Replace in memory (existing is now superseded)
            self._facts[key] = new
            # keep superseded in entity index for archival queries
            self._entity_index.setdefault(entity, set()).add(key)
            self._invalidate_lexical_index()
            self._save()
            return new

        # new fact
        new = FactCard(
            entity=entity,
            attribute=attribute,
            value=value,
            evidence=evidence,
            source_session_id=source_session_id,
            source_event_id=source_event_id,
            confidence=confidence,
            tags=tags or [],
            importance=importance,
            scope=scope,
            intent=intent,
            certainty=certainty,
            temporal_status=temporal_status,
            custom_metadata=custom_metadata or {},
        )
        self._facts[key] = new
        self._entity_index.setdefault(entity, set()).add(key)
        self._invalidate_lexical_index()
        self._save()
        return new

    def remove(self, entity: str, attribute: str) -> FactCard | None:
        """Soft-delete — mark as 'deleted' rather than physically remove."""
        key = (entity, attribute)
        fact = self._facts.get(key)
        if fact and fact.status == "active":
            self._fact_token_cache.pop(fact.fact_id, None)
            fact.status = "deleted"
            fact.updated_at = _utc_now_iso()
            self._invalidate_lexical_index()
            self._save()
            return fact
        return None

    def hard_remove(self, entity: str, attribute: str) -> FactCard | None:
        """Physically remove a fact from Core, including a soft-deleted row."""
        key = (entity, attribute)
        fact = self._facts.pop(key, None)
        if fact is None:
            return None
        self._fact_token_cache.pop(fact.fact_id, None)
        self._invalidate_lexical_index()
        entity_keys = self._entity_index.get(entity)
        if entity_keys is not None:
            entity_keys.discard(key)
            if not entity_keys:
                self._entity_index.pop(entity, None)
        self._save()
        return fact

    def find_by_id(self, fact_id: str) -> FactCard | None:
        """Return a Core fact by physical identifier, regardless of status."""
        return next(
            (fact for fact in self._facts.values() if fact.fact_id == fact_id), None
        )

    def list_by_entity(
        self, entity: str, include_inactive: bool = False
    ) -> list[FactCard]:
        """Return all facts for an entity, newest version first."""
        keys = self._entity_index.get(entity, set())
        facts = []
        for key in keys:
            fact = self._facts.get(key)
            if fact and (include_inactive or fact.status == "active"):
                facts.append(fact)
        facts.sort(key=lambda f: (-f.version, -f.confidence))
        return facts

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 0,
        entity_filter: str | None = None,
        tag_filter: list[str] | None = None,
        exclude_tags: list[str] | None = None,
        attribute_filter: str | None = None,
        scope_filter: str | None = None,
        days_filter: int | None = None,
        min_confidence: float | None = None,
        min_importance: float | None = None,
        min_score: float = 0.0,
    ) -> list[FactCard]:
        """Structured search over active Core facts.

        Scoring: keyword overlap between query and (entity + attribute +
        value + tags) × confidence × importance × recency.
        """
        return self.search_many(
            [query],
            top_k=top_k,
            entity_filter=entity_filter,
            tag_filter=tag_filter,
            exclude_tags=exclude_tags,
            attribute_filter=attribute_filter,
            scope_filter=scope_filter,
            days_filter=days_filter,
            min_confidence=min_confidence,
            min_importance=min_importance,
            min_score=min_score,
        )[0]

    def search_many(
        self,
        queries: list[str] | tuple[str, ...],
        top_k: int = 0,
        entity_filter: str | None = None,
        tag_filter: list[str] | None = None,
        exclude_tags: list[str] | None = None,
        attribute_filter: str | None = None,
        scope_filter: str | None = None,
        days_filter: int | None = None,
        min_confidence: float | None = None,
        min_importance: float | None = None,
        min_score: float = 0.0,
    ) -> list[list[FactCard]]:
        """Search multiple query routes over one filtered Core snapshot."""
        if not queries:
            return []
        if top_k <= 0:
            top_k = int(self.config.core_search_top_k)
        facts_to_search = self.filter_active_facts(
            entity_filter=entity_filter,
            tag_filter=tag_filter,
            exclude_tags=exclude_tags,
            attribute_filter=attribute_filter,
            scope_filter=scope_filter,
            days_filter=days_filter,
            min_confidence=min_confidence,
            min_importance=min_importance,
        )
        recency_window = max(float(self.config.core_recency_window_days), 0.001)
        recency_floor = float(self.config.core_recency_floor)
        kw_weight = float(self.config.core_composite_keyword_weight)
        confidence_weight = float(self.config.core_composite_confidence_weight)
        importance_weight = float(self.config.core_composite_importance_weight)
        recency_weight = float(self.config.core_composite_recency_weight)

        # Precompute per-fact token sets once, then derive IDF over the candidate
        # corpus. Without IDF a low-information term ("deploy", "service") carries
        # the same weight as a discriminative one ("kubernetes"), so a query that
        # mentions both matches nearly every fact in the scope.
        self._ensure_lexical_index()
        allowed_ids = {fact.fact_id for fact in facts_to_search}
        base_scores = {
            fact.fact_id: (
                fact.confidence * confidence_weight
                + fact.importance * importance_weight
                + max(
                    recency_floor,
                    1.0 - self._days_since(fact.updated_at) / recency_window,
                )
                * recency_weight
            )
            for fact in facts_to_search
        }
        full_index = len(allowed_ids) == len(self._active_facts_by_id)
        results: list[list[FactCard]] = []
        for query in queries:
            terms = expand_query_terms(query)
            metadata_only = query.strip() == ""

            if metadata_only:
                scored = [
                    (base_scores[fact.fact_id], fact) for fact in facts_to_search
                ]
            else:
                idf = self._idf_for_indexed_terms(terms, allowed_ids)
                keyword_scores: dict[str, float] = {}
                for term in terms:
                    contribution = self._term_weight(term) * idf[term]
                    postings = self._lexical_postings.get(term, frozenset())
                    matching_ids = postings if full_index else postings & allowed_ids
                    for fact_id in matching_ids:
                        keyword_scores[fact_id] = (
                            keyword_scores.get(fact_id, 0.0) + contribution
                        )
                scored = []
                for fact_id, keyword_score in keyword_scores.items():
                    if keyword_score <= 0:
                        continue
                    fact = self._active_facts_by_id[fact_id]
                    scored.append(
                        (
                            keyword_score * kw_weight + base_scores[fact_id],
                            fact,
                        )
                    )
            if min_score > 0:
                scored = [
                    (score, fact) for score, fact in scored if score >= min_score
                ]
            ranking_key = lambda item: (
                -item[0],
                self._active_fact_order[item[1].fact_id],
            )
            if len(scored) > top_k:
                scored = nsmallest(top_k, scored, key=ranking_key)
            else:
                scored.sort(key=ranking_key)
            if scored and not metadata_only:
                ratio = float(self.config.core_search_relative_cutoff)
                if ratio > 0:
                    floor = scored[0][0] * ratio
                    scored = [
                        (score, fact)
                        for score, fact in scored
                        if score >= floor
                    ]
            ranked: list[FactCard] = []
            for score, fact in scored:
                fact.search_score = round(score, 4)
                ranked.append(fact)
            results.append(ranked)
        return results

    def filter_active_facts(
        self,
        *,
        entity_filter: str | None = None,
        tag_filter: list[str] | None = None,
        exclude_tags: list[str] | None = None,
        attribute_filter: str | None = None,
        scope_filter: str | None = None,
        days_filter: int | None = None,
        min_confidence: float | None = None,
        min_importance: float | None = None,
    ) -> list[FactCard]:
        """Apply metadata filters before lexical or semantic candidate ranking."""
        # Filter out superseded facts: check both status field and superseded_by field
        facts = [
            fact
            for fact in self._facts.values()
            if fact.status == "active"
            and fact.temporal_status != "superseded"
            and fact.superseded_by is None  # Critical: exclude facts that have been superseded
        ]
        if entity_filter:
            facts = [fact for fact in facts if fact.entity == entity_filter]
        if tag_filter:
            facts = [
                fact for fact in facts if all(tag in fact.tags for tag in tag_filter)
            ]
        if exclude_tags:
            facts = [
                fact
                for fact in facts
                if not any(tag in fact.tags for tag in exclude_tags)
            ]
        if attribute_filter:
            facts = [fact for fact in facts if fact.attribute == attribute_filter]
        if scope_filter:
            # Fix: properly filter by scope field, not by entity naming convention
            facts = [fact for fact in facts if fact.scope == scope_filter]
        if days_filter is not None:
            facts = [
                fact
                for fact in facts
                if self._days_since(fact.updated_at) <= days_filter
            ]
        if min_confidence is not None:
            facts = [fact for fact in facts if fact.confidence >= min_confidence]
        if min_importance is not None:
            facts = [fact for fact in facts if fact.importance >= min_importance]
        return facts

    def snapshot(self) -> dict:
        """Export all active facts for inspection."""
        active = []
        for fact in self._facts.values():
            if fact.status == "active":
                active.append(fact.to_dict())
        return {
            "exported_at": _utc_now_iso(),
            "total_active": len(active),
            "facts": sorted(active, key=lambda f: (f["entity"], f["attribute"])),
        }

    def active_facts(self) -> list[FactCard]:
        """Return active facts without changing access metadata."""
        return [fact for fact in self._facts.values() if fact.status == "active"]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                return
        except (json.JSONDecodeError, OSError):
            return

        for item in raw:
            fact = FactCard.from_dict(item)
            self._facts[fact.key] = fact
            self._entity_index.setdefault(fact.entity, set()).add(fact.key)

    def _save(self) -> None:
        payload = [f.to_dict() for f in self._facts.values()]
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return tokenize_search_text(text)

    def _search_terms_for_fact(self, fact: FactCard) -> frozenset[str]:
        """Return immutable fact terms without re-tokenizing on every query."""
        cached = self._fact_token_cache.get(fact.fact_id)
        if cached is not None and cached[0] == fact.updated_at:
            return cached[1]
        text = (
            f"{fact.entity} {fact.scope} {fact.attribute} {fact.value} "
            f"{' '.join(fact.tags)} {fact.evidence}"
        )
        terms = frozenset(self._tokenize(text))
        self._fact_token_cache[fact.fact_id] = (fact.updated_at, terms)
        return terms

    def _invalidate_lexical_index(self) -> None:
        self._lexical_index_dirty = True

    def _ensure_lexical_index(self) -> None:
        if not self._lexical_index_dirty:
            return
        postings: dict[str, set[str]] = {}
        active_by_id: dict[str, FactCard] = {}
        for fact in self._facts.values():
            if (
                fact.status != "active"
                or fact.temporal_status == "superseded"
                or fact.superseded_by is not None
            ):
                continue
            active_by_id[fact.fact_id] = fact
            for term in self._search_terms_for_fact(fact):
                postings.setdefault(term, set()).add(fact.fact_id)
        self._lexical_postings = {
            term: frozenset(fact_ids) for term, fact_ids in postings.items()
        }
        self._active_facts_by_id = active_by_id
        self._active_fact_order = {
            fact_id: index for index, fact_id in enumerate(active_by_id)
        }
        self._lexical_index_dirty = False

    def _idf_for_indexed_terms(
        self, terms: list[str], allowed_ids: set[str]
    ) -> dict[str, float]:
        total = len(allowed_ids)
        if total == 0:
            return dict.fromkeys(terms, 1.0)
        full_index = len(allowed_ids) == len(self._active_facts_by_id)
        hits = {
            term: (
                len(self._lexical_postings.get(term, ()))
                if full_index
                else len(self._lexical_postings.get(term, frozenset()) & allowed_ids)
            )
            for term in terms
        }
        return self._idf_from_hit_counts(terms, total, hits)

    @staticmethod
    def _term_weight(term: str) -> float:
        return term_weight(term)

    @staticmethod
    def _idf_for_terms(
        terms: list[str],
        corpus: list[frozenset[str]],
    ) -> dict[str, float]:
        """Inverse document frequency for each query term over the candidates.

        A term present in every candidate carries no discriminative signal and
        collapses to ~0; a term unique to one candidate keeps close to full
        weight. Returns 1.0 for every term when the corpus is empty so that
        scoring degrades to plain term weighting.
        """
        total = len(corpus)
        if total == 0:
            return dict.fromkeys(terms, 1.0)

        hits = {
            term: sum(1 for tokens in corpus if term in tokens) for term in terms
        }
        return CoreMemoryStore._idf_from_hit_counts(terms, total, hits)

    @staticmethod
    def _idf_from_hit_counts(
        terms: list[str], total: int, hit_counts: dict[str, int]
    ) -> dict[str, float]:
        idf: dict[str, float] = {}
        for term in terms:
            hits = hit_counts.get(term, 0)
            if hits == 0:
                idf[term] = 1.0
                continue
            # log((N + 1) / (hits + 0.5)) normalized against the best possible
            # value, so scores stay in a stable 0..1 band as the corpus grows.
            raw = log((total + 1.0) / (hits + 0.5))
            best = log((total + 1.0) / 1.5)
            idf[term] = max(raw / best, 0.0) if best > 0 else 1.0
        return idf

    @staticmethod
    def _days_since(iso_timestamp: str) -> float:
        if not iso_timestamp:
            return 999.0
        try:
            dt = datetime.fromisoformat(iso_timestamp)
            return max((datetime.now(UTC) - dt).total_seconds() / 86400.0, 0.0)
        except (ValueError, TypeError):
            return 999.0
