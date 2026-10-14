"""Mem0 Recall layer: Core-first, Archival-fallback retrieval strategy.

This mimics Mem0's Recall Memory — it's not a storage tier but a
*retrieval strategy* that decides when to stop searching Core and
when to fall back to deeper Archival search.
"""

from __future__ import annotations

from time import perf_counter

from os_agent.memory.fact_vector_index import FactSemanticHit, FactSemanticIndex
from os_agent.memory.mem0_archival import ArchivalMemoryStore
from os_agent.memory.mem0_core import CoreMemoryStore
from os_agent.memory.mem0_schemas import FactCard, RecallResult
from os_agent.memory.memory_config import MemoryConfig
from os_agent.memory.session_vector_index import SessionSemanticIndex


class RecallStrategy:
    """Core-first recall with cascading fallback to Archival.

    Strategy:
      1. Search Core (structured, fast, small).
      2. If Core returns < min_core_results, search Archival for archived facts.
      3. If still insufficient, search raw event logs.
      4. Merge and deduplicate results.
    """

    def __init__(
        self,
        core: CoreMemoryStore,
        archival: ArchivalMemoryStore,
        min_core_results: int | None = None,
        max_total: int | None = None,
        config: MemoryConfig | None = None,
        semantic_index: FactSemanticIndex | None = None,
        session_semantic_index: SessionSemanticIndex | None = None,
    ) -> None:
        self.config = config or MemoryConfig.global_config()
        self.core = core
        self.archival = archival
        self.min_core_results = (
            min_core_results
            if min_core_results is not None
            else int(self.config.recall_min_core_results)
        )
        self.max_total = (
            max_total if max_total is not None else int(self.config.recall_max_total)
        )
        self.semantic_index = semantic_index
        self.session_semantic_index = session_semantic_index

    def recall(
        self,
        query: str,
        entity_filter: str | None = None,
        tag_filter: list[str] | None = None,
        attribute_filter: str | None = None,
        scope_filter: str | None = None,
        days_filter: int | None = None,
        min_confidence: float | None = None,
        min_importance: float | None = None,
        max_total: int | None = None,
        query_embedding: list[float] | None = None,
        prepared_core_results: list[FactCard] | None = None,
        prepared_semantic_hits: list[FactSemanticHit] | None = None,
    ) -> RecallResult:
        """Run the full recall strategy for a query.

        Returns a RecallResult with facts from all layers, annotated
        with which layer(s) contributed.

        If query_embedding is provided by the caller, it will be used directly
        for semantic search instead of computing embeddings locally.
        """
        t0 = perf_counter()
        layers: list[str] = []
        rankings: list[list[FactCard]] = []
        result_limit = max_total if max_total is not None else self.max_total
        result_limit = max(int(result_limit), 0)
        if result_limit == 0:
            return RecallResult(
                facts=[],
                source_layer="none",
                query=query,
                latency_ms=round((perf_counter() - t0) * 1000, 2),
            )

        # ---- Layer 1: Core ----
        core_results = prepared_core_results
        if core_results is None:
            core_results = self.core.search(
                query,
                top_k=result_limit,
                entity_filter=entity_filter,
                tag_filter=tag_filter,
                exclude_tags=(
                    None
                    if tag_filter and "preference_observation" in tag_filter
                    else ["preference_observation"]
                ),
                attribute_filter=attribute_filter,
                scope_filter=scope_filter,
                days_filter=days_filter,
                min_confidence=min_confidence,
                min_importance=min_importance,
            )
        if self.semantic_index is not None and self.semantic_index.enabled:
            if prepared_semantic_hits is not None:
                semantic_hits = prepared_semantic_hits
            else:
                semantic_candidates = self.core.filter_active_facts(
                    entity_filter=entity_filter,
                    tag_filter=tag_filter,
                    exclude_tags=(
                        None
                        if tag_filter and "preference_observation" in tag_filter
                        else ["preference_observation"]
                    ),
                    attribute_filter=attribute_filter,
                    scope_filter=scope_filter,
                    days_filter=days_filter,
                    min_confidence=min_confidence,
                    min_importance=min_importance,
                )
                if query_embedding is None:
                    semantic_hits = self.semantic_index.search(
                        query,
                        semantic_candidates,
                        result_limit,
                    )
                else:
                    semantic_hits = self.semantic_index.search(
                        query,
                        semantic_candidates,
                        result_limit,
                        query_embedding=query_embedding,
                    )
            # Two-stage filter. The absolute floor removes hits that are simply
            # unrelated; the relative floor removes hits that are only similar
            # because every fact in the scope shares a domain vocabulary. A
            # fixed absolute threshold cannot do both: embeddings put unrelated
            # same-domain facts (React vs Kubernetes) around 0.73, barely below
            # a genuine alias match (k8s vs Kubernetes) at 0.80.
            absolute_floor = float(self.config.core_semantic_min_similarity)
            relative_floor = (
                max(hit.similarity for hit in semantic_hits)
                * float(self.config.core_semantic_relative_cutoff)
                if semantic_hits
                else 0.0
            )
            filtered_semantic = [
                hit.fact
                for hit in semantic_hits
                if hit.similarity >= absolute_floor
                and hit.similarity >= relative_floor
            ]
            core_results = self._rrf_merge(
                core_results,
                filtered_semantic,
                result_limit,
            )
        core_count = len(core_results)
        if core_count > 0:
            layers.append("core")
            rankings.append(core_results)

        # ---- Layer 2: Archival facts ----
        if core_count < self.min_core_results:
            archival_facts = self.archival.search_facts(
                query,
                top_k=result_limit,
                entity_filter=entity_filter,
                tag_filter=tag_filter,
                attribute_filter=attribute_filter,
                scope_filter=scope_filter,
                days_filter=days_filter,
            )
            if archival_facts:
                layers.append("archival")
                rankings.append(archival_facts)

        # ---- Layer 3: Raw events (complete-history candidates) ----
        # Raw retrieval runs alongside Core because several weak Core hits do
        # not prove that the relevant implicit event was structured correctly.
        raw_allowed = not any(
            value is not None
            for value in (
                entity_filter,
                tag_filter,
                attribute_filter,
                scope_filter,
                days_filter,
                min_confidence,
                min_importance,
            )
        )
        if raw_allowed:
            if (
                self.session_semantic_index is not None
                and self.session_semantic_index.enabled
            ):
                if query_embedding is None:
                    session_hits = self.session_semantic_index.search(
                        query,
                        self.archival.session_documents(),
                        result_limit,
                    )
                else:
                    session_hits = self.session_semantic_index.search(
                        query,
                        self.archival.session_documents(),
                        result_limit,
                        query_embedding=query_embedding,
                    )
                session_facts: list[FactCard] = []
                for hit in session_hits:
                    # Filter low-quality semantic hits to prevent diluting RRF
                    if hit.similarity < 0.65:
                        continue
                    document = hit.document
                    fact = FactCard(
                        entity="session",
                        attribute=document.task_id or document.session_id,
                        value=document.text,
                        evidence=document.text[:500],
                        source_session_id=document.session_id,
                        source_event_id=f"session:{document.session_id}",
                        confidence=0.35,
                        fact_id=self.archival.session_memory_id(document.session_id),
                        tier="archival",
                        importance=0.25,
                        tags=["semantic_session", "session_summary"],
                    )
                    fact.semantic_score = hit.similarity
                    session_facts.append(fact)
                if session_facts:
                    layers.append("semantic_sessions")
                    rankings.append(session_facts)

            raw_events = self.archival.search_raw_events(
                query,
                top_k=result_limit,
            )
            raw_facts: list[FactCard] = []
            for raw in raw_events:
                event = raw["event"]
                fact = FactCard(
                    entity="event",
                    attribute=str(
                        event.get("task_id") or event.get("event_type", "unknown")
                    ),
                    value=json_safe_str(event.get("payload", {})),
                    evidence=json_safe_str(event),
                    source_session_id=event.get("session_id", ""),
                    source_event_id=event.get("event_id", ""),
                    fact_id=self.archival.event_memory_id(
                        self.archival.event_identifier(event)
                    ),
                    confidence=float(self.config.recall_raw_event_default_confidence),
                    tier="archival",
                    importance=float(self.config.recall_raw_event_default_importance),
                    tags=["raw_event", str(event.get("event_type", "unknown"))],
                )
                fact.search_score = float(raw.get("score", 0.0))
                raw_facts.append(fact)
            if raw_facts:
                layers.append("raw_events")
                rankings.append(raw_facts)

        all_facts = self._rrf_merge_rankings(rankings, result_limit)
        all_facts = self._drop_superseded_archival_versions(all_facts)

        # mark accessed
        for fact in all_facts:
            fact.mark_accessed()

        latency = (perf_counter() - t0) * 1000
        core_ids = {fact.fact_id for fact in core_results}
        total_core = sum(fact.fact_id in core_ids for fact in all_facts)
        total_archival = len(all_facts) - total_core

        return RecallResult(
            facts=all_facts,
            source_layer="+".join(layers) if layers else "none",
            query=query,
            total_core_hits=total_core,
            total_archival_hits=total_archival,
            latency_ms=round(latency, 2),
        )

    @staticmethod
    def _rrf_merge(
        lexical: list[FactCard],
        semantic: list[FactCard],
        limit: int,
        rrf_k: int = 60,
    ) -> list[FactCard]:
        scores: dict[str, float] = {}
        facts: dict[str, FactCard] = {}
        # Weighted RRF: lexical gets higher weight for Core facts
        for ranking, weight in [(lexical, 1.0), (semantic, 0.75)]:
            for rank, fact in enumerate(ranking, 1):
                facts[fact.fact_id] = fact
                scores[fact.fact_id] = scores.get(fact.fact_id, 0.0) + weight / (
                    rrf_k + rank
                )
        ordered = sorted(
            facts.values(),
            key=lambda fact: (-scores[fact.fact_id], fact.entity, fact.attribute),
        )
        for fact in ordered:
            fact.search_score = round(scores[fact.fact_id], 6)
        return ordered[:limit]

    @staticmethod
    def _rrf_merge_rankings(
        rankings: list[list[FactCard]],
        limit: int,
        rrf_k: int = 60,
    ) -> list[FactCard]:
        scores: dict[str, float] = {}
        facts: dict[str, FactCard] = {}
        for ranking in rankings:
            # Determine weight based on source layer
            first_fact = ranking[0] if ranking else None
            if first_fact:
                if first_fact.tier == "core":
                    weight = 1.0
                elif "semantic_session" in first_fact.tags:
                    weight = 0.75
                else:
                    weight = 0.75
            else:
                weight = 1.0

            for rank, fact in enumerate(ranking, 1):
                key = RecallStrategy._rrf_key(fact)
                if key not in facts or facts[key].tier != "core":
                    facts[key] = fact
                scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)
        ordered_keys = sorted(
            facts,
            key=lambda key: (
                -scores[key],
                RecallStrategy._source_priority(facts[key]),
                facts[key].entity,
                facts[key].attribute,
            ),
        )
        ordered = [facts[key] for key in ordered_keys[:limit]]
        for fact in ordered:
            key = RecallStrategy._rrf_key(fact)
            fact.search_score = round(scores[key], 6)
        return ordered

    @classmethod
    def merge_query_rankings(
        cls,
        rankings: list[list[FactCard]],
        coverage_rankings: list[list[FactCard]],
        limit: int,
    ) -> list[FactCard]:
        """Fuse retrieval routes while reserving evidence for each subquery."""
        if limit <= 0 or not rankings:
            return []
        all_count = sum(len(ranking) for ranking in rankings)
        fused = cls._rrf_merge_rankings(rankings, max(all_count, limit))
        selected: list[FactCard] = []
        selected_keys: set[str] = set()
        for ranking in coverage_rankings:
            for fact in ranking:
                key = cls._rrf_key(fact)
                if key not in selected_keys:
                    selected.append(fact)
                    selected_keys.add(key)
                    break
            if len(selected) == limit:
                return selected
        for fact in fused:
            key = cls._rrf_key(fact)
            if key in selected_keys:
                continue
            selected.append(fact)
            selected_keys.add(key)
            if len(selected) == limit:
                break
        return selected

    @staticmethod
    def _source_priority(fact: FactCard) -> int:
        """Prefer richer representations when RRF scores are exactly tied."""
        if fact.tier == "core":
            return 0
        if "semantic_session" in fact.tags:
            return 1
        return 2

    @staticmethod
    def _rrf_key(fact: FactCard) -> str:
        session_scoped = (
            "preference_observation" in fact.tags or "semantic_session" in fact.tags
        )
        if session_scoped and fact.source_session_id:
            return f"session:{fact.source_session_id}"
        # One user event can legitimately yield several independent Core facts.
        # Deduplicating Core by source_event_id silently collapses multi-fact
        # recall; lexical and semantic representations already share fact_id.
        if fact.tier == "core":
            return fact.fact_id
        return fact.source_event_id or fact.fact_id

    def _drop_superseded_archival_versions(
        self, facts: list[FactCard]
    ) -> list[FactCard]:
        if not any(
            fact.tier == "archival"
            and fact.status in {"superseded", "deleted"}
            for fact in facts
        ):
            return facts
        active_core_keys = {fact.key for fact in self.core.active_facts()}
        filtered: list[FactCard] = []
        for fact in facts:
            if (
                fact.tier == "archival"
                and fact.status in {"superseded", "deleted"}
                and fact.key in active_core_keys
            ):
                continue
            filtered.append(fact)
        return filtered


def json_safe_str(obj: object) -> str:
    """Convert an object to a compact string for FactCard values."""
    import json

    if isinstance(obj, str):
        return obj[:300]
    try:
        return json.dumps(obj, ensure_ascii=False)[:300]
    except (TypeError, ValueError):
        return str(obj)[:300]
