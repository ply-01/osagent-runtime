"""Unified memory module facade for agents and tool use."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from os_agent.memory.agentic_policy import MemoryActionTraceStore, MemoryDistiller
from os_agent.memory.embedding_cache import SqliteEmbeddingCache
from os_agent.memory.embedding_provider import (
    EmbeddingProvider,
    create_embedding_provider,
)
from os_agent.memory.fact_vector_index import FactSemanticIndex
from os_agent.memory.hard_delete import HardDeleteService
from os_agent.memory.interface import (
    ForgetRequest,
    MemoryActionRequest,
    MemoryOperationResult,
    MemoryRecord,
    RecallRequest,
    RememberRequest,
    UpdateRequest,
)
from os_agent.memory.markdown_store import MarkdownMemoryStore
from os_agent.memory.mem0_archival import ArchivalMemoryStore
from os_agent.memory.mem0_core import CoreMemoryStore
from os_agent.memory.mem0_recall import RecallStrategy
from os_agent.memory.mem0_schemas import FactCard
from os_agent.memory.memory_config import MemoryConfig
from os_agent.memory.memory_frame import MemoryFrame, MemoryFrameParser
from os_agent.memory.query_rewriter import QueryRewriter
from os_agent.memory.semantic_alignment import find_aligned_fact
from os_agent.memory.session_vector_index import SessionSemanticIndex
from os_agent.memory.short_term import ShortTermMemoryStore


class AgentMemoryModule:
    """Stable facade over LTM and STM implementations.

    Public operations are intentionally small and tool-friendly:
    remember, forget, update, and recall. LTM uses the Mem0-style
    Core/Archival/Recall stores; STM uses a small transient JSON store.
    """

    def __init__(
        self,
        root: Path | str = "memory",
        *,
        core: CoreMemoryStore | None = None,
        archival: ArchivalMemoryStore | None = None,
        stm: ShortTermMemoryStore | None = None,
        markdown_store: MarkdownMemoryStore | None = None,
        config: MemoryConfig | None = None,
        distiller: MemoryDistiller | None = None,
        trace_store: MemoryActionTraceStore | None = None,
        query_rewriter: QueryRewriter | None = None,
        hard_delete_service: HardDeleteService | None = None,
        semantic_index: FactSemanticIndex | None = None,
        session_semantic_index: SessionSemanticIndex | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.root = Path(root)
        self.config = config or MemoryConfig.global_config()
        self.core = core or CoreMemoryStore(self.root, config=self.config)
        self.archival = archival or ArchivalMemoryStore(self.root, config=self.config)
        self.stm = stm or ShortTermMemoryStore(self.root)
        self.markdown_store = markdown_store or MarkdownMemoryStore(self.root)
        semantic_cache = SqliteEmbeddingCache(
            self.root.parent / "data" / "vector-cache" / "memory-documents.sqlite3"
        )
        provider = embedding_provider
        if provider is None and semantic_index is not None:
            provider = getattr(semantic_index, "provider", None)
        provider = provider or create_embedding_provider()
        self.semantic_index = semantic_index or FactSemanticIndex(
            provider=provider,
            cache=semantic_cache,
        )
        self.session_semantic_index = session_semantic_index or SessionSemanticIndex(
            provider=provider,
            cache=semantic_cache,
        )
        self.recall_strategy = RecallStrategy(
            self.core,
            self.archival,
            config=self.config,
            semantic_index=self.semantic_index,
            session_semantic_index=self.session_semantic_index,
        )
        self.distiller = distiller or MemoryDistiller()
        self.query_rewriter = query_rewriter or QueryRewriter()
        self.frame_parser = MemoryFrameParser()
        self.hard_delete_service = hard_delete_service or HardDeleteService(self.root)
        self.trace_store = trace_store
        if self.trace_store is None and bool(self.config.agentic_policy_trace_enabled):
            trace_name = str(self.config.agentic_policy_trace_file or "policy-traces.jsonl")
            self.trace_store = MemoryActionTraceStore(self.root / trace_name)

    def execute_action(self, request: MemoryActionRequest) -> MemoryOperationResult:
        """Execute one unified memory action and optionally record a policy trace."""
        action = request.action
        executed_request = request
        if action in {"remember", "ltm.add"}:
            scope = self._mutation_scope(
                action,
                request.scope,
                forced_scope="ltm" if action == "ltm.add" else None,
            )
            executed_request = replace(request, scope=scope)  # type: ignore[arg-type]
            result = self._remember(self._remember_request(request, scope=scope))
            canonical = "ltm.add" if scope == "ltm" else "stm.add"
        elif action in {"update", "ltm.update"}:
            scope = self._mutation_scope(
                action,
                request.scope,
                forced_scope="ltm" if action == "ltm.update" else None,
            )
            executed_request = replace(request, scope=scope)  # type: ignore[arg-type]
            result = self._update(self._update_request(request, scope=scope))
            canonical = "ltm.update" if scope == "ltm" else "stm.update"
        elif action in {"forget", "ltm.delete"}:
            scope = self._mutation_scope(
                action,
                request.scope,
                forced_scope="ltm" if action == "ltm.delete" else None,
            )
            executed_request = replace(request, scope=scope)  # type: ignore[arg-type]
            result = self._forget(self._forget_request(request, scope=scope))
            canonical = "ltm.delete" if scope == "ltm" else "stm.delete"
        elif action in {"recall", "stm.retrieve"}:
            scope = "stm" if action == "stm.retrieve" else request.scope
            executed_request = replace(request, scope=scope)  # type: ignore[arg-type]
            result = self._recall(
                RecallRequest(
                    query=request.query,
                    scope=scope,
                    limit=request.limit,
                    entity=request.entity,
                    attribute=request.attribute,
                    fact_scope=request.fact_scope,
                    days=request.days,
                    query_variants=request.query_variants,
                    subqueries=request.subqueries,
                    query_embedding=request.query_embedding,
                )
            )
            if scope == "stm":
                canonical = "stm.retrieve"
            elif scope == "ltm":
                canonical = "ltm.retrieve"
            else:
                canonical = "recall"
        elif action in {"summary", "stm.summary"}:
            executed_request = replace(request, scope="stm")
            result = self._summarize_stm(request)
            canonical = "stm.summary"
        elif action in {"filter", "stm.filter"}:
            executed_request = replace(request, scope="stm")
            result = self._filter_stm(request)
            canonical = "stm.filter"
        elif action == "noop":
            result = MemoryOperationResult(
                ok=True,
                action="noop",
                scope=request.scope,
                message=request.reason or "memory policy chose no operation",
            )
            canonical = "noop"
        else:
            raise ValueError(f"unsupported memory action: {action}")
        self._trace_action(canonical, executed_request, result)
        return result

    def remember(self, request: RememberRequest) -> MemoryOperationResult:
        """Store a new LTM or STM record using the unified dispatcher."""
        return self.execute_action(self._action_from_remember(request))

    def _remember(self, request: RememberRequest) -> MemoryOperationResult:
        """Store a new LTM or STM record."""
        if request.scope == "stm":
            entry = self.stm.remember(
                request.content,
                memory_id=request.memory_id,
                tags=request.tags,
                metadata=request.metadata,
                ttl_seconds=request.ttl_seconds,
            )
            record = self.stm.get(entry.memory_id)
            return MemoryOperationResult(
                ok=True,
                action="remember",
                scope="stm",
                records=[record] if record else [],
                message="short-term memory stored",
            )

        entity, attribute, value, tags, metadata, frame = self._normalize_ltm_request(
            content=request.content,
            entity=request.entity,
            attribute=request.attribute,
            value=request.value,
            tags=request.tags,
            metadata=request.metadata,
        )
        if frame is not None and not frame.should_persist:
            return self._nonpersistent_ltm_result("remember", frame)

        # Infer temporal status from content if not explicitly provided.
        # Beliefs starting with "Previously" or containing "Replaced by" are
        # historical context rather than current state, so mark them superseded.
        temporal_status = str(metadata.get("temporal_status", "stable"))
        if temporal_status == "stable":
            content_lower = request.content.lower()
            if content_lower.startswith("previously ") or " replaced by " in content_lower:
                temporal_status = "superseded"

        incoming = FactCard(
            entity=entity,
            attribute=attribute,
            value=value,
            evidence=request.evidence or request.content,
            source_session_id=request.source_session_id,
            source_event_id=request.source_event_id,
            confidence=request.confidence,
            tags=tags,
            scope=str(metadata.get("scope", "")),
            intent=str(metadata.get("intent", "")),
            certainty=str(metadata.get("certainty", "certain")),
            temporal_status=temporal_status,
        )
        existing = self.core.get(entity, attribute)
        alignment = None
        if existing is None:
            aligned = find_aligned_fact(
                incoming,
                self.core.active_facts(),
                threshold=float(self.config.pipeline_semantic_alignment_threshold),
            )
            if aligned is not None:
                existing, alignment = aligned
        target_entity = existing.entity if existing is not None else entity
        target_attribute = existing.attribute if existing is not None else attribute
        fact = self.core.set(
            entity=target_entity,
            attribute=target_attribute,
            value=value,
            evidence=request.evidence or request.content,
            source_session_id=request.source_session_id,
            source_event_id=request.source_event_id,
            confidence=request.confidence,
            tags=tags,
            importance=float(request.metadata.get("importance", 0.5)),
            scope=str(metadata.get("scope", "")),
            intent=str(metadata.get("intent", "")),
            certainty=str(metadata.get("certainty", "certain")),
            temporal_status=temporal_status,
            custom_metadata=request.metadata,
        )
        if existing and existing.value != value and not (alignment and alignment.same_value):
            self.archival.archive_fact(existing.supersede(fact.fact_id))
        self._sync_markdown(fact)
        return MemoryOperationResult(
            ok=True,
            action="remember",
            scope="ltm",
            records=[self._fact_to_record(fact)],
            message="long-term memory stored",
        )

    def forget(self, request: ForgetRequest) -> MemoryOperationResult:
        """Remove or retire memory using the unified dispatcher."""
        return self.execute_action(self._action_from_forget(request))

    def _forget(self, request: ForgetRequest) -> MemoryOperationResult:
        """Remove STM entry or soft-delete LTM fact."""
        if request.scope == "stm":
            removed = self.stm.forget(memory_id=request.memory_id, query=request.query)
            return MemoryOperationResult(
                ok=bool(removed),
                action="forget",
                scope="stm",
                records=[self._stm_entry_to_record(entry) for entry in removed],
                message="short-term memory removed" if removed else "no matching STM memory",
            )

        targets = self._ltm_forget_targets(request)
        removed_records: list[MemoryRecord] = []
        if not request.hard:
            archival_ids: list[str] = []
            if request.memory_id:
                retired = self.archival.retire_memory_id(request.memory_id)
                if retired:
                    archival_ids.append(retired)
            elif request.query:
                archival_ids.extend(
                    self.archival.retire_matching_raw_events(request.query)
                )
            for memory_id in archival_ids:
                kind = "session" if memory_id.startswith("session:") else "event"
                removed_records.append(
                    MemoryRecord(
                        memory_id=memory_id,
                        scope="ltm",
                        content="",
                        entity=kind,
                        metadata={"tier": "archival", "status": "deleted"},
                    )
                )
        for entity, attribute in targets:
            if request.hard:
                current = self.core.find_by_id(request.memory_id) if request.memory_id else None
                current = current or self.core.hard_remove(entity, attribute)
                if current is not None and self.core.find_by_id(current.fact_id) is not None:
                    self.core.hard_remove(entity, attribute)
                history = self.archival.purge_fact_history(entity, attribute)
                deleted_facts = ([current] if current is not None else []) + history
                if not deleted_facts:
                    continue
                self.markdown_store.remove_core_fact(deleted_facts[0])
                self.semantic_index.clear_cache()
                stats = self.hard_delete_service.purge_derived(deleted_facts)
                self.archival.invalidate_session_documents()
                record = self._fact_to_record(deleted_facts[0])
                record.metadata["hard_deleted"] = True
                record.metadata["purged_versions"] = len(deleted_facts)
                record.metadata["purged_event_rows"] = stats.event_rows
                record.metadata["invalidated_cache_files"] = stats.cache_files
                removed_records.append(record)
                continue
            fact = self.core.remove(entity, attribute)
            if fact is None:
                continue
            self.archival.archive_fact(fact)
            self.markdown_store.remove_core_fact(fact)
            removed_records.append(self._fact_to_record(fact))
        return MemoryOperationResult(
            ok=bool(removed_records),
            action="forget",
            scope="ltm",
            records=removed_records,
            message=(
                "long-term memory physically deleted"
                if removed_records and request.hard
                else "long-term memory retired"
                if removed_records
                else "no matching LTM memory"
            ),
        )

    def update(self, request: UpdateRequest) -> MemoryOperationResult:
        """Update memory using the unified dispatcher."""
        return self.execute_action(self._action_from_update(request))

    def _update(self, request: UpdateRequest) -> MemoryOperationResult:
        """Update an existing memory or create it if LTM key does not exist."""
        if request.scope == "stm":
            if not request.memory_id:
                return MemoryOperationResult(
                    ok=False,
                    action="update",
                    scope="stm",
                    message="memory_id is required for STM update",
                )
            entry = self.stm.update(
                request.memory_id,
                content=request.content,
                tags=request.tags or None,
                metadata=request.metadata or None,
                ttl_seconds=request.ttl_seconds,
            )
            return MemoryOperationResult(
                ok=entry is not None,
                action="update",
                scope="stm",
                records=[self._stm_entry_to_record(entry)] if entry else [],
                message="short-term memory updated" if entry else "STM memory not found",
            )

        entity, attribute, value, tags, metadata, frame = self._normalize_ltm_request(
            content=request.content or "",
            entity=request.entity,
            attribute=request.attribute,
            value=request.value,
            tags=request.tags,
            metadata=request.metadata,
        )
        if frame is not None and not frame.should_persist:
            return self._nonpersistent_ltm_result("update", frame)
        incoming = FactCard(
            entity=entity,
            attribute=attribute,
            value=value,
            evidence=request.evidence or request.content or str(value),
            source_session_id=request.source_session_id,
            source_event_id=request.source_event_id,
            confidence=request.confidence,
            tags=tags,
            scope=str(metadata.get("scope", "")),
            intent=str(metadata.get("intent", "")),
            certainty=str(metadata.get("certainty", "certain")),
            temporal_status=str(metadata.get("temporal_status", "stable")),
        )
        existing = self.core.get(entity, attribute)
        alignment = None
        if existing is None:
            aligned = find_aligned_fact(
                incoming,
                self.core.active_facts(),
                threshold=float(self.config.pipeline_semantic_alignment_threshold),
            )
            if aligned is not None:
                existing, alignment = aligned
        target_entity = existing.entity if existing is not None else entity
        target_attribute = existing.attribute if existing is not None else attribute
        fact = self.core.set(
            entity=target_entity,
            attribute=target_attribute,
            value=value,
            evidence=request.evidence or request.content or str(value),
            source_session_id=request.source_session_id,
            source_event_id=request.source_event_id,
            confidence=request.confidence,
            tags=tags,
            importance=float(metadata.get("importance", 0.5)),
            scope=str(metadata.get("scope", "")),
            intent=str(metadata.get("intent", "")),
            certainty=str(metadata.get("certainty", "certain")),
            temporal_status=str(metadata.get("temporal_status", "stable")),
        )
        if existing and existing.value != value and not (alignment and alignment.same_value):
            self.archival.archive_fact(existing.supersede(fact.fact_id))
        self._sync_markdown(fact)
        return MemoryOperationResult(
            ok=True,
            action="update",
            scope="ltm",
            records=[self._fact_to_record(fact)],
            message="long-term memory updated",
        )

    def _sync_markdown(self, fact: FactCard) -> None:
        if bool(self.config.pipeline_backfill_legacy_markdown):
            self.markdown_store.upsert_core_fact(fact)

    @staticmethod
    def _nonpersistent_ltm_result(
        action: str, frame: MemoryFrame
    ) -> MemoryOperationResult:
        """Acknowledge transient or unverified statements without mutating LTM."""
        return MemoryOperationResult(
            ok=True,
            action=action,
            scope="ltm",
            message=(
                "long-term memory unchanged: "
                f"intent={frame.intent}, certainty={frame.certainty}, "
                f"temporal_status={frame.temporal_status}"
            ),
        )

    def recall(self, request: RecallRequest) -> MemoryOperationResult:
        """Retrieve memory using the unified dispatcher."""
        return self.execute_action(self._action_from_recall(request))

    def refresh_semantic_indexes(self) -> dict[str, int]:
        """Index current Core facts and session summaries before online recall."""
        stable_facts = [
            fact
            for fact in self.core.active_facts()
            if "preference_observation" not in fact.tags
        ]
        return {
            "core_facts": self.semantic_index.refresh(stable_facts),
            "session_documents": self.session_semantic_index.refresh(
                self.archival.session_documents()
            ),
        }

    def _recall(self, request: RecallRequest) -> MemoryOperationResult:
        """Retrieve memory from STM, LTM, or both."""
        if request.limit <= 0:
            return MemoryOperationResult(
                ok=False,
                action="recall",
                scope=request.scope,
                message="no matching memory",
            )
        records: list[MemoryRecord] = []
        candidate_limit = max(
            request.limit,
            int(self.config.agentic_policy_distillation_candidate_limit),
        )
        active_facts = self.core.active_facts()
        constraints = self.frame_parser.query_constraints(
            request.query,
            known_entities=sorted({fact.entity for fact in active_facts}),
            known_attributes=sorted({fact.attribute for fact in active_facts}),
        )
        query_variants = list(request.query_variants)
        structured_query = self._structured_query_route(constraints)
        if structured_query and structured_query.casefold() != request.query.casefold():
            query_variants.append(structured_query)
        entity_subqueries = []
        if not request.entity and len(constraints.entities) > 1:
            route_tail = " ".join(
                part
                for part in (constraints.scope, constraints.attribute or "")
                if part
            )
            entity_subqueries = [
                " ".join(part for part in (entity, route_tail) if part)
                for entity in constraints.entities
            ]
        plan = self.query_rewriter.plan(
            request.query,
            variants=query_variants,
            subqueries=(*request.subqueries, *entity_subqueries),
            rewrite=bool(self.config.query_rewrite_enabled),
        )
        entity_filter = request.entity or constraints.entity
        attribute_filter = (
            self.frame_parser.canonical_attribute(request.attribute)
            if request.attribute
            else constraints.attribute
        ) or request.attribute

        inferred_scope = constraints.scope or None
        scope_filter = request.fact_scope
        if (
            scope_filter is None
            and inferred_scope
            and any(
                fact.scope == inferred_scope
                or (
                    not fact.scope
                    and fact.entity.endswith(f"::{inferred_scope}")
                )
                for fact in active_facts
            )
        ):
            scope_filter = inferred_scope
        known_entities = {fact.entity for fact in active_facts}
        if entity_filter and scope_filter:
            scoped_entity = f"{entity_filter}::{scope_filter}"
            if scoped_entity in known_entities:
                entity_filter = scoped_entity
        days_filter = request.days if request.days is not None else constraints.days
        if request.scope in {"ltm", "all"}:
            fallback_attribute = attribute_filter if request.attribute else None
            fallback_days = days_filter if request.days is not None else None
            inferred_filters_relaxed = (
                fallback_attribute != attribute_filter
                or fallback_days != days_filter
            )

            retrieval_queries = plan.retrieval_queries
            prepared_lexical = self.core.search_many(
                retrieval_queries,
                top_k=candidate_limit,
                entity_filter=entity_filter,
                exclude_tags=["preference_observation"],
                attribute_filter=attribute_filter,
                scope_filter=scope_filter,
                days_filter=days_filter,
            )
            route_embeddings: list[list[float] | None] = [
                None for _ in retrieval_queries
            ]
            if request.query_embedding and route_embeddings:
                route_embeddings[0] = request.query_embedding
            embed_queries = getattr(self.semantic_index, "embed_queries", None)
            if self.semantic_index.enabled and callable(embed_queries):
                missing_indices = [
                    index
                    for index, embedding in enumerate(route_embeddings)
                    if embedding is None
                ]
                embedded = embed_queries(
                    [retrieval_queries[index] for index in missing_indices]
                )
                if len(embedded) == len(missing_indices):
                    for index, embedding in zip(missing_indices, embedded):
                        route_embeddings[index] = embedding

            prepared_semantic: list[list[Any]] | None = None
            search_many = getattr(self.semantic_index, "search_many", None)
            if self.semantic_index.enabled and callable(search_many):
                semantic_candidates = self.core.filter_active_facts(
                    entity_filter=entity_filter,
                    exclude_tags=["preference_observation"],
                    attribute_filter=attribute_filter,
                    scope_filter=scope_filter,
                    days_filter=days_filter,
                )
                prepared_semantic = search_many(
                    list(retrieval_queries),
                    semantic_candidates,
                    candidate_limit,
                    query_embeddings=route_embeddings,
                )

            query_results = []
            for query_index, search_query in enumerate(retrieval_queries):
                result = self.recall_strategy.recall(
                    search_query,
                    entity_filter=entity_filter,
                    attribute_filter=attribute_filter,
                    scope_filter=scope_filter,
                    days_filter=days_filter,
                    max_total=candidate_limit,
                    query_embedding=route_embeddings[query_index],
                    prepared_core_results=prepared_lexical[query_index],
                    prepared_semantic_hits=(
                        prepared_semantic[query_index]
                        if prepared_semantic is not None
                        else None
                    ),
                )
                if not result.facts and inferred_filters_relaxed:
                    result = self.recall_strategy.recall(
                        search_query,
                        entity_filter=entity_filter,
                        attribute_filter=fallback_attribute,
                        scope_filter=scope_filter,
                        days_filter=fallback_days,
                        max_total=candidate_limit,
                        query_embedding=route_embeddings[query_index],
                        prepared_semantic_hits=None,
                    )
                query_results.append(result)

            rankings = [result.facts for result in query_results]
            subquery_offset = 1 + len(plan.variants)
            subquery_rankings = rankings[
                subquery_offset : subquery_offset + len(plan.subqueries)
            ]
            facts = RecallStrategy.merge_query_rankings(
                rankings,
                subquery_rankings,
                candidate_limit,
            )
            hit_map: dict[str, list[int]] = {}
            subquery_hit_map: dict[str, list[int]] = {}
            for query_index, ranking in enumerate(rankings):
                for fact in ranking:
                    key = RecallStrategy._rrf_key(fact)
                    hit_map.setdefault(key, []).append(query_index)
                    if query_index >= subquery_offset:
                        subquery_hit_map.setdefault(key, []).append(
                            query_index - subquery_offset
                        )
            for fact in facts:
                record = self._fact_to_record(fact)
                key = RecallStrategy._rrf_key(fact)
                record.metadata["query_hits"] = sorted(set(hit_map.get(key, [])))
                record.metadata["subquery_hits"] = sorted(
                    set(subquery_hit_map.get(key, []))
                )
                records.append(record)
        if request.scope in {"stm", "all"}:
            records.extend(self.stm.search(request.query, limit=candidate_limit))
        records = self.distiller.distill(records, request.query, request.limit)
        return MemoryOperationResult(
            ok=bool(records),
            action="recall",
            scope=request.scope,
            records=records,
            message="memory recalled" if records else "no matching memory",
        )

    @staticmethod
    def _structured_query_route(constraints: object) -> str:
        """Render parser constraints as a compact, provider-neutral route."""
        parts = [
            str(getattr(constraints, "entity", "") or "").strip(),
            str(getattr(constraints, "scope", "") or "").strip(),
            str(getattr(constraints, "attribute", "") or "").strip(),
        ]
        return " ".join(dict.fromkeys(part for part in parts if part))

    def summarize_stm(self, request: MemoryActionRequest) -> MemoryOperationResult:
        """Summarize STM using the unified dispatcher."""
        return self.execute_action(replace(request, action="stm.summary"))

    def _summarize_stm(self, request: MemoryActionRequest) -> MemoryOperationResult:
        entry = self.stm.summarize(
            summary=request.summary or request.content,
            memory_ids=request.memory_ids,
            query=request.query,
            max_chars=max(int(request.metadata.get("max_chars", 1200)), 1),
            replace_sources=request.replace_sources,
        )
        record = self._stm_entry_to_record(entry) if entry else None
        return MemoryOperationResult(
            ok=record is not None,
            action="summary",
            scope="stm",
            records=[record] if record else [],
            message="STM context summarized" if record else "no STM entries to summarize",
        )

    def filter_stm(self, request: MemoryActionRequest) -> MemoryOperationResult:
        """Filter STM using the unified dispatcher."""
        return self.execute_action(replace(request, action="stm.filter"))

    def _filter_stm(self, request: MemoryActionRequest) -> MemoryOperationResult:
        kept, removed = self.stm.filter(
            request.query,
            limit=request.limit,
            memory_ids=request.memory_ids,
            remove_irrelevant=request.remove_irrelevant,
        )
        return MemoryOperationResult(
            ok=bool(kept),
            action="filter",
            scope="stm",
            records=kept,
            message=f"STM filter kept {len(kept)} and removed {len(removed)} entries",
        )

    def recall_context_block(self, query: str, entity: str | None = None) -> str:
        """Render LTM recall as a planner-friendly context block."""
        result = self.recall(
            RecallRequest(
                query=query,
                scope="ltm",
                limit=max(int(self.recall_strategy.max_total), 0),
                entity=entity,
            )
        )
        if not result.records:
            return ""
        lines = [f'<mem0_recall source="distilled" count="{len(result.records)}">']
        for record in result.records:
            lines.append(
                f"  - [{record.entity}] {record.attribute} = {record.value}"
                f"  (conf={record.confidence:.2f})"
            )
        lines.append("</mem0_recall>")
        return "\n".join(lines)

    def _ltm_forget_targets(self, request: ForgetRequest) -> list[tuple[str, str]]:
        if request.entity and request.attribute:
            return [(request.entity, request.attribute)]
        if request.memory_id:
            fact = self.core.find_by_id(request.memory_id)
            return [fact.key] if fact is not None else []
        if request.query:
            return [fact.key for fact in self.core.search(request.query, top_k=1)]
        return []

    def _ltm_key_value(
        self,
        *,
        content: str,
        entity: str | None,
        attribute: str | None,
        value: Any,
        frame: MemoryFrame | None = None,
    ) -> tuple[str, str, Any]:
        clean = content.strip()
        if frame is not None and frame.should_persist:
            if not entity or (
                frame.scope and "::" not in entity and entity == frame.entity
            ):
                entity = frame.storage_entity
            if not attribute:
                attribute = frame.attribute
            if value is None:
                value = frame.value
        final_entity = (entity or "memory").strip()
        raw_attribute = (attribute or self._default_attribute(clean)).strip()
        final_attribute = (
            self.frame_parser.canonical_attribute(raw_attribute) or raw_attribute
        )
        final_value = value if value is not None else clean
        if not final_attribute:
            final_attribute = "note"
        return final_entity, final_attribute, final_value

    def _normalize_ltm_request(
        self,
        *,
        content: str,
        entity: str | None,
        attribute: str | None,
        value: Any,
        tags: list[str],
        metadata: dict[str, Any],
    ) -> tuple[str, str, Any, list[str], dict[str, Any], MemoryFrame | None]:
        clean = content.strip()
        merged_metadata = dict(metadata)
        frame = self.frame_parser.parse(clean, metadata={**merged_metadata, "entity": entity or "", "attribute": attribute or "", "value": value})
        merged_tags = list(tags)
        if frame is not None and frame.should_persist:
            merged_metadata.setdefault("intent", frame.intent)
            merged_metadata.setdefault("certainty", frame.certainty)
            merged_metadata.setdefault("temporal_status", frame.temporal_status)
            if frame.scope and not merged_metadata.get("scope"):
                merged_metadata["scope"] = frame.scope
        final_entity, final_attribute, final_value = self._ltm_key_value(
            content=clean,
            entity=entity,
            attribute=attribute,
            value=value,
            frame=frame,
        )
        return final_entity, final_attribute, final_value, merged_tags, merged_metadata, frame

    @staticmethod
    def _default_attribute(content: str) -> str:
        first_line = content.splitlines()[0].strip() if content.strip() else "note"
        return first_line[:80] or "note"

    def _fact_to_record(self, fact: FactCard) -> MemoryRecord:
        return MemoryRecord(
            memory_id=fact.fact_id,
            scope="ltm",
            content=f"{fact.entity}.{fact.attribute} = {fact.value}",
            entity=fact.entity,
            attribute=fact.attribute,
            value=fact.value,
            confidence=fact.confidence,
            tags=fact.tags,
            metadata={
                "version": fact.version,
                "status": fact.status,
                "tier": fact.tier,
                "importance": fact.importance,
                "created_at": fact.created_at,
                "updated_at": fact.updated_at,
                "source_session_id": fact.source_session_id,
                "source_event_id": fact.source_event_id,
                "search_score": fact.search_score,
                "semantic_score": fact.semantic_score,
                "scope": fact.scope,
                "intent": fact.intent,
                "certainty": fact.certainty,
                "temporal_status": fact.temporal_status,
                **fact.custom_metadata,  # ← 合并 custom_metadata
            },
        )

    @staticmethod
    def _stm_entry_to_record(entry: Any) -> MemoryRecord:
        return MemoryRecord(
            memory_id=entry.memory_id,
            scope="stm",
            content=entry.content,
            value=entry.content,
            tags=entry.tags,
            metadata={
                **entry.metadata,
                "created_at": entry.created_at,
                "updated_at": entry.updated_at,
                "expires_at": entry.expires_at,
            },
        )

    @staticmethod
    def _remember_request(request: MemoryActionRequest, *, scope: str) -> RememberRequest:
        return RememberRequest(
            content=request.content,
            scope=scope,  # type: ignore[arg-type]
            memory_id=request.memory_id,
            entity=request.entity,
            attribute=request.attribute,
            value=request.value,
            evidence=request.evidence,
            confidence=request.confidence,
            tags=request.tags,
            metadata=request.metadata,
            source_session_id=request.source_session_id,
            source_event_id=request.source_event_id,
            ttl_seconds=request.ttl_seconds,
        )

    @staticmethod
    def _update_request(request: MemoryActionRequest, *, scope: str) -> UpdateRequest:
        return UpdateRequest(
            scope=scope,  # type: ignore[arg-type]
            memory_id=request.memory_id,
            content=request.content or None,
            entity=request.entity,
            attribute=request.attribute,
            value=request.value,
            evidence=request.evidence,
            confidence=request.confidence,
            tags=request.tags,
            metadata=request.metadata,
            source_session_id=request.source_session_id,
            source_event_id=request.source_event_id,
            ttl_seconds=request.ttl_seconds,
        )

    @staticmethod
    def _forget_request(request: MemoryActionRequest, *, scope: str) -> ForgetRequest:
        return ForgetRequest(
            scope=scope,  # type: ignore[arg-type]
            memory_id=request.memory_id,
            entity=request.entity,
            attribute=request.attribute,
            query=request.query or None,
            hard=request.hard,
        )

    @staticmethod
    def _action_from_remember(request: RememberRequest) -> MemoryActionRequest:
        return MemoryActionRequest(
            action="remember",
            scope=request.scope,
            content=request.content,
            memory_id=request.memory_id,
            entity=request.entity,
            attribute=request.attribute,
            value=request.value,
            evidence=request.evidence,
            confidence=request.confidence,
            tags=request.tags,
            metadata=request.metadata,
            source_session_id=request.source_session_id,
            source_event_id=request.source_event_id,
            ttl_seconds=request.ttl_seconds,
        )

    @staticmethod
    def _action_from_update(request: UpdateRequest) -> MemoryActionRequest:
        return MemoryActionRequest(
            action="update",
            scope=request.scope,
            content=request.content or "",
            memory_id=request.memory_id,
            entity=request.entity,
            attribute=request.attribute,
            value=request.value,
            evidence=request.evidence,
            confidence=request.confidence,
            tags=request.tags,
            metadata=request.metadata,
            source_session_id=request.source_session_id,
            source_event_id=request.source_event_id,
            ttl_seconds=request.ttl_seconds,
        )

    @staticmethod
    def _action_from_forget(request: ForgetRequest) -> MemoryActionRequest:
        return MemoryActionRequest(
            action="forget",
            scope=request.scope,
            memory_id=request.memory_id,
            entity=request.entity,
            attribute=request.attribute,
            query=request.query or "",
            hard=request.hard,
        )

    @staticmethod
    def _action_from_recall(request: RecallRequest) -> MemoryActionRequest:
        return MemoryActionRequest(
            action="recall",
            scope=request.scope,
            query=request.query,
            limit=request.limit,
            entity=request.entity,
            attribute=request.attribute,
            fact_scope=request.fact_scope,
            days=request.days,
            query_variants=request.query_variants,
            subqueries=request.subqueries,
            query_embedding=request.query_embedding,
        )

    @staticmethod
    def _mutation_scope(
        action: str,
        scope: str,
        *,
        forced_scope: str | None = None,
    ) -> str:
        final_scope = forced_scope or scope
        if final_scope not in {"ltm", "stm"}:
            raise ValueError(f"{action} requires scope 'ltm' or 'stm'")
        return final_scope

    def _trace_action(
        self,
        action: str,
        request: MemoryActionRequest,
        result: MemoryOperationResult,
    ) -> None:
        if self.trace_store is not None:
            self.trace_store.append(action, request.to_dict(), result)
