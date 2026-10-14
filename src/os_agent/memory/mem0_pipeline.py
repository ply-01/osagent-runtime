"""Mem0 Extract-Update Pipeline.

Batch-processes raw event/session logs to:
  1. Extract structured FactCards from raw interactions.
  2. Merge / update existing Core facts (version bump on change).
  3. Archive superseded facts.
  4. Compute importance scores based on access patterns.

This replaces the real-time MemoryHook→Markdown flow with a
Mem0-style batch pipeline that runs after each session.
"""

from __future__ import annotations

import json
import re
from hashlib import blake2b
from pathlib import Path

from os_agent.memory.behavior_aggregator import BehaviorPreferenceAggregator
from os_agent.memory.markdown_store import MarkdownMemoryStore
from os_agent.memory.mem0_archival import ArchivalMemoryStore
from os_agent.memory.mem0_core import CoreMemoryStore
from os_agent.memory.mem0_schemas import FactCard
from os_agent.memory.memory_config import MemoryConfig
from os_agent.memory.memory_frame import MemoryFrameParser
from os_agent.memory.preference_extractor import PreferenceExtractor
from os_agent.memory.retrieval_text import tokenize_search_text
from os_agent.memory.semantic_alignment import find_aligned_fact
from os_agent.memory.semantic_extractor import (
    SemanticExtractionError,
    SemanticMemoryExtractor,
    create_semantic_extractor,
)
from os_agent.memory.state_assembler import StateAssembler, normalized_value

_GENERIC_PREFERENCE_TERMS = {
    "appreciate",
    "available",
    "care",
    "curious",
    "enjoy",
    "explore",
    "favor",
    "favour",
    "hope",
    "ideal",
    "interested",
    "lean",
    "love",
    "plan",
    "prefer",
    "preference",
    "rather",
    "recommend",
    "user",
    "value",
}

_NEGATIVE_PREFERENCE = re.compile(
    r"(?:不要|不想|不喜欢|讨厌|避免|排除|不能|无法接受)|"
    r"\b(?:avoid(?:ing|ed)?|exclude|dislike|hate|without|do not want|don't want|cannot accept)\b",
    re.IGNORECASE,
)


class ExtractUpdatePipeline:
    """Batch pipeline that processes raw agent events into Core facts.

    Usage:
        pipeline = ExtractUpdatePipeline(core, archival)
        pipeline.process_events(session_events)  # or process_jsonl(path)

    The pipeline is idempotent — processing the same events twice
    will reinforce (boost confidence) rather than duplicate facts.
    """

    def __init__(
        self,
        core: CoreMemoryStore,
        archival: ArchivalMemoryStore,
        legacy_store: MarkdownMemoryStore | None = None,
        config: MemoryConfig | None = None,
        semantic_extractor: SemanticMemoryExtractor | None = None,
    ) -> None:
        self.core = core
        self.archival = archival
        self.legacy = legacy_store
        self.config = config or MemoryConfig.global_config()
        self.preference_extractor = PreferenceExtractor()
        self.frame_parser = MemoryFrameParser(self.preference_extractor)
        self.semantic_extractor = semantic_extractor
        if self.semantic_extractor is None:
            try:
                self.semantic_extractor = create_semantic_extractor(self.config)
            except SemanticExtractionError:
                if not bool(self.config.pipeline_semantic_fallback_to_rules):
                    raise
        self.state_assembler = StateAssembler(
            getattr(self.semantic_extractor, "provider", None),
            context_limit=int(self.config.pipeline_state_resolver_context_limit),
        )
        self.behavior_aggregator = BehaviorPreferenceAggregator(
            self.core.root,
            min_observations=int(self.config.pipeline_behavior_min_observations),
        )

    # ------------------------------------------------------------------
    # Main pipeline entry point
    # ------------------------------------------------------------------

    def process_session(
        self,
        events: list[dict],
        session_id: str = "",
    ) -> list[FactCard]:
        """Process a list of raw event dicts from a single session.

        Returns the list of FactCards that were created or updated.
        """
        semantic_facts: list[FactCard] | None = None
        if self.semantic_extractor is not None:
            try:
                semantic_facts = self.semantic_extractor.extract(events, session_id)
            except SemanticExtractionError:
                if not bool(self.config.pipeline_semantic_fallback_to_rules):
                    raise
        if bool(self.config.pipeline_archive_raw_events):
            archived_events = [
                {
                    **event,
                    "session_id": str(event.get("session_id") or session_id),
                }
                for event in events
            ]
            self.archival.append_raw_events(archived_events)
        extracted = self._extract_facts(
            events,
            session_id,
            semantic_facts=semantic_facts,
        )
        if (
            semantic_facts is None
            and bool(self.config.pipeline_session_preference_consolidation_enabled)
            and not any("preference" in fact.tags for fact in extracted)
        ):
            observation = self._consolidate_session_preference(events, session_id)
            if observation is not None:
                extracted.append(observation)
        if bool(self.config.pipeline_behavior_enabled):
            extracted.extend(self.behavior_aggregator.observe(events, session_id))
        if bool(self.config.pipeline_session_dedup) and semantic_facts is None:
            latest_by_key: dict[tuple[str, str], FactCard] = {}
            for fact in extracted:
                latest_by_key[fact.key] = fact
            extracted = list(latest_by_key.values())
        results: list[FactCard] = []
        for fact in extracted:
            result = self._merge_into_core(fact)
            if result is not None:
                results.append(result)
        if bool(self.config.pipeline_preference_reinforcement_enabled):
            for observation in [
                fact for fact in results if "preference_observation" in fact.tags
            ]:
                reinforced = self._reinforce_preference_observation(observation)
                if reinforced is not None:
                    merged = self._merge_into_core(reinforced)
                    if merged is not None:
                        results.append(merged)
        return results

    @property
    def runtime_stats(self) -> dict[str, int | float]:
        """Return aggregate semantic-stage diagnostics without example text."""

        combined: dict[str, int | float] = {}
        if self.semantic_extractor is not None:
            combined.update(dict(self.semantic_extractor.stats))
        for key, value in self.state_assembler.stats.items():
            combined[key] = combined.get(key, 0) + value
        return combined

    def process_jsonl(self, path: Path | str) -> list[FactCard]:
        """Process a JSONL file (data/sessions/*.jsonl or data/events/*.jsonl)."""
        path = Path(path)
        if not path.exists():
            return []
        events = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        session_id = events[0].get("session_id", "") if events else ""
        return self.process_session(events, session_id=session_id)

    # ------------------------------------------------------------------
    # Extraction rules (port from MemoryHook → structured FactCards)
    # ------------------------------------------------------------------

    def _extract_facts(
        self,
        events: list[dict],
        session_id: str,
        *,
        semantic_facts: list[FactCard] | None = None,
    ) -> list[FactCard]:
        """Extract structured FactCards from raw events.

        Rules (ported from MemoryHook, but output is structured):
          R1: read_file success → knowledge fact
          R2: list_dir success   → knowledge fact
          R3: shell success      → tool_history fact
          R4: tool failure       → tool_history fact
          R5: user preference    → preference fact
          R6: user safety        → safety fact
        """
        facts: list[FactCard] = list(semantic_facts or [])
        entity = "project"  # default entity; could be inferred from context
        conf_read = float(self.config.pipeline_rule_confidence_read_file_success)
        conf_list = float(self.config.pipeline_rule_confidence_list_dir_success)
        conf_shell = float(self.config.pipeline_rule_confidence_shell_success)
        conf_fail = float(self.config.pipeline_rule_confidence_tool_failure)
        conf_pref = float(self.config.pipeline_rule_confidence_user_preference)
        conf_safety = float(self.config.pipeline_rule_confidence_user_safety)
        conf_pdf = float(self.config.pipeline_rule_confidence_read_pdf_success)
        imp_read = float(self.config.pipeline_rule_importance_read_file_success)
        imp_list = float(self.config.pipeline_rule_importance_list_dir_success)
        imp_shell = float(self.config.pipeline_rule_importance_shell_success)
        imp_fail = float(self.config.pipeline_rule_importance_tool_failure)
        imp_pref = float(self.config.pipeline_rule_importance_user_preference)
        imp_safety = float(self.config.pipeline_rule_importance_user_safety)
        imp_pdf = float(self.config.pipeline_rule_importance_read_pdf_success)

        for raw in events:
            event_type = raw.get("event_type", "")
            payload = raw.get("payload", {})
            event_id = raw.get("event_id", "")

            # --- message_start (user) for R5 / R6 ---
            if event_type == "message_start" and payload.get("role") == "user":
                text = str(payload.get("content", "")).strip()
                if not text:
                    continue
                # R5: asserted preferences with normalized memory fields. A
                # successful semantic pass owns user-preference decisions,
                # including the valid decision that a message has no fact.
                if semantic_facts is None:
                    frame = self.frame_parser.parse(
                        text,
                        metadata={**raw, **payload},
                    )
                    if frame is not None and frame.should_persist:
                        facts.append(
                            FactCard(
                                entity=frame.storage_entity,
                                attribute=frame.attribute,
                                value=frame.value,
                                evidence=text[:300],
                                source_session_id=session_id,
                                source_event_id=event_id,
                                confidence=max(conf_pref, frame.confidence),
                                tags=[*frame.tags, "source:user_message"],
                                scope=frame.scope,
                                intent=frame.intent,
                                certainty=frame.certainty,
                                temporal_status=frame.temporal_status,
                                importance=imp_pref,
                            )
                        )

                # R6: safety
                safety_cn = re.search(
                    r"(不要|别|禁止|严禁|绝不|千万[别再]|避免|当心|小心)", text
                )
                safety_en = re.search(
                    r"\b(?:don'?t\s|never\s|forbid|prohibit|avoid|beware|warning)",
                    text,
                    re.IGNORECASE,
                )
                if safety_cn or safety_en:
                    facts.append(
                        FactCard(
                            entity="用户",
                            attribute="安全约束",
                            value=text[:200],
                            evidence=text[:300],
                            source_session_id=session_id,
                            source_event_id=event_id,
                            confidence=conf_safety,
                            tags=["user_message", "safety"],
                            importance=imp_safety,
                        )
                    )

            # --- tool_result for R1-R4 ---
            if event_type != "tool_result":
                continue

            tool_name = str(payload.get("tool_name", ""))
            status = str(payload.get("status", ""))

            # R1: read_file success
            if tool_name == "filesystem.read_file" and status == "success":
                output = payload.get("output", {})
                path = output.get("path", "")
                if bool(
                    self.config.pipeline_skip_memory_file_facts
                ) and self._is_memory_metadata_path(path):
                    continue
                content = str(output.get("content", "")).strip()
                if path and content:
                    summary = self._summarize_text(content)
                    attribute = "文件内容"
                    if bool(self.config.pipeline_per_file_attributes):
                        attribute = f"文件内容:{Path(path).name}"
                    facts.append(
                        FactCard(
                            entity=entity,
                            attribute=attribute,
                            value=f"{path}: {summary[:300]}",
                            evidence=f"读取文件 {path}",
                            source_session_id=session_id,
                            source_event_id=event_id,
                            confidence=conf_read,
                            tags=["tool_trace", "file"],
                            importance=imp_read,
                        )
                    )

            # R2: list_dir success
            if tool_name == "filesystem.list_dir" and status == "success":
                output = payload.get("output", {})
                path = output.get("path", ".")
                entries = output.get("entries", [])
                if entries:
                    names = [e.get("name", "") for e in entries[:20]]
                    facts.append(
                        FactCard(
                            entity=entity,
                            attribute="目录结构",
                            value=f"{path}: {', '.join(names)}",
                            evidence=f"列出目录 {path}",
                            source_session_id=session_id,
                            source_event_id=event_id,
                            confidence=conf_list,
                            tags=["tool_trace", "directory"],
                            importance=imp_list,
                        )
                    )

            # R3: shell success
            if tool_name == "shell.run_safe" and status == "success":
                output = payload.get("output", {})
                command = str(payload.get("input", {}).get("command", "")).strip()
                stdout = str(output.get("stdout", "")).strip()
                if command:
                    facts.append(
                        FactCard(
                            entity=entity,
                            attribute="Shell执行",
                            value=f"{command}: {self._summarize_text(stdout) if stdout else '(无输出)'}",
                            evidence=f"执行命令 {command}",
                            source_session_id=session_id,
                            source_event_id=event_id,
                            confidence=conf_shell,
                            tags=["tool_trace", "shell"],
                            importance=imp_shell,
                        )
                    )

            # R4: tool failure
            if tool_name and status != "success":
                error = str(payload.get("error", "unknown error"))
                facts.append(
                    FactCard(
                        entity=entity,
                        attribute="工具失败",
                        value=f"{tool_name}: {error[:200]}",
                        evidence=f"工具 {tool_name} 执行失败: {error[:300]}",
                        source_session_id=session_id,
                        source_event_id=event_id,
                        confidence=conf_fail,
                        tags=["tool_trace", "failure"],
                        importance=imp_fail,
                    )
                )

            # new: read_pdf success
            if tool_name == "filesystem.read_pdf" and status == "success":
                output = payload.get("output", {})
                path = output.get("path", "")
                pages = output.get("pages", [])
                total = output.get("total_pages", len(pages))
                if path and pages:
                    first_text = pages[0].get("text", "")[:300] if pages else ""
                    attribute = "PDF内容"
                    if bool(self.config.pipeline_per_file_attributes):
                        attribute = f"PDF内容:{Path(path).name}"
                    facts.append(
                        FactCard(
                            entity=entity,
                            attribute=attribute,
                            value=f"{path} ({total}页): {self._summarize_text(first_text)}",
                            evidence=f"读取PDF {path}, 共{total}页",
                            source_session_id=session_id,
                            source_event_id=event_id,
                            confidence=conf_pdf,
                            tags=["tool_trace", "pdf"],
                            importance=imp_pdf,
                        )
                    )

        return facts

    def _consolidate_session_preference(
        self,
        events: list[dict],
        session_id: str,
    ) -> FactCard | None:
        observations: list[tuple[str, str, str]] = []
        task_id = ""
        for raw in events:
            task_id = task_id or str(raw.get("task_id", "")).strip()
            if raw.get("event_type") != "message_start":
                continue
            payload = raw.get("payload", {})
            if not isinstance(payload, dict) or payload.get("role") != "user":
                continue
            frame = self.frame_parser.parse(
                str(payload.get("content", "")),
                metadata={**raw, **payload},
            )
            if frame is not None and not frame.should_persist:
                continue
            signal = self.preference_extractor.extract_observation(
                str(payload.get("content", ""))
            )
            if signal is not None:
                observations.append(
                    (str(raw.get("event_id", "")), signal.value, signal.attribute)
                )
        if not observations:
            return None
        stable_session_id = session_id or task_id or observations[0][0]
        digest = blake2b(stable_session_id.encode("utf-8"), digest_size=8).hexdigest()
        value = "\n".join(dict.fromkeys(value for _, value, _ in observations))[:900]
        attributes = list(dict.fromkeys(attribute for _, _, attribute in observations))
        polarities = {
            self._preference_polarity(item_value)
            for _, item_value, _ in observations
        }
        polarity = polarities.pop() if len(polarities) == 1 else "mixed"
        return FactCard(
            entity="用户",
            attribute=f"preference.session_observation.{digest}",
            value=value,
            evidence=value[:500],
            source_session_id=stable_session_id,
            source_event_id=observations[-1][0],
            confidence=float(self.config.pipeline_session_preference_confidence),
            tags=[
                "user_message",
                "preference",
                "preference_observation",
                "source:session_consolidation",
                f"polarity:{polarity}",
                *(f"attribute:{attribute}" for attribute in attributes),
            ],
            importance=float(self.config.pipeline_session_preference_importance),
        )

    def _reinforce_preference_observation(
        self, observation: FactCard
    ) -> FactCard | None:
        polarity = self._tag_value(observation, "polarity:")
        if not polarity or polarity == "mixed":
            return None
        threshold = float(self.config.pipeline_preference_reinforcement_similarity)
        candidates: list[tuple[float, FactCard]] = []
        for fact in self.core.active_facts():
            if "preference_observation" not in fact.tags:
                continue
            if fact.source_session_id == observation.source_session_id:
                continue
            if self._tag_value(fact, "polarity:") != polarity:
                continue
            similarity = self._preference_similarity(observation, fact)
            if similarity >= threshold:
                candidates.append((similarity, fact))
        candidates.sort(key=lambda item: (-item[0], item[1].fact_id))
        evidence = [observation, *(fact for _, fact in candidates)]
        distinct_sessions = {
            fact.source_session_id for fact in evidence if fact.source_session_id
        }
        minimum = max(
            int(self.config.pipeline_preference_reinforcement_min_sessions), 2
        )
        if len(distinct_sessions) < minimum:
            return None

        existing = next(
            (
                fact
                for fact in self.core.active_facts()
                if "preference_reinforced" in fact.tags
                and self._tag_value(fact, "polarity:") == polarity
                and self._preference_similarity(observation, fact) >= threshold
            ),
            None,
        )
        if existing is not None:
            attribute = existing.attribute
            values = [str(existing.value), *(str(fact.value) for fact in evidence)]
            prior_session_tags = [
                tag for tag in existing.tags if tag.startswith("evidence_session:")
            ]
        else:
            shared_terms = self._preference_terms(observation)
            for fact in evidence[1:]:
                shared_terms &= self._preference_terms(fact)
            topics = sorted(self._preference_topics(observation))
            signature = "|".join([polarity, *topics, *sorted(shared_terms)])
            digest = blake2b(signature.encode("utf-8"), digest_size=8).hexdigest()
            attribute = f"preference.reinforced.{digest}"
            values = [str(fact.value) for fact in evidence]
            prior_session_tags = []

        session_tags = [
            f"evidence_session:{blake2b(session.encode('utf-8'), digest_size=6).hexdigest()}"
            for session in sorted(distinct_sessions)
        ]
        unique_values = list(dict.fromkeys(value.strip() for value in values if value.strip()))
        confidence = min(
            float(self.config.pipeline_preference_reinforcement_confidence)
            + max(len(set(prior_session_tags + session_tags)) - minimum, 0) * 0.05,
            0.95,
        )
        topic_tags = sorted(
            {
                tag
                for fact in evidence
                for tag in fact.tags
                if tag.startswith("attribute:")
            }
        )
        return FactCard(
            entity="用户",
            attribute=attribute,
            value="\n".join(unique_values)[:1200],
            evidence=(
                f"Reinforced by {len(set(prior_session_tags + session_tags))} "
                "independent sessions: " + " | ".join(unique_values)
            )[:500],
            source_session_id=observation.source_session_id,
            source_event_id=observation.source_event_id,
            confidence=confidence,
            tags=[
                "preference",
                "preference_reinforced",
                "source:cross_session_consolidation",
                f"polarity:{polarity}",
                *topic_tags,
                *sorted(set(prior_session_tags + session_tags)),
            ],
            importance=float(self.config.pipeline_preference_reinforcement_importance),
        )

    @staticmethod
    def _tag_value(fact: FactCard, prefix: str) -> str:
        return next(
            (tag.removeprefix(prefix) for tag in fact.tags if tag.startswith(prefix)),
            "",
        )

    @staticmethod
    def _preference_polarity(value: str) -> str:
        return "avoid" if _NEGATIVE_PREFERENCE.search(value) else "prefer"

    @staticmethod
    def _preference_topics(fact: FactCard) -> set[str]:
        return {
            tag.removeprefix("attribute:")
            for tag in fact.tags
            if tag.startswith("attribute:")
            and not tag.startswith("attribute:preference.unclassified.")
        }

    @staticmethod
    def _preference_terms(fact: FactCard) -> set[str]:
        return {
            term
            for term in tokenize_search_text(str(fact.value))
            if term not in _GENERIC_PREFERENCE_TERMS
            and not (len(term) == 1 and "\u4e00" <= term <= "\u9fff")
        }

    @classmethod
    def _preference_similarity(cls, left: FactCard, right: FactCard) -> float:
        left_topics = cls._preference_topics(left)
        right_topics = cls._preference_topics(right)
        if left_topics and right_topics and left_topics.isdisjoint(right_topics):
            return 0.0
        left_terms = cls._preference_terms(left)
        right_terms = cls._preference_terms(right)
        if not left_terms or not right_terms:
            return 0.0
        overlap = left_terms & right_terms
        required = 1 if min(len(left_terms), len(right_terms)) <= 2 else 2
        if len(overlap) < required:
            return 0.0
        return len(overlap) / min(len(left_terms), len(right_terms))

    # ------------------------------------------------------------------
    # Core merge logic
    # ------------------------------------------------------------------

    def _merge_into_core(self, fact: FactCard) -> FactCard | None:
        """Merge an extracted fact into Core.

        - New fact → insert
        - Same entity+attribute, same value → reinforce (boost confidence)
        - Aligned entity/scope/attribute, different value → supersede old
        """
        semantic_mention = (
            fact.custom_metadata.get("extraction") == "semantic_mention"
            or "source:semantic_mention" in fact.tags
        )
        existing = self.core.get(fact.entity, fact.attribute)
        alignment = None
        needs_recheck = False
        if existing is None:
            if semantic_mention:
                decision = self.state_assembler.resolve_slot(
                    fact, self.core.active_facts()
                )
                if decision.relation == "same_slot":
                    existing = decision.target
                elif decision.relation == "ambiguous":
                    # Preserve both claims and surface the uncertainty in
                    # metadata; an uncertain classifier never overwrites.
                    needs_recheck = True
            else:
                aligned = find_aligned_fact(
                    fact,
                    self.core.active_facts(),
                    threshold=float(self.config.pipeline_semantic_alignment_threshold),
                )
                if aligned is not None:
                    existing, alignment = aligned

        if (
            semantic_mention
            and existing is None
            and fact.intent == "revert"
            and not str(fact.value or "").strip()
        ):
            self.state_assembler.stats["unresolved_reverts"] += 1
            return None

        target_entity = existing.entity if existing is not None else fact.entity
        target_attribute = existing.attribute if existing is not None else fact.attribute
        target_scope = existing.scope if existing is not None else fact.scope
        target_tags = list(fact.tags)
        if semantic_mention:
            target_tags = [
                tag
                for tag in target_tags
                if not tag.startswith("attribute:") and not tag.startswith("scope:")
            ]
            if "preference" in target_tags:
                target_tags.append(f"attribute:{target_attribute}")
                if target_scope:
                    target_tags.append(f"scope:{target_scope}")
        metadata = (
            self.state_assembler.transition_metadata(
                fact,
                existing,
                needs_recheck=needs_recheck,
            )
            if semantic_mention
            else dict(fact.custom_metadata)
        )

        if existing is None:
            # Insert new
            result = self.core.set(
                entity=target_entity,
                attribute=target_attribute,
                value=fact.value,
                evidence=fact.evidence,
                source_session_id=fact.source_session_id,
                source_event_id=fact.source_event_id,
                confidence=fact.confidence,
                tags=target_tags,
                importance=fact.importance,
                scope=target_scope,
                intent=fact.intent,
                certainty=fact.certainty,
                temporal_status=fact.temporal_status,
                custom_metadata=metadata,
            )
            self._write_legacy(result)
            return result

        # Value unchanged → reinforce
        value = fact.value
        if semantic_mention and fact.intent == "revert":
            value = self.state_assembler.resolve_revert_value(
                fact,
                existing,
                self.archival.get_fact_history(existing.entity, existing.attribute),
            )

        if (
            existing.value == value
            or (semantic_mention and normalized_value(existing.value) == normalized_value(value))
            or (alignment is not None and alignment.same_value)
        ):
            existing.confidence = min(
                existing.confidence + float(self.config.pipeline_confidence_boost),
                1.0,
            )
            if semantic_mention:
                existing.custom_metadata = metadata
                existing.tags = list(dict.fromkeys([*existing.tags, *target_tags]))
            self.core._save()
            self._write_legacy(existing)
            return existing

        result = self.core.set(
            entity=target_entity,
            attribute=target_attribute,
            value=value,
            evidence=fact.evidence,
            source_session_id=fact.source_session_id,
            source_event_id=fact.source_event_id,
            confidence=fact.confidence,
            tags=target_tags,
            importance=fact.importance,
            scope=target_scope,
            intent=fact.intent,
            certainty=fact.certainty,
            temporal_status=fact.temporal_status,
            custom_metadata=metadata,
        )
        self.archival.archive_fact(existing.supersede(result.fact_id))
        self._write_legacy(result)
        return result

    def _write_legacy(self, fact: FactCard) -> None:
        """Keep the readable Markdown mirror derived from Core."""
        if self.legacy is None or not bool(
            self.config.pipeline_backfill_legacy_markdown
        ):
            return
        self.legacy.upsert_core_fact(fact)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _summarize_text(text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return ""
        joined = " ".join(lines[:3])
        return joined[:500]

    @staticmethod
    def _is_memory_metadata_path(path: object) -> bool:
        normalized = str(path).replace("\\", "/").lower()
        metadata_names = (
            "/profile.md",
            "/safety.md",
            "/tool-history.md",
            "/project-knowledge.md",
            "/workflows.md",
            "/core_facts.json",
            "/archive_facts.json",
        )
        return (
            "/memory/" in normalized
            or normalized.startswith("memory/")
            or normalized.endswith(metadata_names)
        )
