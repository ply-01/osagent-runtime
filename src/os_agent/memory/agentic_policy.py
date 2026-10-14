"""Policy support for agent-controlled memory actions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from os_agent.memory.interface import MemoryOperationResult, MemoryRecord
from os_agent.memory.retrieval_text import tokenize_search_text


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class MemoryDistiller:
    """Select the most useful records from a larger recall candidate pool.

    This deterministic implementation is the runtime fallback. A learned
    policy can replace it later without changing AgentMemoryModule.
    """

    def distill(
        self,
        records: list[MemoryRecord],
        query: str,
        limit: int,
    ) -> list[MemoryRecord]:
        if limit <= 0 or not records:
            return []
        terms = self._tokenize(query)
        query_lower = query.lower()
        scored: list[tuple[float, int, MemoryRecord]] = []
        for index, record in enumerate(records):
            text = " ".join(
                [
                    record.content,
                    record.entity,
                    record.attribute,
                    str(record.value),
                    " ".join(record.tags),
                ]
            ).lower()
            record_terms = set(self._tokenize(text))
            overlap = sum(1 for term in terms if term in record_terms)
            lexical_ratio = overlap / max(1, len(set(terms)))
            importance = self._float(record.metadata.get("importance"), 0.5)
            upstream = self._float(record.metadata.get("search_score"), 0.0)
            semantic = max(
                0.0,
                min(self._float(record.metadata.get("semantic_score"), 0.0), 1.0),
            )
            rank_signal = max(0.0, min(upstream * 61.0, 1.0))
            evidence_agreement = lexical_ratio * semantic
            entity_match = self._field_match(query_lower, terms, record.entity)
            attribute_match = self._field_match(
                query_lower, terms, record.attribute
            )
            score = (
                lexical_ratio * 0.32
                + semantic * 0.23
                + evidence_agreement * 0.04
                + rank_signal * 0.10
                + record.confidence * 0.05
                + importance * 0.05
                + entity_match * 0.16
                + attribute_match * 0.05
                + self._source_bonus(record)
            )
            record.metadata["distillation_score"] = round(score, 6)
            record.metadata["evidence_agreement"] = round(evidence_agreement, 6)
            scored.append((score, index, record))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return self._select_query_coverage(scored, limit)

    @classmethod
    def _select_query_coverage(
        cls,
        scored: list[tuple[float, int, MemoryRecord]],
        limit: int,
    ) -> list[MemoryRecord]:
        subqueries = sorted(
            {
                int(hit)
                for _, _, record in scored
                for hit in record.metadata.get("subquery_hits", [])
            }
        )
        if not subqueries:
            return cls._select_session_diverse(scored, limit)

        selected: list[MemoryRecord] = []
        selected_ids: set[str] = set()
        covered: set[int] = set()
        for _, _, record in scored:
            hits = {
                int(hit) for hit in record.metadata.get("subquery_hits", [])
            }
            if not hits - covered:
                continue
            selected.append(record)
            selected_ids.add(record.memory_id)
            covered.update(hits)
            if len(selected) == limit or len(covered) == len(subqueries):
                break
        if len(selected) == limit:
            return selected
        remaining = [
            item for item in scored if item[2].memory_id not in selected_ids
        ]
        selected.extend(
            cls._select_session_diverse(remaining, limit - len(selected))
        )
        return selected[:limit]

    @classmethod
    def _field_match(
        cls,
        query_lower: str,
        query_terms: list[str],
        field: str,
    ) -> float:
        clean = field.strip().lower()
        if not clean:
            return 0.0
        aliases = {
            clean,
            clean.replace("_", " "),
            clean.replace("_", "-"),
            clean.split("::", 1)[-1],
        }
        if any(alias and alias in query_lower for alias in aliases):
            return 1.0
        field_terms = set(cls._tokenize(clean))
        if not field_terms:
            return 0.0
        overlap = field_terms & set(query_terms)
        return len(overlap) / len(field_terms)

    @staticmethod
    def _source_bonus(record: MemoryRecord) -> float:
        tags = set(record.tags)
        if "preference_reinforced" in tags:
            return 0.04
        if "preference" in tags and "preference_observation" not in tags:
            return 0.02
        if "semantic_session" in tags:
            return 0.01
        if "raw_event" in tags:
            return -0.02
        return 0.0

    @staticmethod
    def _select_session_diverse(
        scored: list[tuple[float, int, MemoryRecord]], limit: int
    ) -> list[MemoryRecord]:
        selected: list[MemoryRecord] = []
        deferred: list[MemoryRecord] = []
        per_session: dict[str, int] = {}
        for _, _, record in scored:
            session_id = str(record.metadata.get("source_session_id", "")).strip()
            if session_id and per_session.get(session_id, 0) >= 2:
                deferred.append(record)
                continue
            selected.append(record)
            if session_id:
                per_session[session_id] = per_session.get(session_id, 0) + 1
            if len(selected) == limit:
                return selected
        selected.extend(deferred[: max(limit - len(selected), 0)])
        return selected[:limit]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return tokenize_search_text(text)

    @staticmethod
    def _float(value: object, default: float) -> float:
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default


@dataclass(slots=True)
class MemoryActionTrace:
    """One policy decision suitable for later supervised/RL training."""

    action: str
    arguments: dict[str, Any]
    ok: bool
    message: str
    record_ids: list[str] = field(default_factory=list)
    policy: str = "agent"
    reward: float | None = None
    trace_id: str = field(default_factory=lambda: f"memtrace-{uuid4().hex[:12]}")
    created_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemoryActionTraceStore:
    """Append-only JSONL action traces; disabled unless explicitly configured."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def append(
        self,
        action: str,
        arguments: dict[str, Any],
        result: MemoryOperationResult,
        *,
        policy: str = "agent",
    ) -> MemoryActionTrace:
        trace = MemoryActionTrace(
            action=action,
            arguments=arguments,
            ok=result.ok,
            message=result.message,
            record_ids=[record.memory_id for record in result.records],
            policy=policy,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(trace.to_dict(), ensure_ascii=False, default=str) + "\n"
            )
        return trace
