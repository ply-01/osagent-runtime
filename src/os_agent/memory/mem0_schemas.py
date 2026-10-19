"""Mem0-style data structures: FactCard, CoreSnapshot, ArchivalEntry."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class FactCard:
    """A structured, versioned fact stored in Core memory.

    Mimics Mem0's Core layer: each card is an independent fact
    (entity + attribute + value) with full provenance.
    """

    entity: str          # 主体，如 "用户"、"os-agent项目"、"操作系统"
    attribute: str       # 属性，如 "输出格式偏好"、"预算上限"、"Python版本"
    value: Any           # 值，如 "PDF"、800、"3.11+"
    evidence: str        # 原始证据文本（对话片段或工具结果摘要）
    source_session_id: str
    source_event_id: str
    confidence: float = 0.55

    fact_id: str = field(default_factory=lambda: f"fact-{uuid4().hex[:12]}")
    version: int = 1
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)
    last_accessed_at: str = ""
    access_count: int = 0
    status: str = "active"                # active | superseded | deleted
    tags: list[str] = field(default_factory=list)
    scope: str = ""
    intent: str = ""
    certainty: str = "certain"
    temporal_status: str = "stable"
    supersedes: str | None = None         # fact_id of previous version
    superseded_by: str | None = None      # fact_id of newer version (for archive)

    # -- routing metadata (for Mem0-style tiering) --
    tier: str = "core"                    # core | archival | recall (Mem0 layers)
    importance: float = 0.5               # 0-1, computed by pipeline
    recency_boost: float = 1.0            # time-decay multiplier
    search_score: float = 0.0             # transient search score, not persisted
    semantic_score: float = 0.0           # transient cosine similarity, not persisted

    # -- custom metadata for external integrations (e.g., PrecisionMemBench) --
    custom_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "entity": self.entity,
            "attribute": self.attribute,
            "value": self.value,
            "evidence": self.evidence[:500],
            "source_session_id": self.source_session_id,
            "source_event_id": self.source_event_id,
            "confidence": self.confidence,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_accessed_at": self.last_accessed_at,
            "access_count": self.access_count,
            "status": self.status,
            "tags": self.tags,
            "scope": self.scope,
            "intent": self.intent,
            "certainty": self.certainty,
            "temporal_status": self.temporal_status,
            "supersedes": self.supersedes,
            "superseded_by": self.superseded_by,
            "tier": self.tier,
            "importance": self.importance,
            "recency_boost": self.recency_boost,
            "custom_metadata": self.custom_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FactCard:
        return cls(
            entity=data["entity"],
            attribute=data["attribute"],
            value=data.get("value"),
            evidence=data.get("evidence", ""),
            source_session_id=data.get("source_session_id", ""),
            source_event_id=data.get("source_event_id", ""),
            confidence=data.get("confidence", 0.55),
            fact_id=data.get("fact_id", f"fact-{uuid4().hex[:12]}"),
            version=data.get("version", 1),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            last_accessed_at=data.get("last_accessed_at", ""),
            access_count=data.get("access_count", 0),
            status=data.get("status", "active"),
            tags=data.get("tags", []),
            scope=data.get("scope", ""),
            intent=data.get("intent", ""),
            certainty=data.get("certainty", "certain"),
            temporal_status=data.get("temporal_status", "stable"),
            supersedes=data.get("supersedes"),
            superseded_by=data.get("superseded_by"),
            tier=data.get("tier", "core"),
            importance=data.get("importance", 0.5),
            recency_boost=data.get("recency_boost", 1.0),
            custom_metadata=data.get("custom_metadata", {}),
        )

    @property
    def key(self) -> tuple[str, str]:
        """Unique logical key: (entity, attribute)."""
        return (self.entity, self.attribute)

    def mark_accessed(self) -> None:
        self.access_count += 1
        self.last_accessed_at = _utc_now_iso()

    def supersede(self, new_fact_id: str) -> FactCard:
        """Return a copy of this card marked as superseded."""
        import copy
        card = copy.copy(self)
        card.status = "superseded"
        card.superseded_by = new_fact_id
        card.tier = "archival"
        return card


@dataclass(slots=True)
class CoreSnapshot:
    """A serializable snapshot of all active Core facts."""

    facts: list[dict[str, Any]]
    exported_at: str = field(default_factory=_utc_now_iso)
    total_active: int = 0
    entity_index: dict[str, list[str]] = field(default_factory=dict)  # entity → [fact_id, ...]


@dataclass(frozen=True, slots=True)
class SessionDocument:
    """Deterministic extractive summary embedded in the shared vector space."""

    document_id: str
    session_id: str
    task_id: str
    text: str
    embedding_text: str = ""
    source_event_ids: tuple[str, ...] = ()
    timestamp: str = ""


@dataclass(slots=True)
class RecallResult:
    """Unified retrieval result from the Mem0-style recall strategy."""

    facts: list[FactCard]
    source_layer: str          # "core" | "archival" | "core+archival"
    query: str
    total_core_hits: int = 0
    total_archival_hits: int = 0
    latency_ms: float = 0.0

    def to_context_block(self) -> str:
        """Render as a context block for the LLM planner."""
        if not self.facts:
            return ""
        lines = [f'<mem0_recall source="{self.source_layer}" count="{len(self.facts)}">']
        for fact in self.facts:
            lines.append(
                f"  - [{fact.entity}] {fact.attribute} = {fact.value}"
                f"  (v{fact.version}, conf={fact.confidence:.2f})"
            )
        lines.append("</mem0_recall>")
        return "\n".join(lines)
