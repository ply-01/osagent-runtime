"""Stable memory interface contracts for agent/tool use."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

MemoryScope = Literal["ltm", "stm"]
MemoryAction = Literal[
    "remember",
    "forget",
    "update",
    "recall",
    "summary",
    "filter",
    "noop",
    "ltm.add",
    "ltm.update",
    "ltm.delete",
    "stm.retrieve",
    "stm.summary",
    "stm.filter",
]


@dataclass(slots=True)
class MemoryRecord:
    """A normalized memory record returned by the memory module."""

    memory_id: str
    scope: MemoryScope
    content: str
    entity: str = ""
    attribute: str = ""
    value: Any = None
    confidence: float = 0.55
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RememberRequest:
    """Request to store a new memory."""

    content: str
    scope: MemoryScope = "ltm"
    memory_id: str | None = None
    entity: str | None = None
    attribute: str | None = None
    value: Any = None
    evidence: str = ""
    confidence: float = 0.55
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    source_session_id: str = ""
    source_event_id: str = ""
    ttl_seconds: int | None = None


@dataclass(slots=True)
class ForgetRequest:
    """Request to remove or retire memory."""

    scope: MemoryScope = "ltm"
    memory_id: str | None = None
    entity: str | None = None
    attribute: str | None = None
    query: str | None = None
    hard: bool = False


@dataclass(slots=True)
class UpdateRequest:
    """Request to update an existing memory."""

    scope: MemoryScope = "ltm"
    memory_id: str | None = None
    content: str | None = None
    entity: str | None = None
    attribute: str | None = None
    value: Any = None
    evidence: str = ""
    confidence: float = 0.55
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    source_session_id: str = ""
    source_event_id: str = ""
    ttl_seconds: int | None = None


@dataclass(slots=True)
class RecallRequest:
    """Request to retrieve memory from one or more scopes."""

    query: str
    scope: Literal["ltm", "stm", "all"] = "all"
    limit: int = 10
    entity: str | None = None
    attribute: str | None = None
    fact_scope: str | None = None
    days: int | None = None
    query_variants: list[str] = field(default_factory=list)
    subqueries: list[str] = field(default_factory=list)
    query_embedding: list[float] | None = None


@dataclass(slots=True)
class MemoryActionRequest:
    """Unified tool-style memory action for agentic memory policies."""

    action: MemoryAction
    scope: Literal["ltm", "stm", "all"] = "ltm"
    content: str = ""
    memory_id: str | None = None
    memory_ids: list[str] = field(default_factory=list)
    entity: str | None = None
    attribute: str | None = None
    value: Any = None
    evidence: str = ""
    confidence: float = 0.55
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    source_session_id: str = ""
    source_event_id: str = ""
    ttl_seconds: int | None = None
    query: str = ""
    limit: int = 10
    fact_scope: str | None = None
    days: int | None = None
    query_variants: list[str] = field(default_factory=list)
    subqueries: list[str] = field(default_factory=list)
    query_embedding: list[float] | None = None
    summary: str = ""
    replace_sources: bool = False
    remove_irrelevant: bool = False
    hard: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MemoryOperationResult:
    """Normalized operation result for direct calls and tool calls."""

    ok: bool
    action: MemoryAction
    scope: str
    records: list[MemoryRecord] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "action": self.action,
            "scope": self.scope,
            "records": [record.to_dict() for record in self.records],
            "message": self.message,
        }
