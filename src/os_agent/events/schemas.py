"""Shared data structures for agent events."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utc_now_iso() -> str:
    """Return a stable ISO timestamp for event records."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class AgentRequest:
    """A user request entering the agent loop."""

    content: str
    session_id: str
    task_id: str = field(default_factory=lambda: f"task-{uuid4().hex[:12]}")


@dataclass(slots=True)
class AgentResponse:
    """The final response returned by the agent loop."""

    content: str
    session_id: str
    task_id: str
    steps: int
    memory_candidates: list[str] = field(default_factory=list)
    retrieved_memories: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ToolCall:
    """A normalized request to execute a tool."""

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    call_id: str = field(default_factory=lambda: f"call-{uuid4().hex[:12]}")


@dataclass(slots=True)
class ToolResult:
    """A normalized result returned from a tool."""

    tool_name: str
    status: str
    input: dict[str, Any]
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    duration_ms: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "success"


@dataclass(slots=True)
class AgentEvent:
    """A JSONL-friendly event emitted by the agent runtime."""

    event_type: str
    session_id: str
    task_id: str
    payload: dict[str, Any]
    event_id: str = field(default_factory=lambda: f"evt-{uuid4().hex}")
    timestamp: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MemoryCandidate:
    """A first-stage candidate memory written as Markdown."""

    title: str
    memory_type: str
    content: str
    source_event_id: str
    confidence: float
    status: str = "candidate"
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)
