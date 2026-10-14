"""Tool wrappers for the unified memory module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from os_agent.events.schemas import ToolResult
from os_agent.memory.interface import (
    ForgetRequest,
    RecallRequest,
    RememberRequest,
    UpdateRequest,
)
from os_agent.memory.module import AgentMemoryModule
from os_agent.tools.base import Tool


class MemoryToolBase(Tool):
    """Base class for memory module tools."""

    def __init__(self, root: Path | str = "memory", module: AgentMemoryModule | None = None) -> None:
        self.module = module or AgentMemoryModule(root)

    @staticmethod
    def _scope(arguments: dict[str, Any]) -> str:
        scope = str(arguments.get("scope", "ltm")).lower()
        if scope not in {"ltm", "stm"}:
            raise ValueError("scope must be 'ltm' or 'stm'")
        return scope

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None or value == "":
            return None
        return int(value)


class RememberTool(MemoryToolBase):
    name = "memory.remember"
    description = (
        "Store a memory in the main agent's local long-term or short-term store. "
        "When subagent.memory_manage is available, use that tool for durable "
        "cross-session specialist memory requested by the user."
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        def action() -> dict[str, Any]:
            content = str(arguments.get("content", "")).strip()
            if not content:
                raise ValueError("content is required")
            request = RememberRequest(
                scope=self._scope(arguments),  # type: ignore[arg-type]
                content=content,
                memory_id=arguments.get("memory_id"),
                entity=arguments.get("entity"),
                attribute=arguments.get("attribute"),
                value=arguments.get("value"),
                evidence=str(arguments.get("evidence", "")),
                confidence=float(arguments.get("confidence", 0.55)),
                tags=list(arguments.get("tags") or []),
                metadata=dict(arguments.get("metadata") or {}),
                source_session_id=str(arguments.get("source_session_id", "")),
                source_event_id=str(arguments.get("source_event_id", "")),
                ttl_seconds=self._optional_int(arguments.get("ttl_seconds")),
            )
            return self.module.remember(request).to_dict()

        return self._timed_result(arguments, action)


class ForgetTool(MemoryToolBase):
    name = "memory.forget"
    description = (
        "Forget or retire memory from the main agent's local store. When "
        "subagent.memory_manage is available, use it to remove durable "
        "cross-session specialist memory."
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        def action() -> dict[str, Any]:
            request = ForgetRequest(
                scope=self._scope(arguments),  # type: ignore[arg-type]
                memory_id=arguments.get("memory_id"),
                entity=arguments.get("entity"),
                attribute=arguments.get("attribute"),
                query=arguments.get("query"),
                hard=bool(arguments.get("hard", False)),
            )
            return self.module.forget(request).to_dict()

        return self._timed_result(arguments, action)


class UpdateMemoryTool(MemoryToolBase):
    name = "memory.update"
    description = (
        "Update memory in the main agent's local store. When "
        "subagent.memory_manage is available, use it to correct durable "
        "cross-session specialist memory."
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        def action() -> dict[str, Any]:
            request = UpdateRequest(
                scope=self._scope(arguments),  # type: ignore[arg-type]
                memory_id=arguments.get("memory_id"),
                content=arguments.get("content"),
                entity=arguments.get("entity"),
                attribute=arguments.get("attribute"),
                value=arguments.get("value"),
                evidence=str(arguments.get("evidence", "")),
                confidence=float(arguments.get("confidence", 0.55)),
                tags=list(arguments.get("tags") or []),
                metadata=dict(arguments.get("metadata") or {}),
                source_session_id=str(arguments.get("source_session_id", "")),
                source_event_id=str(arguments.get("source_event_id", "")),
                ttl_seconds=self._optional_int(arguments.get("ttl_seconds")),
            )
            return self.module.update(request).to_dict()

        return self._timed_result(arguments, action)


class RecallMemoryTool(MemoryToolBase):
    name = "memory.recall"
    description = (
        "Recall a direct, simple, or recent fact from OSagent's local long-term "
        "and short-term memory. When subagent.memory_search is available, do not "
        "use this local tool for cross-session history, implicit constraints, or "
        "patterns spread across earlier tool interactions."
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        def action() -> dict[str, Any]:
            query = str(arguments.get("query", "")).strip()
            if not query:
                raise ValueError("query is required")
            scope = str(arguments.get("scope", "all")).lower()
            if scope not in {"ltm", "stm", "all"}:
                raise ValueError("scope must be 'ltm', 'stm', or 'all'")
            request = RecallRequest(
                query=query,
                scope=scope,  # type: ignore[arg-type]
                limit=int(arguments.get("limit", 10)),
                entity=arguments.get("entity"),
                attribute=arguments.get("attribute"),
                fact_scope=arguments.get("fact_scope"),
                days=self._optional_int(arguments.get("days")),
                query_variants=[
                    str(value)
                    for value in arguments.get("query_variants", [])
                    if str(value).strip()
                ],
                subqueries=[
                    str(value)
                    for value in arguments.get("subqueries", [])
                    if str(value).strip()
                ],
            )
            return self.module.recall(request).to_dict()

        return self._timed_result(arguments, action)
