"""Tool wrapper for calling specialist subagents from the main agent."""

from __future__ import annotations

from typing import Any

from os_agent.events.schemas import ToolResult
from os_agent.subagents.memory import MemorySubagent
from os_agent.tools.base import Tool


class CallMemorySubagentTool(Tool):
    """Allow main agent to explicitly call the memory subagent for complex queries."""

    name = "subagent.memory_search"
    description = (
        "Call the specialist memory subagent when the current task may depend on "
        "earlier interactions, prior tool calls, implicit user constraints, "
        "cross-session preferences, or large-scale semantic history. Prefer this "
        "over memory.recall for task-oriented requests whose required historical "
        "context is not a single explicit recent fact."
    )

    def __init__(
        self,
        subagent: MemorySubagent,
        session_id: str,
        default_query: str = "",
    ) -> None:
        self.subagent = subagent
        self.session_id = session_id
        self.default_query = default_query

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        def action() -> dict[str, Any]:
            query = str(arguments.get("query", "")).strip() or self.default_query
            if not query:
                raise ValueError("query is required")

            # Call the memory subagent
            hits = self.subagent.recall(query, session_id=self.session_id)

            # Format results
            results = []
            for hit in hits:
                results.append({
                    "memory_id": hit.memory_id,
                    "content": hit.content,
                    "kind": hit.kind,
                    "score": hit.score,
                    "metadata": hit.metadata,
                })

            return {
                "query": query,
                "subagent": self.subagent.name,
                "hits": len(hits),
                "results": results,
            }

        return self._timed_result(arguments, action)


class ManageMemorySubagentTool(Tool):
    """Let the main agent explicitly mutate the native subagent's memory."""

    name = "subagent.memory_manage"
    description = (
        "Explicitly change the specialist subagent's durable memory. Use action "
        "remember with content and optional entity/attribute/value; update with "
        "memory_id or entity/attribute plus the new content/value; forget with "
        "memory_id, entity/attribute, or query. Use this for direct user requests "
        "to remember, correct, or forget information in cross-session memory."
    )

    def __init__(self, subagent: Any, session_id: str) -> None:
        self.subagent = subagent
        self.session_id = session_id

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        def action() -> dict[str, Any]:
            operation = str(arguments.get("action", "")).strip().lower()
            if operation not in {"remember", "update", "forget"}:
                raise ValueError("action must be remember, update, or forget")
            manager = getattr(self.subagent, "manage", None)
            if not callable(manager):
                raise TypeError("memory subagent does not support explicit mutations")
            mutation_arguments = {
                key: value for key, value in arguments.items() if key != "action"
            }
            result = manager(
                operation,
                mutation_arguments,
                session_id=self.session_id,
            )
            return {
                "subagent": self.subagent.name,
                "action": operation,
                "result": result,
            }

        return self._timed_result(arguments, action)
