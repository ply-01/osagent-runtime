"""Tool registry."""

from __future__ import annotations

from os_agent.events.schemas import ToolCall, ToolResult
from os_agent.tools.base import Tool


class ToolRegistry:
    """Register and execute tools by stable name."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return sorted(self._tools)

    def descriptions(self) -> dict[str, str]:
        return {
            name: str(tool.description).strip()
            for name, tool in sorted(self._tools.items())
        }

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    def execute(self, call: ToolCall) -> ToolResult:
        return self.get(call.tool_name).run(call.arguments)
