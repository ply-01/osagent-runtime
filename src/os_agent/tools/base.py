"""Tool protocol and helper base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any

from os_agent.events.schemas import ToolResult


class Tool(ABC):
    """Base class for all tools exposed to the agent loop."""

    name: str
    description: str

    @abstractmethod
    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute the tool with validated-ish arguments."""

    def _timed_result(
        self,
        arguments: dict[str, Any],
        fn,
    ) -> ToolResult:
        start = perf_counter()
        try:
            output = fn()
            status = "success"
            error = None
        except Exception as exc:  # noqa: BLE001 - tools must normalize failures.
            output = {}
            status = "error"
            error = str(exc)
        duration_ms = int((perf_counter() - start) * 1000)
        return ToolResult(
            tool_name=self.name,
            status=status,
            input=arguments,
            output=output,
            error=error,
            duration_ms=duration_ms,
        )
