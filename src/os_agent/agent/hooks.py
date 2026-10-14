"""Agent loop extension hooks."""

from __future__ import annotations

from dataclasses import dataclass

from os_agent.events.schemas import AgentRequest, ToolCall, ToolResult


@dataclass(slots=True)
class BeforeToolCallDecision:
    """Decision returned before a tool call is executed."""

    block: bool = False
    reason: str = ""


class AgentHook:
    """Synchronous extension points around agent loop execution."""

    def before_tool_call(
        self,
        request: AgentRequest,
        call: ToolCall,
        turn_index: int,
    ) -> BeforeToolCallDecision | None:
        return None

    def after_tool_result(
        self,
        request: AgentRequest,
        call: ToolCall,
        result: ToolResult,
        turn_index: int,
    ) -> ToolResult | None:
        return None
