"""LLM-driven action planner and responder."""

from __future__ import annotations

import json
from typing import Any

from os_agent.agent.context_compressor import ContextCompressor
from os_agent.agent.plan import PlanStep, TaskPlan
from os_agent.events.schemas import ToolCall, ToolResult
from os_agent.llm.provider import ChatMessage, LLMError, OpenAICompatibleProvider


class LLMPlanner:
    """Use an LLM to choose tool calls and compose final responses."""

    def __init__(
        self,
        provider: OpenAICompatibleProvider,
        compressor: ContextCompressor | None = None,
    ) -> None:
        self.provider = provider
        self.compressor = compressor or ContextCompressor()

    @property
    def enabled(self) -> bool:
        return self.provider.enabled

    def plan(
        self,
        user_input: str,
        context: str,
        tool_names: list[str],
        max_steps: int,
        tool_descriptions: dict[str, str] | None = None,
    ) -> list[ToolCall]:
        return self.create_plan(
            user_input,
            context,
            tool_names,
            max_steps,
            tool_descriptions=tool_descriptions,
        ).tool_calls()

    def create_plan(
        self,
        user_input: str,
        context: str,
        tool_names: list[str],
        max_steps: int,
        tool_descriptions: dict[str, str] | None = None,
    ) -> TaskPlan:
        system = (
            "You are the planner for a local OS Agent. "
            "Return JSON only. Create a concise task plan with at most the requested number of steps. "
            "Use only available tool names. Prefer no tool only for greetings or pure chat. "
            "Choose tools according to their descriptions; when tools overlap, use the one "
            "whose stated scope best matches the request. "
            "Do not read or write profile.md or files under memory/; the memory system manages them. "
            "Schema: {\"objective\":\"...\",\"confidence\":0.0,"
            "\"steps\":[{\"goal\":\"...\",\"tool_name\":\"... or null\","
            "\"arguments\":{},\"reason\":\"...\"}],\"done\":false}"
        )
        prompt = "\n".join(
            [
                f"Max plan steps: {max_steps}",
                "Available tools:",
                self._tool_catalog(tool_names, tool_descriptions),
                "",
                self._compressed_context(
                    context,
                    user_input,
                    self._config_int("context_max_chars", 12000),
                ),
                "",
                f"User input: {user_input}",
            ]
        )
        raw = self.provider.complete(
            [
                ChatMessage("system", system),
                ChatMessage("user", prompt),
            ],
            response_format="json",
        )
        payload = self._loads_json(raw)
        return self._plan_from_payload(payload, user_input, tool_names, max_steps)

    def revise_plan(
        self,
        user_input: str,
        previous_plan: TaskPlan,
        context: str,
        results: list[ToolResult],
        tool_names: list[str],
        max_steps: int,
        tool_descriptions: dict[str, str] | None = None,
    ) -> TaskPlan:
        system = (
            "You are revising a local OS Agent task plan after tool observations. "
            "Return JSON only. Add only the next useful steps. "
            "Do not repeat identical tool calls unless the observation proves a retry is useful. "
            "Use only available tool names. If the task can be answered from observations, "
            "choose any additional tool according to its description and stated scope. "
            "return done=true and no steps. Do not read or write profile.md or files under memory/; "
            "the memory system manages them. "
            "Schema: {\"objective\":\"...\",\"confidence\":0.0,"
            "\"steps\":[{\"goal\":\"...\",\"tool_name\":\"... or null\","
            "\"arguments\":{},\"reason\":\"...\"}],\"done\":false}"
        )
        prompt = json.dumps(
            {
                "max_plan_steps": max_steps,
                "available_tools": [
                    {
                        "name": name,
                        "description": (tool_descriptions or {}).get(name, ""),
                    }
                    for name in tool_names
                ],
                "user_input": user_input,
                "previous_plan": previous_plan.to_dict(),
                "runtime_context_excerpt": self._compressed_context(
                    context,
                    user_input,
                    self._config_int("replan_context_max_chars", 8000),
                ),
                "tool_results": [
                    {
                        "tool_name": result.tool_name,
                        "status": result.status,
                        "input": result.input,
                        "output": self._trim(result.output),
                        "error": result.error,
                    }
                    for result in results
                ],
            },
            ensure_ascii=False,
        )
        raw = self.provider.complete(
            [
                ChatMessage("system", system),
                ChatMessage("user", prompt),
            ],
            response_format="json",
        )
        payload = self._loads_json(raw)
        return self._plan_from_payload(payload, user_input, tool_names, max_steps)

    def should_stop(self, plan: TaskPlan, results: list[ToolResult]) -> bool:
        return plan.done or not plan.has_pending_tool_calls()

    def _plan_from_payload(
        self,
        payload: dict[str, Any],
        user_input: str,
        tool_names: list[str],
        max_steps: int,
    ) -> TaskPlan:
        raw_steps = payload.get("steps", payload.get("calls", []))
        if not isinstance(raw_steps, list):
            raise LLMError("LLM planner returned non-list steps")

        allowed = set(tool_names)
        steps: list[PlanStep] = []
        for item in raw_steps[:max_steps]:
            if not isinstance(item, dict):
                continue
            tool_name = str(item.get("tool_name", "")).strip()
            if tool_name and tool_name not in allowed:
                continue
            arguments = item.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}
            tool_call = None
            if tool_name:
                tool_call = ToolCall(
                    tool_name=tool_name,
                    arguments=arguments,
                    reason=str(item.get("reason", "")),
                )
            steps.append(
                PlanStep(
                    goal=str(item.get("goal") or item.get("reason") or tool_name or "Respond."),
                    tool_call=tool_call,
                ),
            )
        confidence = self._float_between(payload.get("confidence", 0.0), 0.0, 1.0)
        return TaskPlan(
            objective=str(payload.get("objective") or user_input),
            steps=steps,
            confidence=confidence,
            done=bool(payload.get("done", False)),
            planner="llm",
        )

    def respond(
        self,
        user_input: str,
        context: str,
        results: list[ToolResult],
        memory_titles: list[str],
    ) -> str:
        system = (
            "You are the responder for a local OS Agent. "
            "Answer in Chinese by default. Be concise and explain what was done. "
            "Do not invent tool results. Do not suggest editing profile.md or files under memory/; "
            "the memory system manages them."
        )
        tool_payload = [
            {
                "tool_name": result.tool_name,
                "status": result.status,
                "input": result.input,
                "output": self._trim(result.output),
                "error": result.error,
                "duration_ms": result.duration_ms,
            }
            for result in results
        ]
        prompt = json.dumps(
            {
                "user_input": user_input,
                "runtime_context_excerpt": self._compressed_context(
                    context,
                    user_input,
                    self._config_int("response_context_max_chars", 6000),
                ),
                "tool_results": tool_payload,
                "memory_candidates": memory_titles,
            },
            ensure_ascii=False,
        )
        return self.provider.complete(
            [
                ChatMessage("system", system),
                ChatMessage("user", prompt),
            ]
        )

    def _loads_json(self, raw: str) -> dict[str, Any]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and start < end:
                return json.loads(raw[start : end + 1])
            raise LLMError(f"LLM planner did not return JSON: {raw[:500]}") from exc

    def _trim(self, value: dict[str, Any]) -> dict[str, Any]:
        text = json.dumps(value, ensure_ascii=False, default=str)
        if len(text) <= 4000:
            return value
        return {"truncated_json": text[:4000]}

    def _float_between(self, value: Any, low: float, high: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return low
        return min(max(parsed, low), high)

    def _compressed_context(self, context: str, query: str, budget: int) -> str:
        return self.compressor.compress(context, query=query, max_chars=budget)

    def _config_int(self, name: str, default: int) -> int:
        config = getattr(self.provider, "config", None)
        try:
            return max(int(getattr(config, name)), 256)
        except (AttributeError, TypeError, ValueError):
            return default

    @staticmethod
    def _tool_catalog(
        tool_names: list[str],
        tool_descriptions: dict[str, str] | None,
    ) -> str:
        descriptions = tool_descriptions or {}
        return "\n".join(
            f"- {name}: {descriptions.get(name, '')}".rstrip()
            for name in tool_names
        )
