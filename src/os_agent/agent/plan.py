"""Planner data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from os_agent.events.schemas import ToolCall


@dataclass(slots=True)
class PlanStep:
    """A goal-oriented step inside a task plan."""

    goal: str
    tool_call: ToolCall | None = None
    status: str = "pending"
    observation: str = ""
    step_id: str = field(default_factory=lambda: f"step-{uuid4().hex[:12]}")
    depends_on: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "goal": self.goal,
            "status": self.status,
            "observation": self.observation,
            "depends_on": self.depends_on,
            "tool_call": None
            if self.tool_call is None
            else {
                "tool_call_id": self.tool_call.call_id,
                "tool_name": self.tool_call.tool_name,
                "arguments": self.tool_call.arguments,
                "reason": self.tool_call.reason,
            },
        }


@dataclass(slots=True)
class TaskPlan:
    """A bounded plan for satisfying a user request."""

    objective: str
    steps: list[PlanStep] = field(default_factory=list)
    confidence: float = 0.0
    done: bool = False
    planner: str = "simple"

    def pending_steps(self) -> list[PlanStep]:
        return [step for step in self.steps if step.status == "pending"]

    def tool_calls(self) -> list[ToolCall]:
        return [
            step.tool_call
            for step in self.steps
            if step.status == "pending" and step.tool_call is not None
        ]

    def has_pending_tool_calls(self) -> bool:
        return bool(self.tool_calls())

    def add_steps(self, steps: list[PlanStep]) -> None:
        self.steps.extend(steps)

    def to_dict(self) -> dict:
        return {
            "objective": self.objective,
            "confidence": self.confidence,
            "done": self.done,
            "planner": self.planner,
            "steps": [step.to_dict() for step in self.steps],
        }
