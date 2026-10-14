"""Automatic memory governance policy for stage one."""

from __future__ import annotations

from dataclasses import dataclass

from os_agent.events.schemas import MemoryCandidate


@dataclass(slots=True)
class MemoryDecision:
    """The policy decision for a candidate memory."""

    action: str
    target: str | None
    reason: str

    @property
    def stores_memory(self) -> bool:
        return self.action in {"stable", "pending"}


class MemoryPolicy:
    """Score, classify, and route candidate memories without human review."""

    TARGETS = {
        "preference": "profile.md",
        "knowledge": "project-knowledge.md",
        "workflow": "workflows.md",
        "tool_history": "tool-history.md",
        "safety": "safety.md",
    }

    HIGH_RISK_TYPES: set[str] = set()  # conflicts handled by has_conflict() now
    LOW_VALUE_TAGS = {"tool_trace"}

    def decide(self, candidate: MemoryCandidate, has_conflict: bool = False) -> MemoryDecision:
        if has_conflict:
            return MemoryDecision(
                action="pending",
                target="pending-confirmation",
                reason="possible_conflict",
            )
        if candidate.memory_type in self.HIGH_RISK_TYPES:
            return MemoryDecision(
                action="pending",
                target="pending-confirmation",
                reason=f"high_risk_{candidate.memory_type}",
            )
        if candidate.confidence < 0.50:
            return MemoryDecision(
                action="discard",
                target=None,
                reason="low_confidence",
            )
        if (
            candidate.memory_type == "tool_history"
            and set(candidate.tags).issubset(self.LOW_VALUE_TAGS)
        ):
            return MemoryDecision(
                action="discard",
                target=None,
                reason="low_value_tool_trace",
            )
        target = self.TARGETS.get(candidate.memory_type, "project-knowledge.md")
        return MemoryDecision(
            action="stable",
            target=target,
            reason="low_risk_stable_write",
        )
