"""Plan validation and normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from os_agent.agent.plan import PlanStep, TaskPlan
from os_agent.events.schemas import ToolCall


@dataclass(slots=True)
class PlanIssue:
    """A validation issue found while normalizing a plan."""

    code: str
    message: str
    severity: str = "warning"
    step_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "step_index": self.step_index,
        }


@dataclass(slots=True)
class PlanValidationResult:
    """Normalized plan plus validation diagnostics."""

    plan: TaskPlan
    issues: list[PlanIssue]

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "issues": [issue.to_dict() for issue in self.issues],
            "issue_count": len(self.issues),
        }


class PlanValidator:
    """Keep planner output executable before it reaches the tool layer."""

    VALID_STATUSES = {"pending", "running", "done", "skipped", "error"}

    def validate(
        self,
        plan: TaskPlan,
        available_tools: list[str],
        max_steps: int,
        executed_keys: set[tuple[str, str]] | None = None,
    ) -> PlanValidationResult:
        issues: list[PlanIssue] = []
        allowed_tools = set(available_tools)
        executed = executed_keys or set()
        seen: set[tuple[str, str]] = set()
        normalized_steps: list[PlanStep] = []

        objective = plan.objective.strip() or "Respond to user request."
        if objective != plan.objective:
            issues.append(PlanIssue("missing_objective", "Plan objective was empty."))

        for index, step in enumerate(plan.steps):
            normalized = self._normalize_step(
                step=step,
                step_index=index,
                allowed_tools=allowed_tools,
                executed_keys=executed,
                seen_keys=seen,
                issues=issues,
            )
            if normalized is not None:
                normalized_steps.append(normalized)

        if len(normalized_steps) > max_steps:
            issues.append(
                PlanIssue(
                    code="max_steps_exceeded",
                    message=f"Plan had {len(normalized_steps)} steps; truncated to {max_steps}.",
                )
            )
            normalized_steps = normalized_steps[:max_steps]

        confidence = self._float_between(plan.confidence, 0.0, 1.0)
        if confidence != plan.confidence:
            issues.append(
                PlanIssue("confidence_clamped", "Plan confidence was outside 0.0-1.0.")
            )

        done = plan.done or not self._has_pending_tool_call(normalized_steps)
        normalized_plan = TaskPlan(
            objective=objective,
            steps=normalized_steps,
            confidence=confidence,
            done=done,
            planner=plan.planner,
        )
        return PlanValidationResult(plan=normalized_plan, issues=issues)

    def _normalize_step(
        self,
        step: PlanStep,
        step_index: int,
        allowed_tools: set[str],
        executed_keys: set[tuple[str, str]],
        seen_keys: set[tuple[str, str]],
        issues: list[PlanIssue],
    ) -> PlanStep | None:
        status = step.status
        if status not in self.VALID_STATUSES:
            issues.append(
                PlanIssue(
                    code="invalid_step_status",
                    message=f"Invalid step status {status!r}; reset to pending.",
                    step_index=step_index,
                )
            )
            status = "pending"

        tool_call = step.tool_call
        if tool_call is None:
            if status == "pending":
                issues.append(
                    PlanIssue(
                        code="non_executable_pending_step",
                        message="Pending step has no tool call; marked done.",
                        step_index=step_index,
                    )
                )
                status = "done"
            return PlanStep(
                goal=step.goal.strip() or "Respond without a tool.",
                tool_call=None,
                status=status,
                observation=step.observation,
                step_id=step.step_id,
                depends_on=step.depends_on,
            )

        if tool_call.tool_name not in allowed_tools:
            issues.append(
                PlanIssue(
                    code="unknown_tool",
                    message=f"Tool {tool_call.tool_name!r} is not available.",
                    severity="error",
                    step_index=step_index,
                )
            )
            return None

        sanitized_call = self._sanitize_tool_call(tool_call, step_index, issues)
        if sanitized_call is None:
            return None

        key = self._call_key(sanitized_call)
        if key in executed_keys:
            issues.append(
                PlanIssue(
                    code="already_executed_tool_call",
                    message="Tool call was already executed in this task.",
                    step_index=step_index,
                )
            )
            return None
        if key in seen_keys:
            issues.append(
                PlanIssue(
                    code="duplicate_tool_call",
                    message="Duplicate tool call removed from plan.",
                    step_index=step_index,
                )
            )
            return None
        seen_keys.add(key)

        return PlanStep(
            goal=step.goal.strip()
            or sanitized_call.reason
            or f"Run {sanitized_call.tool_name}.",
            tool_call=sanitized_call,
            status=status,
            observation=step.observation,
            step_id=step.step_id,
            depends_on=step.depends_on,
        )

    def _sanitize_tool_call(
        self,
        call: ToolCall,
        step_index: int,
        issues: list[PlanIssue],
    ) -> ToolCall | None:
        arguments = dict(call.arguments)
        if call.tool_name == "filesystem.list_dir":
            if not arguments.get("path"):
                arguments["path"] = "."
                issues.append(
                    PlanIssue(
                        code="defaulted_list_dir_path",
                        message="filesystem.list_dir path defaulted to '.'.",
                        step_index=step_index,
                    )
                )
        elif call.tool_name == "filesystem.read_file":
            if not arguments.get("path"):
                issues.append(
                    PlanIssue(
                        code="missing_read_file_path",
                        message="filesystem.read_file requires a path.",
                        severity="error",
                        step_index=step_index,
                    )
                )
                return None
        elif call.tool_name == "filesystem.read_pdf":
            if not arguments.get("path"):
                issues.append(
                    PlanIssue(
                        code="missing_read_pdf_path",
                        message="filesystem.read_pdf requires a path.",
                        severity="error",
                        step_index=step_index,
                    )
                )
                return None
        elif call.tool_name == "shell.run_safe":
            if not str(arguments.get("command", "")).strip():
                issues.append(
                    PlanIssue(
                        code="missing_shell_command",
                        message="shell.run_safe requires a command.",
                        severity="error",
                        step_index=step_index,
                    )
                )
                return None

        return ToolCall(
            tool_name=call.tool_name,
            arguments=arguments,
            reason=call.reason,
            call_id=call.call_id,
        )

    def _has_pending_tool_call(self, steps: list[PlanStep]) -> bool:
        return any(step.status == "pending" and step.tool_call is not None for step in steps)

    def _call_key(self, call: ToolCall) -> tuple[str, str]:
        return (call.tool_name, repr(sorted(call.arguments.items())))

    def _float_between(self, value: Any, low: float, high: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return low
        return min(max(parsed, low), high)
