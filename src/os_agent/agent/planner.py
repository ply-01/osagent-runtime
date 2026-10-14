"""Small deterministic planner for phase-one smoke workflows."""

from __future__ import annotations

import re
from pathlib import Path

from os_agent.agent.plan import PlanStep, TaskPlan
from os_agent.events.schemas import ToolCall, ToolResult


class SimplePlanner:
    """Route common natural-language requests to tool calls.

    This is intentionally small. Later phases can replace it with an LLM planner
    while keeping the ToolCall schema stable.
    """

    def plan(self, user_input: str) -> list[ToolCall]:
        return self.create_plan(user_input).tool_calls()

    def create_plan(self, user_input: str, context: str = "") -> TaskPlan:
        text = user_input.lower()
        steps: list[PlanStep] = []

        if self._asks_for_listing(text):
            steps.append(
                PlanStep(
                    goal="Inspect workspace files.",
                    tool_call=ToolCall(
                        tool_name="filesystem.list_dir",
                        arguments={"path": "."},
                        reason="User asked to inspect project files.",
                    ),
                ),
            )

        requested_file = self._extract_file_path(user_input)
        if requested_file:
            steps.extend(self._tools_for_file(requested_file, "User asked to inspect a file."))
        elif "readme" in text or "项目" in user_input or "project" in text:
            steps.append(
                PlanStep(
                    goal="Read README.md for project context.",
                    tool_call=ToolCall(
                        tool_name="filesystem.read_file",
                        arguments={"path": "README.md"},
                        reason="README is useful for project summary.",
                    ),
                ),
            )

        if "app state" in text or "桌面" in user_input or "应用状态" in user_input:
            steps.append(
                PlanStep(
                    goal="Inspect mock OS application state.",
                    tool_call=ToolCall(
                        tool_name="mock_os.get_app_state",
                        arguments={},
                        reason="User asked about OS/application state.",
                    ),
                ),
            )

        if not steps:
            steps.append(
                PlanStep(
                    goal="Produce an observable fallback state check.",
                    tool_call=ToolCall(
                        tool_name="mock_os.get_app_state",
                        arguments={"app": "workspace"},
                        reason="Fallback tool call keeps the first-stage loop observable.",
                    ),
                ),
            )

        return TaskPlan(
            objective=user_input,
            steps=steps,
            confidence=0.60 if steps else 0.0,
            planner="simple",
        )

    def revise_plan(
        self,
        user_input: str,
        plan: TaskPlan,
        results: list[ToolResult],
        executed_keys: set[tuple[str, str]],
        context: str = "",
    ) -> TaskPlan:
        recovery_steps: list[PlanStep] = []
        last_result = results[-1] if results else None
        if (
            last_result
            and not last_result.ok
            and last_result.tool_name == "filesystem.read_file"
        ):
            call = ToolCall(
                tool_name="filesystem.list_dir",
                arguments={"path": "."},
                reason="Recover from failed file read by inspecting workspace files.",
            )
            if self._call_key(call) not in executed_keys:
                recovery_steps.append(
                    PlanStep(
                        goal="Recover by listing workspace files.",
                        tool_call=call,
                    )
                )
        return TaskPlan(
            objective=plan.objective or user_input,
            steps=recovery_steps,
            confidence=0.45 if recovery_steps else plan.confidence,
            done=not recovery_steps,
            planner="simple",
        )

    def should_stop(self, plan: TaskPlan, results: list[ToolResult]) -> bool:
        return plan.done or not plan.has_pending_tool_calls()

    def _asks_for_listing(self, text: str) -> bool:
        return any(
            phrase in text
            for phrase in [
                "list",
                "files",
                "directory",
                "root",
                "目录",
                "文件",
                "根目录",
            ]
        )

    def _extract_file_path(self, user_input: str) -> str | None:
        code_match = re.search(r"`([^`]+\.(?:md|txt|py|json|toml|yml|yaml|pdf|csv|html?|xml|cfg|ini|log))`", user_input)
        if code_match:
            return code_match.group(1)
        # ASCII-only word boundary to avoid matching Chinese characters as part of the path
        plain_match = re.search(
            r"(?<![a-zA-Z0-9])"  # preceding char must not be ASCII alphanumeric
            r"([a-zA-Z0-9_./\\-]+"
            r"\.(?:md|txt|py|json|toml|yml|yaml|pdf|csv|html?|xml|cfg|ini|log))",
            user_input,
        )
        if plain_match:
            value = plain_match.group(1)
            return str(Path(value))
        return None

    @staticmethod
    def _is_pdf(path: str) -> bool:
        return path.lower().endswith(".pdf")

    def _tools_for_file(self, path: str, reason: str) -> list[PlanStep]:
        """Return the appropriate tool step(s) for a given file path."""
        if self._is_pdf(path):
            return [
                PlanStep(
                    goal=f"Extract text from PDF `{path}`.",
                    tool_call=ToolCall(
                        tool_name="filesystem.read_pdf",
                        arguments={"path": path},
                        reason=reason,
                    ),
                ),
            ]
        return [
            PlanStep(
                goal=f"Read `{path}`.",
                tool_call=ToolCall(
                    tool_name="filesystem.read_file",
                    arguments={"path": path},
                    reason=reason,
                ),
            ),
        ]

    def _call_key(self, call: ToolCall) -> tuple[str, str]:
        return (call.tool_name, repr(sorted(call.arguments.items())))
