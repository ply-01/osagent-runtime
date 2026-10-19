"""Mock OS tools used before real desktop automation is introduced."""

from __future__ import annotations

from typing import Any

from os_agent.events.schemas import ToolResult
from os_agent.tools.base import Tool


class MockAppStateTool(Tool):
    name = "mock_os.get_app_state"
    description = "Return mock desktop/app state for demos."

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        def action() -> dict[str, Any]:
            return {
                "active_app": arguments.get("app", "workspace"),
                "platform": "mock-desktop",
                "windows": ["terminal", "file-manager", "browser"],
            }

        return self._timed_result(arguments, action)


class MockOpenAppTool(Tool):
    name = "mock_os.open_app"
    description = "Pretend to open an application."

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        def action() -> dict[str, Any]:
            app = str(arguments.get("app", "")).strip()
            if not app:
                raise ValueError("app is required")
            return {"app": app, "opened": True}

        return self._timed_result(arguments, action)
