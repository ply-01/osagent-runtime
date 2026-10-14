"""Conservative shell tool for smoke-test friendly command execution."""

from __future__ import annotations

import subprocess
import os
import re
import shlex
from pathlib import Path
from typing import Any

from os_agent.events.schemas import ToolResult
from os_agent.tools.base import Tool


class SafeShellTool(Tool):
    name = "shell.run_safe"
    description = "Run a short command from an allowlist."

    DEFAULT_ALLOWLIST = {
        "dir",
        "echo",
        "get-childitem",
        "ls",
        "pwd",
        "rg",
        "type",
        "where",
    }
    SHELL_METACHARS = re.compile(r"[;&|<>`$()\r\n]")
    FILE_COMMANDS = {"dir", "get-childitem", "ls", "rg", "type"}

    def __init__(
        self,
        cwd: str = ".",
        allowlist: set[str] | None = None,
        workspace: str | None = None,
    ) -> None:
        self.cwd = str(Path(cwd).resolve())
        self.workspace = Path(workspace or cwd).resolve()
        self.allowlist = {item.lower() for item in (allowlist or self.DEFAULT_ALLOWLIST)}
        if not self._is_within_workspace(Path(self.cwd)):
            raise ValueError("cwd must be inside workspace")

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        def action() -> dict[str, Any]:
            command = str(arguments.get("command", "")).strip()
            if not command:
                raise ValueError("command is required")
            if self.SHELL_METACHARS.search(command):
                raise PermissionError("shell metacharacters are not allowed")
            head = command.split()[0].lower()
            if head not in self.allowlist:
                raise PermissionError(f"command not allowed: {head}")
            self._validate_paths(command)
            if os.name == "nt":
                shell_command = ["powershell", "-NoProfile", "-Command", command]
            else:
                # PowerShell's file-reading `type` is shell introspection on POSIX.
                tokens = shlex.split(command)
                if head == "type":
                    shell_command = ["cat", "--", *tokens[1:]]
                elif head in {"dir", "get-childitem"}:
                    shell_command = ["ls", *tokens[1:]]
                elif head == "where":
                    shell_command = ["which", *tokens[1:]]
                else:
                    shell_command = ["/bin/sh", "-lc", command]
            completed = subprocess.run(
                shell_command,
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=int(arguments.get("timeout_seconds", 10)),
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            return {
                "returncode": completed.returncode,
                "stdout": completed.stdout[-12000:],
                "stderr": completed.stderr[-4000:],
            }

        return self._timed_result(arguments, action)

    def _validate_paths(self, command: str) -> None:
        try:
            tokens = shlex.split(command, posix=False)
        except ValueError as exc:
            raise PermissionError("invalid command quoting") from exc

        head = tokens[0].lower() if tokens else ""
        if head not in self.FILE_COMMANDS:
            return
        path_arguments = self._path_arguments(head, tokens[1:])
        for raw_token in tokens[1:]:
            token = raw_token.strip().strip('"\'')
            if not token or token.startswith("-"):
                continue
            managed_path = self._is_managed_memory_path(token)
            if managed_path and (
                self._looks_like_path(token)
                or raw_token in path_arguments
                or token.casefold() in {"profile.md", ".tier_index", ".tier_index.json"}
            ):
                raise PermissionError("memory files are managed by the agent")
            if not self._looks_like_path(token):
                continue
            candidate = Path(token)
            if not candidate.is_absolute():
                candidate = Path(self.cwd) / candidate
            if not self._is_within_workspace(candidate):
                raise PermissionError(f"path escapes workspace: {token}")

    @staticmethod
    def _path_arguments(head: str, arguments: list[str]) -> list[str]:
        positional = [token for token in arguments if not token.startswith("-")]
        if head != "rg" or "--files" in arguments:
            return positional
        # The first positional rg argument is the search pattern, not a path.
        return positional[1:]

    @staticmethod
    def _is_managed_memory_path(token: str) -> bool:
        normalized = token.strip('"\'').replace("\\", "/").casefold()
        parts = [part for part in normalized.split("/") if part not in {"", "."}]
        return (
            "memory" in parts
            or "profile.md" in parts
            or ".tier_index" in parts
            or ".tier_index.json" in parts
        )

    @staticmethod
    def _looks_like_path(token: str) -> bool:
        return (
            token in {".", ".."}
            or token.startswith(("./", ".\\", "../", "..\\", "/", "\\"))
            or "/" in token
            or "\\" in token
            or bool(re.match(r"^[A-Za-z]:", token))
        )

    def _is_within_workspace(self, candidate: Path) -> bool:
        try:
            candidate.resolve().relative_to(self.workspace)
        except (OSError, ValueError):
            return False
        return True
