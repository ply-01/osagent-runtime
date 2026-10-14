"""Filesystem tools for the phase-one agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from os_agent.events.schemas import ToolResult
from os_agent.tools.base import Tool

# ---------------------------------------------------------------------------
# shared path resolution (workspace-scoped)
# ---------------------------------------------------------------------------


def _resolve_in_workspace(raw: Any, workspace: Path) -> Path:
    candidate = Path(str(raw))
    if not candidate.is_absolute():
        candidate = workspace / candidate
    target = candidate.resolve()
    try:
        target.relative_to(workspace)
    except ValueError as exc:
        raise PermissionError(f"path escapes workspace: {target}") from exc
    return target


class ListDirTool(Tool):
    name = "filesystem.list_dir"
    description = "List entries in a directory."

    def __init__(self, workspace: Path | str = ".") -> None:
        self.workspace = Path(workspace).resolve()

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        def action() -> dict[str, Any]:
            target = _resolve_in_workspace(arguments.get("path", "."), self.workspace)
            if not target.is_dir():
                raise NotADirectoryError(str(target))
            entries = []
            for item in sorted(target.iterdir(), key=lambda p: p.name.lower()):
                entries.append(
                    {
                        "name": item.name,
                        "path": str(item.relative_to(self.workspace)),
                        "type": "directory" if item.is_dir() else "file",
                    }
                )
            return {"path": str(target), "entries": entries}

        return self._timed_result(arguments, action)


class ReadFileTool(Tool):
    name = "filesystem.read_file"
    description = "Read a UTF-8 text file from the workspace."

    def __init__(self, workspace: Path | str = ".", max_chars: int = 12000) -> None:
        self.workspace = Path(workspace).resolve()
        self.max_chars = max_chars

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        def action() -> dict[str, Any]:
            path_arg = arguments.get("path")
            if not path_arg:
                raise ValueError("path is required")
            target = _resolve_in_workspace(path_arg, self.workspace)
            if not target.is_file():
                raise FileNotFoundError(str(target))
            text = target.read_text(encoding="utf-8", errors="replace")
            truncated = len(text) > self.max_chars
            if truncated:
                text = text[: self.max_chars]
            return {
                "path": str(target),
                "content": text,
                "truncated": truncated,
                "chars": len(text),
            }

        return self._timed_result(arguments, action)


class ReadPdfTool(Tool):
    name = "filesystem.read_pdf"
    description = (
        "Extract text from a PDF file within the workspace. "
        "Returns the full text content split by page."
    )

    def __init__(self, workspace: Path | str = ".", max_chars: int = 30000) -> None:
        self.workspace = Path(workspace).resolve()
        self.max_chars = max_chars

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        def action() -> dict[str, Any]:
            import pdfplumber

            path_arg = arguments.get("path")
            if not path_arg:
                raise ValueError("path is required")
            target = _resolve_in_workspace(path_arg, self.workspace)
            if not target.is_file():
                raise FileNotFoundError(str(target))

            pages: list[dict[str, Any]] = []
            total_chars = 0
            with pdfplumber.open(target) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    if total_chars + len(text) > self.max_chars:
                        remaining = self.max_chars - total_chars
                        if remaining > 0:
                            text = text[:remaining]
                        else:
                            break
                    pages.append({"page": i + 1, "chars": len(text), "text": text})
                    total_chars += len(text)

            return {
                "path": str(target),
                "total_pages": len(pdf.pages) if hasattr(pdf, "pages") else len(pages),
                "pages": pages,
                "chars": total_chars,
            }

        return self._timed_result(arguments, action)
