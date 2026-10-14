"""Factory for the first-stage built-in tools."""

from __future__ import annotations

from pathlib import Path

from os_agent.memory.module import AgentMemoryModule
from os_agent.tools.filesystem import ListDirTool, ReadFileTool, ReadPdfTool
from os_agent.tools.memory import ForgetTool, RecallMemoryTool, RememberTool, UpdateMemoryTool
from os_agent.tools.mock_os import MockAppStateTool, MockOpenAppTool
from os_agent.tools.registry import ToolRegistry
from os_agent.tools.shell import SafeShellTool


def build_default_registry(
    workspace: Path | str = ".",
    memory_root: Path | str = "memory",
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ListDirTool(workspace))
    registry.register(ReadFileTool(workspace))
    registry.register(ReadPdfTool(workspace))
    registry.register(SafeShellTool(str(Path(workspace).resolve())))
    memory_module = AgentMemoryModule(memory_root)
    registry.register(RememberTool(memory_root, memory_module))
    registry.register(ForgetTool(memory_root, memory_module))
    registry.register(UpdateMemoryTool(memory_root, memory_module))
    registry.register(RecallMemoryTool(memory_root, memory_module))
    registry.register(MockAppStateTool())
    registry.register(MockOpenAppTool())
    return registry
