"""Build runtime context for the agent loop."""

from __future__ import annotations

from os_agent.memory.markdown_store import MarkdownMemoryStore
from os_agent.skills.registry import Skill


class ContextBuilder:
    """Load Markdown memory and tool hints into a compact context block."""

    def __init__(self, memory_store: MarkdownMemoryStore) -> None:
        self.memory_store = memory_store

    def build(
        self,
        tool_names: list[str],
        skills: list[Skill] | None = None,
        query: str = "",
        memory_page: int = 1,
        memory_page_size: int = 20,
        skip_markdown: bool = False,
    ) -> str:
        memory_block = ""
        retrieved = []
        if not skip_markdown:
            memory_block = self.memory_store.page_memory(
                page=memory_page,
                page_size=memory_page_size,
                query=query,
                strategy="priority",
            ).as_prompt_block()
            retrieved = self.memory_store.retrieve_hybrid(query)
        tools = "\n".join(f"- {name}" for name in tool_names)
        parts = [
            "# Runtime Context",
            "Available tools:",
            tools,
        ]
        if skills:
            skill_lines = []
            for skill in skills:
                tools_text = ", ".join(skill.tools) if skill.tools else "(no direct tools)"
                skill_lines.append(f"- {skill.name}: {skill.description} [{tools_text}]")
            parts.extend(["", "Available skills:", "\n".join(skill_lines)])
        if retrieved:
            memory_lines = [
                (
                    f"- {item.title} ({item.source}, {item.retrieval_method}, "
                    f"score={item.score}): {item.excerpt}"
                )
                for item in retrieved
            ]
            parts.extend(["", "# Retrieved Memory", "\n".join(memory_lines)])
        if memory_block:
            parts.extend(["", "# Markdown Memory Page", memory_block])
        return "\n".join(parts)
