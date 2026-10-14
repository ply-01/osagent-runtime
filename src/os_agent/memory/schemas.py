"""Memory-specific structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from html import escape


@dataclass(slots=True)
class RetrievedMemory:
    """A memory excerpt selected for a specific task."""

    source: str
    title: str
    excerpt: str
    score: int
    retrieval_method: str = "keyword"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class MemoryPageItem:
    """A single Markdown memory section inside a paged result."""

    source: str
    title: str
    content: str
    index: int
    layer: str = "cold"
    access_count: int = 0
    last_accessed: str = ""
    relevance_score: float = 0.0
    recency_score: float = 0.0
    importance_score: float = 0.0
    priority_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class MemoryPage:
    """A bounded page of Markdown memory sections."""

    items: list[MemoryPageItem]
    page: int
    page_size: int
    total_items: int
    total_pages: int
    source: str | None = None
    layer: str | None = None
    query: str | None = None
    strategy: str = "layered"

    @property
    def has_previous(self) -> bool:
        return self.page > 1 and self.total_pages > 0

    @property
    def has_next(self) -> bool:
        return self.page < self.total_pages

    def to_dict(self) -> dict:
        return {
            "items": [item.to_dict() for item in self.items],
            "page": self.page,
            "page_size": self.page_size,
            "total_items": self.total_items,
            "total_pages": self.total_pages,
            "source": self.source,
            "layer": self.layer,
            "query": self.query,
            "strategy": self.strategy,
            "has_previous": self.has_previous,
            "has_next": self.has_next,
        }

    def as_prompt_block(self, max_chars_per_item: int = 1200) -> str:
        if not self.items:
            return ""
        header = (
            f'<markdown_memory_page page="{self.page}" page_size="{self.page_size}" '
            f'total_pages="{self.total_pages}" total_items="{self.total_items}"'
        )
        if self.source:
            header += f' source="{escape(self.source, quote=True)}"'
        if self.layer:
            header += f' layer="{escape(self.layer, quote=True)}"'
        if self.query:
            header += f' query="{escape(self.query, quote=True)}"'
        if self.strategy:
            header += f' strategy="{escape(self.strategy, quote=True)}"'
        header += ">"
        parts = [header]
        for item in self.items:
            content = item.content.strip()
            if len(content) > max_chars_per_item:
                content = content[:max_chars_per_item] + "\n...[truncated]"
            parts.extend(
                [
                    (
                        f'<item index="{item.index}" source="{escape(item.source, quote=True)}" '
                        f'title="{escape(item.title, quote=True)}" layer="{item.layer}" '
                        f'access_count="{item.access_count}" '
                        f'priority_score="{item.priority_score:.3f}" '
                        f'relevance_score="{item.relevance_score:.3f}" '
                        f'recency_score="{item.recency_score:.3f}" '
                        f'importance_score="{item.importance_score:.3f}">'
                    ),
                    content,
                    "</item>",
                ]
            )
        parts.append("</markdown_memory_page>")
        return "\n".join(parts)


@dataclass(slots=True)
class MemoryContext:
    """Markdown memory loaded into the agent context."""

    profile: str = ""
    project_knowledge: str = ""
    workflows: str = ""
    tool_history: str = ""
    safety: str = ""

    def as_prompt_block(self) -> str:
        sections = [
            ("profile", self.profile),
            ("project_knowledge", self.project_knowledge),
            ("workflows", self.workflows),
            ("tool_history", self.tool_history),
            ("safety", self.safety),
        ]
        rendered = []
        for name, text in sections:
            clean = text.strip()
            if clean:
                rendered.append(f"<{name}>\n{clean}\n</{name}>")
        return "\n\n".join(rendered)
