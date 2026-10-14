"""Automatic memory maintenance service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from os_agent.memory.markdown_store import MarkdownMemoryStore


@dataclass(slots=True)
class MaintenanceReport:
    """Summary of a memory maintenance pass."""

    compacted_files: int
    removed_duplicates: int
    archived_sections: int
    pending_items: int


class MemoryMaintenanceService:
    """Run automatic memory compaction and pending-memory inspection."""

    def __init__(self, root: Path | str = "memory") -> None:
        self.store = MarkdownMemoryStore(root)

    def maintain(self, max_sections_per_file: int = 80) -> MaintenanceReport:
        compacted = self.store.compact_stable_files(max_sections_per_file=max_sections_per_file)
        return MaintenanceReport(
            compacted_files=compacted["compacted_files"],
            removed_duplicates=compacted["removed_duplicates"],
            archived_sections=compacted["archived_sections"],
            pending_items=len(self.store.list_pending_confirmations()),
        )

    def list_pending(self) -> list[Path]:
        return self.store.list_pending_confirmations()
