"""Short-term memory store for transient agent notes."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from os_agent.memory.interface import MemoryRecord


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class ShortTermEntry:
    """A transient memory entry with optional expiration."""

    memory_id: str
    content: str
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)
    expires_at: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShortTermEntry":
        return cls(
            memory_id=str(data.get("memory_id", f"stm-{uuid4().hex[:12]}")),
            content=str(data.get("content", "")),
            created_at=str(data.get("created_at", "")) or _utc_now_iso(),
            updated_at=str(data.get("updated_at", "")) or _utc_now_iso(),
            expires_at=str(data.get("expires_at", "")),
            tags=list(data.get("tags") or []),
            metadata=dict(data.get("metadata") or {}),
        )

    def expired(self) -> bool:
        if not self.expires_at:
            return False
        try:
            return datetime.now(timezone.utc) >= datetime.fromisoformat(self.expires_at)
        except ValueError:
            return False


class ShortTermMemoryStore:
    """Small JSON-backed STM store.

    STM is intentionally simple and tool-friendly: remember, update, forget,
    and search by token overlap. Expired entries are pruned on each operation.
    """

    def __init__(self, root: Path | str = "memory") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._path = self.root / "stm.json"
        self._entries: dict[str, ShortTermEntry] = {}
        self._load()
        self._prune_expired()

    def remember(
        self,
        content: str,
        *,
        memory_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        ttl_seconds: int | None = None,
    ) -> ShortTermEntry:
        entry_id = memory_id or f"stm-{uuid4().hex[:12]}"
        expires_at = ""
        if ttl_seconds and ttl_seconds > 0:
            expires_at = (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat()
        entry = ShortTermEntry(
            memory_id=entry_id,
            content=content,
            expires_at=expires_at,
            tags=tags or [],
            metadata=metadata or {},
        )
        self._entries[entry.memory_id] = entry
        self._save()
        return entry

    def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        ttl_seconds: int | None = None,
    ) -> ShortTermEntry | None:
        self._prune_expired()
        entry = self._entries.get(memory_id)
        if entry is None:
            return None
        if content is not None:
            entry.content = content
        if tags is not None:
            entry.tags = tags
        if metadata is not None:
            entry.metadata = metadata
        if ttl_seconds is not None:
            entry.expires_at = ""
            if ttl_seconds > 0:
                entry.expires_at = (
                    datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
                ).isoformat()
        entry.updated_at = _utc_now_iso()
        self._save()
        return entry

    def forget(self, *, memory_id: str | None = None, query: str | None = None) -> list[ShortTermEntry]:
        self._prune_expired()
        removed: list[ShortTermEntry] = []
        if memory_id:
            entry = self._entries.pop(memory_id, None)
            if entry:
                removed.append(entry)
        elif query:
            matches = self.search(query, limit=1)
            for record in matches:
                entry = self._entries.pop(record.memory_id, None)
                if entry:
                    removed.append(entry)
        if removed:
            self._save()
        return removed

    def search(self, query: str, limit: int = 10) -> list[MemoryRecord]:
        self._prune_expired()
        terms = self._tokenize(query)
        if not terms:
            return []
        scored: list[tuple[int, ShortTermEntry]] = []
        for entry in self._entries.values():
            haystack = f"{entry.content} {' '.join(entry.tags)}".lower()
            score = sum(haystack.count(term) for term in terms)
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda item: (-item[0], item[1].updated_at, item[1].memory_id))
        return [self._to_record(entry) for _, entry in scored[: max(limit, 0)]]

    def get(self, memory_id: str) -> MemoryRecord | None:
        self._prune_expired()
        entry = self._entries.get(memory_id)
        return self._to_record(entry) if entry else None

    def list(self) -> list[MemoryRecord]:
        self._prune_expired()
        entries = sorted(self._entries.values(), key=lambda item: item.updated_at, reverse=True)
        return [self._to_record(entry) for entry in entries]

    def summarize(
        self,
        *,
        summary: str = "",
        memory_ids: list[str] | None = None,
        query: str = "",
        max_chars: int = 1200,
        replace_sources: bool = False,
    ) -> ShortTermEntry | None:
        """Compress selected STM entries into one traceable summary entry."""
        self._prune_expired()
        selected = self._select_entries(memory_ids=memory_ids, query=query)
        if not selected and not summary.strip():
            return None
        source_ids = [entry.memory_id for entry in selected]
        content = summary.strip() or self._extractive_summary(selected, max_chars=max_chars)
        if not content:
            return None
        entry = ShortTermEntry(
            memory_id=f"stm-summary-{uuid4().hex[:12]}",
            content=content[: max(max_chars, 1)],
            tags=["summary"],
            metadata={"source_memory_ids": source_ids, "summary_method": "agent" if summary.strip() else "extractive"},
        )
        if replace_sources:
            for memory_id in source_ids:
                self._entries.pop(memory_id, None)
        self._entries[entry.memory_id] = entry
        self._save()
        return entry

    def filter(
        self,
        query: str,
        *,
        limit: int = 10,
        memory_ids: list[str] | None = None,
        remove_irrelevant: bool = False,
    ) -> tuple[list[MemoryRecord], list[MemoryRecord]]:
        """Keep relevant STM entries and optionally remove the rest."""
        self._prune_expired()
        if memory_ids:
            keep_ids = {memory_id for memory_id in memory_ids if memory_id in self._entries}
            kept = [self._to_record(self._entries[memory_id]) for memory_id in memory_ids if memory_id in keep_ids]
        else:
            kept = self.search(query, limit=limit)
            keep_ids = {record.memory_id for record in kept}
        irrelevant = [
            self._to_record(entry)
            for memory_id, entry in self._entries.items()
            if memory_id not in keep_ids
        ]
        removed: list[MemoryRecord] = []
        if remove_irrelevant:
            removed = irrelevant
            for record in removed:
                self._entries.pop(record.memory_id, None)
            if removed:
                self._save()
        return kept, removed

    def _to_record(self, entry: ShortTermEntry) -> MemoryRecord:
        return MemoryRecord(
            memory_id=entry.memory_id,
            scope="stm",
            content=entry.content,
            value=entry.content,
            tags=entry.tags,
            metadata={
                **entry.metadata,
                "created_at": entry.created_at,
                "updated_at": entry.updated_at,
                "expires_at": entry.expires_at,
            },
        )

    def _select_entries(
        self,
        *,
        memory_ids: list[str] | None,
        query: str,
    ) -> list[ShortTermEntry]:
        if memory_ids:
            return [self._entries[memory_id] for memory_id in memory_ids if memory_id in self._entries]
        if query.strip():
            return [
                self._entries[record.memory_id]
                for record in self.search(query, limit=max(len(self._entries), 1))
                if record.memory_id in self._entries
            ]
        return sorted(self._entries.values(), key=lambda item: item.updated_at)

    @staticmethod
    def _extractive_summary(entries: list[ShortTermEntry], max_chars: int) -> str:
        parts = [f"- {entry.content.strip()}" for entry in entries if entry.content.strip()]
        return "\n".join(parts)[: max(max_chars, 1)]

    def _prune_expired(self) -> None:
        expired = [key for key, entry in self._entries.items() if entry.expired()]
        if not expired:
            return
        for key in expired:
            del self._entries[key]
        self._save()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(raw, list):
            return
        for item in raw:
            if isinstance(item, dict):
                entry = ShortTermEntry.from_dict(item)
                self._entries[entry.memory_id] = entry

    def _save(self) -> None:
        payload = [entry.to_dict() for entry in self._entries.values()]
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        ascii_terms = re.findall(r"[a-z0-9_./\\-]{2,}", text.lower())
        cjk_terms = re.findall(r"[一-鿿]{2,}", text)
        return sorted(set(ascii_terms + cjk_terms))
