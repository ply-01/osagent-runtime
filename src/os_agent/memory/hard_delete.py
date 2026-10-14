"""Physical deletion support for memory facts and their derived artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from os_agent.memory.mem0_schemas import FactCard


@dataclass(frozen=True, slots=True)
class HardDeleteStats:
    """Counts of derived artifacts removed during one hard delete."""

    event_rows: int = 0
    cache_files: int = 0


class HardDeleteService:
    """Remove source rows and invalidate caches derived from deleted facts."""

    def __init__(self, memory_root: Path | str) -> None:
        self.memory_root = Path(memory_root).resolve()
        self.project_root = self.memory_root.parent

    def purge_derived(self, facts: Iterable[FactCard]) -> HardDeleteStats:
        fact_list = list(facts)
        if not fact_list:
            return HardDeleteStats()
        event_ids = {fact.source_event_id for fact in fact_list if fact.source_event_id}
        needles = self._content_needles(fact_list)
        event_rows = sum(
            self._purge_jsonl_directory(directory, event_ids, needles)
            for directory in (
                self.project_root / "data" / "events",
                self.project_root / "data" / "sessions",
            )
        )
        event_rows += self._purge_jsonl_file(
            self.memory_root / "raw_events.jsonl", event_ids, needles
        )
        cache_files = self._invalidate_caches()
        return HardDeleteStats(event_rows=event_rows, cache_files=cache_files)

    @staticmethod
    def _content_needles(facts: list[FactCard]) -> set[str]:
        needles: set[str] = set()
        for fact in facts:
            for value in (fact.value, fact.evidence):
                text = str(value).strip()
                # Event ids remain the primary locator. Content matching is a
                # fallback only for sufficiently specific values, avoiding
                # collateral deletion for generic values such as "true".
                if len(text) >= 12:
                    needles.add(text)
        return needles

    def _purge_jsonl_directory(
        self,
        directory: Path,
        event_ids: set[str],
        needles: set[str],
    ) -> int:
        if not directory.exists():
            return 0
        removed = 0
        for path in directory.glob("*.jsonl"):
            removed += self._purge_jsonl_file(path, event_ids, needles)
        return removed

    @staticmethod
    def _purge_jsonl_file(
        path: Path,
        event_ids: set[str],
        needles: set[str],
    ) -> int:
        if not path.exists():
            return 0
        removed = 0
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return 0
        kept: list[str] = []
        changed = False
        for line in lines:
            if not line.strip():
                continue
            should_remove = any(needle in line for needle in needles)
            if not should_remove and event_ids:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    event = {}
                should_remove = str(event.get("event_id", "")) in event_ids
            if should_remove:
                removed += 1
                changed = True
            else:
                kept.append(line)
        if changed:
            payload = "\n".join(kept)
            path.write_text(
                payload + ("\n" if payload else ""),
                encoding="utf-8",
                newline="\n",
            )
        return removed

    def _invalidate_caches(self) -> int:
        removed = 0
        vector_cache = self.project_root / "data" / "vector-cache"
        cache_paths = list(vector_cache.glob("*.json"))
        cache_paths.extend(vector_cache.glob("*.sqlite3"))
        cache_paths.extend(vector_cache.glob("*.sqlite3-wal"))
        cache_paths.extend(vector_cache.glob("*.sqlite3-shm"))
        cache_paths.extend((self.memory_root / ".paging").glob("*.json"))
        cache_paths.append(self.memory_root / ".tier_index.json")
        cache_paths.append(self.memory_root / "behavior_evidence.json")
        for path in cache_paths:
            if not path.is_file():
                continue
            try:
                path.unlink()
                removed += 1
            except OSError:
                continue
        return removed
