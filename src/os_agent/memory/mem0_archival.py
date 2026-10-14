"""Mem0 Archival layer: raw event indexing + fact history.

Archival Memory stores the complete history — all raw events,
superseded facts, and deleted facts — indexed for deep retrieval
when Core misses.

This is analogous to Mem0's Archival Memory: comprehensive, slower,
used as fallback when Core doesn't have enough results.
"""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from os_agent.memory.mem0_schemas import FactCard, SessionDocument
from os_agent.memory.memory_config import MemoryConfig
from os_agent.memory.preference_extractor import PreferenceExtractor
from os_agent.memory.retrieval_text import term_weight, tokenize_search_text

_PREFERENCE_EXTRACTOR = PreferenceExtractor()


class ArchivalMemoryStore:
    """Raw log indexer with full-text search capability.

    Sources (auto-discovered):
      - data/events/*.jsonl     → per-day event logs
      - data/sessions/*.jsonl   → per-session trace logs

    Also stores superseded/deleted FactCards that were evicted from Core.
    """

    def __init__(
        self, root: Path | str = "memory", config: MemoryConfig | None = None
    ) -> None:
        self.root = Path(root)
        self.config = config or MemoryConfig.global_config()
        self.root.mkdir(parents=True, exist_ok=True)
        self._archive_path = self.root / "archive_facts.json"
        self._raw_event_path = self.root / "raw_events.jsonl"
        self._raw_event_cold_path = self.root / "raw_events.archive.jsonl.gz"
        self._tombstone_path = self.root / "archival_tombstones.json"
        self._facts: list[FactCard] = []  # archived facts (superseded/deleted)
        self._session_documents_cache: dict[
            tuple[int, int, int, int], list[SessionDocument]
        ] = {}
        self._load_archived_facts()
        self._deleted_event_ids, self._deleted_session_ids = self._load_tombstones()
        self._raw_event_ids = self._load_raw_event_ids()
        self._hot_event_count = self._count_raw_events(self._raw_event_path)

    # ------------------------------------------------------------------
    # Fact archiving
    # ------------------------------------------------------------------

    def archive_fact(self, fact: FactCard) -> None:
        """Store a superseded or deleted fact in the archive."""
        fact.tier = "archival"
        self._facts.append(fact)
        self._save_facts()

    def get_fact_history(self, entity: str, attribute: str) -> list[FactCard]:
        """Return all versions of a fact (active + archived), newest first."""
        results = [
            f for f in self._facts if f.entity == entity and f.attribute == attribute
        ]
        results.sort(key=lambda f: -f.version)
        return results

    def purge_fact_history(self, entity: str, attribute: str) -> list[FactCard]:
        """Physically remove every archived version for one logical fact."""
        removed = [
            fact
            for fact in self._facts
            if fact.entity == entity and fact.attribute == attribute
        ]
        if not removed:
            return []
        self._facts = [
            fact
            for fact in self._facts
            if fact.entity != entity or fact.attribute != attribute
        ]
        self._save_facts()
        return removed

    # ------------------------------------------------------------------
    # Full-text search over archived facts
    # ------------------------------------------------------------------

    def search_facts(
        self,
        query: str,
        top_k: int = 0,
        include_deleted: bool = False,
        entity_filter: str | None = None,
        tag_filter: list[str] | None = None,
        attribute_filter: str | None = None,
        scope_filter: str | None = None,
        days_filter: int | None = None,
    ) -> list[FactCard]:
        """Keyword + recency search over archived facts."""
        if top_k <= 0:
            top_k = int(self.config.archival_search_facts_top_k)
        terms = self._tokenize(query)
        if not terms:
            return []
        scored: list[tuple[float, FactCard]] = []
        recency_window = max(float(self.config.archival_recency_window_days), 0.001)
        recency_floor = float(self.config.archival_recency_floor)
        kw_weight = float(self.config.archival_composite_keyword_weight)
        recency_weight = float(self.config.archival_composite_recency_weight)
        for fact in self._facts:
            if fact.status == "deleted" and not include_deleted:
                continue
            if entity_filter and fact.entity != entity_filter:
                continue
            if tag_filter and not all(tag in fact.tags for tag in tag_filter):
                continue
            if attribute_filter and fact.attribute != attribute_filter:
                continue
            if scope_filter and not (
                fact.scope == scope_filter
                or (not fact.scope and fact.entity.endswith(f"::{scope_filter}"))
            ):
                continue
            if (
                days_filter is not None
                and self._days_since(fact.updated_at) > days_filter
            ):
                continue
            text = f"{fact.entity} {fact.attribute} {fact.value} {fact.evidence[:300]}"
            text_terms = set(self._tokenize(text))
            keyword_score = sum(
                self._term_weight(term) for term in terms if term in text_terms
            )
            if keyword_score <= 0:
                continue
            recency = self._days_since(fact.updated_at)
            recency_factor = max(recency_floor, 1.0 - recency / recency_window)
            composite = keyword_score * kw_weight + recency_factor * recency_weight
            scored.append((composite, fact))
        scored.sort(key=lambda x: -x[0])
        return [fact for _, fact in scored[:top_k]]

    # ------------------------------------------------------------------
    # Raw event search
    # ------------------------------------------------------------------

    @staticmethod
    def event_memory_id(event_id: str) -> str:
        return f"event:{event_id}"

    @staticmethod
    def session_memory_id(session_id: str) -> str:
        return f"session:{session_id}"

    @classmethod
    def event_identifier(cls, event: dict[str, Any]) -> str:
        return str(event.get("event_id", "")) or cls._event_fingerprint(event)

    def retire_memory_id(self, memory_id: str) -> str | None:
        """Persist a soft-delete tombstone for one recalled archival record."""
        if memory_id.startswith("event:"):
            event_id = memory_id.removeprefix("event:").strip()
            if (
                not event_id
                or event_id not in self._raw_event_ids
                or event_id in self._deleted_event_ids
            ):
                return None
            self._deleted_event_ids.add(event_id)
        elif memory_id.startswith("session:"):
            session_id = memory_id.removeprefix("session:").strip()
            if (
                not session_id
                or session_id in self._deleted_session_ids
                or not self._session_exists(session_id)
            ):
                return None
            self._deleted_session_ids.add(session_id)
        else:
            return None
        self._save_tombstones()
        self.invalidate_session_documents()
        return memory_id

    def retire_matching_raw_events(
        self,
        query: str,
        *,
        limit: int = 1000,
    ) -> list[str]:
        """Soft-delete raw events explicitly selected by a forget query."""
        matches = self.search_raw_events(query, top_k=max(limit, 1))
        retired: list[str] = []
        for match in matches:
            event = match["event"]
            event_id = self.event_identifier(event)
            self._deleted_event_ids.add(event_id)
            retired.append(self.event_memory_id(event_id))
        if retired:
            self._save_tombstones()
            self.invalidate_session_documents()
        return retired

    def append_raw_events(self, events: list[dict[str, Any]]) -> int:
        """Persist previously unseen events for complete deep retrieval."""
        rows: list[str] = []
        for event in events:
            event_id = str(event.get("event_id", "")).strip()
            dedup_key = event_id or self._event_fingerprint(event)
            if dedup_key in self._raw_event_ids:
                continue
            self._raw_event_ids.add(dedup_key)
            rows.append(json.dumps(event, ensure_ascii=False, default=str))
        if not rows:
            return 0
        with self._raw_event_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(rows) + "\n")
        self._hot_event_count += len(rows)
        self._compact_raw_events_if_needed()
        self.invalidate_session_documents()
        return len(rows)

    def _compact_raw_events_if_needed(self) -> int:
        """Move old hot events to a gzip cold archive without losing provenance."""
        limit = max(int(self.config.archival_raw_event_hot_max_events), 1)
        if self._hot_event_count <= limit:
            return 0
        try:
            events = list(self._read_json_events(self._raw_event_path))
        except OSError:
            return 0
        if len(events) <= limit:
            self._hot_event_count = len(events)
            return 0
        evicted = events[:-limit]
        retained = events[-limit:]
        if bool(self.config.archival_raw_event_archive_enabled):
            with gzip.open(self._raw_event_cold_path, "at", encoding="utf-8") as handle:
                for event in evicted:
                    handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        self._raw_event_path.write_text(
            "".join(
                json.dumps(event, ensure_ascii=False, default=str) + "\n"
                for event in retained
            ),
            encoding="utf-8",
        )
        self._hot_event_count = len(retained)
        return len(evicted)

    def search_raw_events(
        self,
        query: str,
        event_dir: str | Path | None = None,
        session_dir: str | Path | None = None,
        top_k: int = 0,
    ) -> list[dict[str, Any]]:
        """Deep full-text search over this store and explicit external logs.

        Returns matching event dicts with surrounding context.

        Session logs outside this store are opt-in. A memory instance must not
        silently read process-wide ``data/events`` or ``data/sessions`` paths,
        because those paths can belong to another user, project, or evaluation
        run and make query cost depend on unrelated session history.
        """
        if top_k <= 0:
            top_k = int(self.config.archival_search_raw_events_top_k)
        terms = self._tokenize(query)
        if not terms:
            return []

        results: list[dict[str, Any]] = []
        seen_events: set[str] = set()
        paths = [self._raw_event_path, self._raw_event_cold_path]
        for external_dir in (event_dir, session_dir):
            if external_dir is None:
                continue
            dir_path = Path(external_dir)
            if not dir_path.exists():
                continue
            paths.extend(sorted(dir_path.glob("*.jsonl")))
        for jsonl_file in paths:
            if not jsonl_file.exists():
                continue
            for event in self._read_json_events(jsonl_file):
                dedup_key = str(event.get("event_id", "")) or self._event_fingerprint(
                    event
                )
                if dedup_key in seen_events:
                    continue
                seen_events.add(dedup_key)
                if self._event_is_deleted(event, dedup_key):
                    continue
                searchable = json.dumps(event, ensure_ascii=False)
                searchable_terms = set(self._tokenize(searchable))
                score = sum(
                    self._term_weight(term)
                    for term in terms
                    if term in searchable_terms
                )
                if score > 0:
                    results.append(
                        {
                            "event": event,
                            "score": score,
                            "source_file": str(jsonl_file),
                        }
                    )

        results.sort(key=lambda x: -x["score"])
        return results[:top_k]

    def session_documents(
        self, max_chars: int = 4500, embedding_max_chars: int = 900
    ) -> list[SessionDocument]:
        """Build bounded extractive summaries without benchmark-only metadata."""
        global_max_chars = max(int(self.config.archival_global_summary_max_chars), 0)
        global_max_sessions = max(
            int(self.config.archival_global_summary_max_sessions), 0
        )
        cache_key = (
            max_chars,
            embedding_max_chars,
            global_max_chars,
            global_max_sessions,
        )
        cached = self._session_documents_cache.get(cache_key)
        if cached is not None:
            return cached
        if (
            not self._raw_event_path.exists()
            and not self._raw_event_cold_path.exists()
        ) or max_chars <= 0:
            return []
        grouped: dict[str, list[dict[str, Any]]] = {}
        seen_events: set[str] = set()
        for path in (self._raw_event_cold_path, self._raw_event_path):
            for event in self._read_json_events(path):
                dedup_key = str(event.get("event_id", "")) or self._event_fingerprint(
                    event
                )
                if dedup_key in seen_events:
                    continue
                seen_events.add(dedup_key)
                if self._event_is_deleted(event, dedup_key):
                    continue
                session_id = str(event.get("session_id", "")).strip()
                if not session_id:
                    session_id = str(event.get("task_id", "")).strip()
                if not session_id:
                    session_id = self._event_fingerprint(event)
                grouped.setdefault(session_id, []).append(event)

        documents: list[SessionDocument] = []
        for session_id, events in grouped.items():
            document = self._session_document(
                session_id,
                events,
                max_chars=max_chars,
                embedding_max_chars=embedding_max_chars,
            )
            if document is not None:
                documents.append(document)
        documents.sort(key=lambda item: (item.timestamp, item.session_id))
        global_document = self._global_session_document(
            documents,
            max_chars=global_max_chars,
            max_sessions=global_max_sessions,
        )
        if (
            global_document is not None
            and global_document.session_id not in self._deleted_session_ids
        ):
            documents.append(global_document)
        self._session_documents_cache[cache_key] = documents
        return documents

    def invalidate_session_documents(self) -> None:
        self._session_documents_cache.clear()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_facts(self) -> None:
        payload = [f.to_dict() for f in self._facts]
        self._archive_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_archived_facts(self) -> None:
        if not self._archive_path.exists():
            return
        try:
            raw = json.loads(self._archive_path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                return
        except (json.JSONDecodeError, OSError):
            return
        for item in raw:
            self._facts.append(FactCard.from_dict(item))

    def _load_tombstones(self) -> tuple[set[str], set[str]]:
        if not self._tombstone_path.exists():
            return set(), set()
        try:
            raw = json.loads(self._tombstone_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return set(), set()
        if not isinstance(raw, dict):
            return set(), set()
        return (
            {str(value) for value in raw.get("event_ids", []) if str(value)},
            {str(value) for value in raw.get("session_ids", []) if str(value)},
        )

    def _save_tombstones(self) -> None:
        payload = {
            "schema_version": 1,
            "event_ids": sorted(self._deleted_event_ids),
            "session_ids": sorted(self._deleted_session_ids),
        }
        self._tombstone_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _event_is_deleted(self, event: dict[str, Any], event_id: str) -> bool:
        session_id = str(event.get("session_id", "")).strip()
        return event_id in self._deleted_event_ids or (
            bool(session_id) and session_id in self._deleted_session_ids
        )

    def _session_exists(self, session_id: str) -> bool:
        if session_id == "__global__":
            return bool(self._raw_event_ids)
        for path in (self._raw_event_cold_path, self._raw_event_path):
            for event in self._read_json_events(path):
                if str(event.get("session_id", "")).strip() == session_id:
                    return True
        return False

    def _load_raw_event_ids(self) -> set[str]:
        identifiers: set[str] = set()
        for path in (self._raw_event_path, self._raw_event_cold_path):
            for event in self._read_json_events(path):
                identifiers.add(
                    str(event.get("event_id", "")) or self._event_fingerprint(event)
                )
        return identifiers

    @staticmethod
    def _read_json_events(path: Path):
        if not path.exists():
            return
        try:
            opener = gzip.open if path.suffix == ".gz" else Path.open
            if path.suffix == ".gz":
                handle = opener(path, "rt", encoding="utf-8", errors="replace")
            else:
                handle = opener(path, "r", encoding="utf-8", errors="replace")
            with handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event, dict):
                        yield event
        except OSError:
            return

    @classmethod
    def _count_raw_events(cls, path: Path) -> int:
        return sum(1 for _ in cls._read_json_events(path))

    @classmethod
    def _global_session_document(
        cls,
        documents: list[SessionDocument],
        *,
        max_chars: int,
        max_sessions: int,
    ) -> SessionDocument | None:
        import hashlib

        if max_chars <= 0 or len(documents) < 2 or max_sessions <= 0:
            return None
        selected = documents[-max_sessions:]
        lines = ["Global memory summary across sessions:"]
        source_ids: list[str] = []
        for document in selected:
            excerpt = cls._compact_text(document.text, 420).replace("\n", " ")
            if not excerpt:
                continue
            lines.append(f"Session {document.session_id}: {excerpt}")
            source_ids.extend(document.source_event_ids)
        text = "\n".join(lines)[:max_chars]
        if len(lines) <= 1:
            return None
        digest = hashlib.blake2b(
            "|".join(document.document_id for document in selected).encode("utf-8"),
            digest_size=12,
        ).hexdigest()
        return SessionDocument(
            document_id=f"global-session-{digest}",
            session_id="__global__",
            task_id="global-memory",
            text=text,
            embedding_text=text[: min(900, max_chars)],
            source_event_ids=tuple(dict.fromkeys(source_ids)),
            timestamp=selected[0].timestamp,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return tokenize_search_text(text)

    @staticmethod
    def _term_weight(term: str) -> float:
        return term_weight(term)

    @staticmethod
    def _event_fingerprint(event: dict[str, Any]) -> str:
        import hashlib

        payload = json.dumps(event, ensure_ascii=False, sort_keys=True, default=str)
        return (
            "raw-"
            + hashlib.blake2b(payload.encode("utf-8"), digest_size=12).hexdigest()
        )

    @classmethod
    def _session_document(
        cls,
        session_id: str,
        events: list[dict[str, Any]],
        *,
        max_chars: int,
        embedding_max_chars: int,
    ) -> SessionDocument | None:
        import hashlib

        task_ids = cls._stable_values(events, "task_id")
        user_lines: list[str] = []
        preference_user_lines: list[str] = []
        full_user_lines: list[str] = []
        assistant_lines: list[str] = []
        full_assistant_lines: list[str] = []
        tool_lines: list[str] = []
        event_ids: list[str] = []
        timestamps: list[str] = []
        for event in events:
            event_id = str(event.get("event_id", "")).strip()
            if event_id:
                event_ids.append(event_id)
            timestamp = str(event.get("timestamp", "")).strip()
            if timestamp:
                timestamps.append(timestamp)
            payload = event.get("payload", {})
            if not isinstance(payload, dict):
                continue
            event_type = str(event.get("event_type", ""))
            if event_type == "message_start":
                role = str(payload.get("role", "")).lower()
                raw_content = str(payload.get("content", ""))
                content = cls._compact_text(raw_content, 180)
                if not content:
                    continue
                if role == "user":
                    full_user_lines.append(
                        f"User: {cls._compact_text(raw_content, 900)}"
                    )
                    line = f"User: {content}"
                    if _PREFERENCE_EXTRACTOR.extract(
                        raw_content
                    ) or _PREFERENCE_EXTRACTOR.extract_observation(raw_content):
                        preference_user_lines.append(line)
                    else:
                        user_lines.append(line)
                elif role == "assistant":
                    full_assistant_lines.append(
                        f"Assistant: {cls._compact_text(raw_content, 900)}"
                    )
                    assistant_lines.append(
                        f"Assistant: {cls._compact_text(raw_content, 100)}"
                    )
                continue
            if event_type == "tool_result":
                tool_name = cls._compact_text(payload.get("tool_name", "tool"), 120)
                status = cls._compact_text(payload.get("status", ""), 40)
                detail = payload.get("error") if status != "success" else payload.get("output")
                compact_detail = cls._compact_text(detail, 120)
                tool_lines.append(
                    f"Tool {tool_name} ({status or 'unknown'}): {compact_detail}"
                )

        header = f"Session: {session_id}"
        if task_ids:
            header += f"\nTask: {', '.join(task_ids[:4])}"
        prioritized_user_lines = preference_user_lines + cls._edge_first(user_lines)
        embedding_sections = [
            header,
            cls._bounded_lines(
                prioritized_user_lines, int(embedding_max_chars * 0.90)
            ),
            cls._bounded_lines(
                cls._edge_first(assistant_lines), int(embedding_max_chars * 0.08)
            ),
            cls._bounded_lines(tool_lines, int(embedding_max_chars * 0.02)),
        ]
        embedding_text = "\n".join(
            section for section in embedding_sections if section
        ).strip()[:embedding_max_chars]
        full_sections = [
            header,
            cls._bounded_lines(full_user_lines, int(max_chars * 0.62)),
            cls._bounded_lines(full_assistant_lines, int(max_chars * 0.25)),
            cls._bounded_lines(tool_lines, int(max_chars * 0.13)),
        ]
        text = "\n".join(section for section in full_sections if section).strip()[
            :max_chars
        ]
        if text == header and not task_ids:
            return None
        digest = hashlib.blake2b(
            session_id.encode("utf-8"), digest_size=12
        ).hexdigest()
        return SessionDocument(
            document_id=f"session-{digest}",
            session_id=session_id,
            task_id=task_ids[0] if task_ids else "",
            text=text,
            embedding_text=embedding_text,
            source_event_ids=tuple(dict.fromkeys(event_ids)),
            timestamp=min(timestamps) if timestamps else "",
        )

    @staticmethod
    def _stable_values(events: list[dict[str, Any]], key: str) -> list[str]:
        return list(
            dict.fromkeys(
                value
                for event in events
                if (value := str(event.get(key, "")).strip())
            )
        )

    @staticmethod
    def _compact_text(value: object, limit: int) -> str:
        if isinstance(value, str):
            rendered = value
        else:
            try:
                rendered = json.dumps(value, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                rendered = str(value)
        return " ".join(rendered.split())[:limit]

    @staticmethod
    def _bounded_lines(lines: list[str], budget: int) -> str:
        if budget <= 0 or not lines:
            return ""
        selected: list[str] = []
        used = 0
        for line in lines:
            remaining = budget - used
            if remaining <= 0:
                break
            selected.append(line[:remaining])
            used += len(selected[-1]) + 1
        return "\n".join(selected)

    @staticmethod
    def _edge_first(lines: list[str]) -> list[str]:
        if len(lines) <= 2:
            return lines
        return [lines[0], lines[-1], *lines[1:-1]]

    @staticmethod
    def _days_since(iso_timestamp: str) -> float:
        if not iso_timestamp:
            return 999.0
        try:
            dt = datetime.fromisoformat(iso_timestamp)
            return max((datetime.now(UTC) - dt).total_seconds() / 86400.0, 0.0)
        except (ValueError, TypeError):
            return 999.0
