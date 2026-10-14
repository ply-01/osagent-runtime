"""Per-session JSONL event persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

from os_agent.events.schemas import AgentEvent


@dataclass(slots=True)
class SessionSummary:
    """Small index record for a persisted session trace."""

    session_id: str
    path: Path
    events: int
    first_timestamp: str = ""
    last_timestamp: str = ""
    last_event_type: str = ""


class SessionRecorder:
    """Append agent events to one JSONL file per session."""

    def __init__(self, root: Path | str = "data/sessions") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def append(self, event: AgentEvent) -> Path:
        path = self.path_for(event.session_id)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        return path

    def read(self, session_id: str) -> list[dict[str, Any]]:
        path = self.path_for(session_id)
        if not path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
        return events

    def list(self) -> list[SessionSummary]:
        summaries: list[SessionSummary] = []
        for path in sorted(self.root.glob("*.jsonl")):
            events = self._read_file(path)
            if not events:
                continue
            first = events[0]
            last = events[-1]
            summaries.append(
                SessionSummary(
                    session_id=str(first.get("session_id") or path.stem),
                    path=path,
                    events=len(events),
                    first_timestamp=str(first.get("timestamp") or ""),
                    last_timestamp=str(last.get("timestamp") or ""),
                    last_event_type=str(last.get("event_type") or ""),
                )
            )
        return summaries

    def path_for(self, session_id: str) -> Path:
        return self.root / f"{self._safe_session_id(session_id)}.jsonl"

    def _read_file(self, path: Path) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events

    def _safe_session_id(self, session_id: str) -> str:
        safe = "".join(
            char if char.isalnum() or char in "-_." else "_" for char in session_id
        ).strip("._")
        if not safe:
            return sha1(session_id.encode("utf-8")).hexdigest()[:16]
        if len(safe) <= 120:
            return safe
        digest = sha1(session_id.encode("utf-8")).hexdigest()[:10]
        return f"{safe[:109]}-{digest}"
