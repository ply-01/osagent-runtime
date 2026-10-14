"""JSONL event logger for phase-one observability."""

from __future__ import annotations

import json
from pathlib import Path

from os_agent.events.schemas import AgentEvent, utc_now_iso


class EventLogger:
    """Append structured events to per-day JSONL files."""

    def __init__(self, root: Path | str = "data/events") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def append(self, event: AgentEvent) -> Path:
        day = utc_now_iso()[:10]
        path = self.root / f"{day}.jsonl"
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        return path
