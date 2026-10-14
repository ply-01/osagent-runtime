"""Cross-session aggregation of repeated tool-parameter preferences."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, ClassVar

from os_agent.memory.mem0_schemas import FactCard


class BehaviorPreferenceAggregator:
    """Promote repeated, non-sensitive tool parameters into preference facts."""

    SENSITIVE_KEYS: ClassVar[set[str]] = {
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "credential",
        "password",
        "secret",
        "token",
    }

    def __init__(self, memory_root: Path | str, *, min_observations: int = 3) -> None:
        self.memory_root = Path(memory_root)
        self.path = self.memory_root / "behavior_evidence.json"
        self.min_observations = max(min_observations, 2)

    def observe(self, events: list[dict], session_id: str) -> list[FactCard]:
        state = self._load()
        signals: dict[str, dict[str, Any]] = state.setdefault("signals", {})
        seen: set[str] = set(state.setdefault("seen_event_ids", []))
        updated_keys: set[str] = set()
        for event in events:
            event_id = str(event.get("event_id", ""))
            if event_id and event_id in seen:
                continue
            payload = event.get("payload", {})
            if event.get("event_type") != "tool_result" or payload.get("status") != "success":
                continue
            tool_name = str(payload.get("tool_name", "")).strip()
            inputs = payload.get("input", {})
            if not tool_name or not isinstance(inputs, dict):
                continue
            for parameter, value in inputs.items():
                if not self._is_safe_scalar(parameter, value):
                    continue
                attribute = (
                    f"tool.parameter.{self._slug(tool_name)}.{self._slug(str(parameter))}"
                )
                signature = json.dumps(
                    [attribute, value], ensure_ascii=False, sort_keys=True, default=str
                )
                row = signals.setdefault(
                    signature,
                    {
                        "attribute": attribute,
                        "value": value,
                        "count": 0,
                        "last_event_id": "",
                    },
                )
                row["count"] = int(row.get("count", 0)) + 1
                row["last_event_id"] = event_id
                updated_keys.add(signature)
            if event_id:
                seen.add(event_id)
        state["seen_event_ids"] = sorted(seen)[-10000:]
        self._save(state)

        facts: list[FactCard] = []
        for signature in updated_keys:
            row = signals[signature]
            count = int(row.get("count", 0))
            if count < self.min_observations:
                continue
            confidence = min(0.6 + 0.05 * (count - self.min_observations), 0.9)
            facts.append(
                FactCard(
                    entity="用户",
                    attribute=str(row["attribute"]),
                    value=row["value"],
                    evidence=f"Observed the same tool parameter {count} times",
                    source_session_id=session_id,
                    source_event_id=str(row.get("last_event_id", "")),
                    confidence=confidence,
                    tags=["preference", "behavior", "tool_trace"],
                    importance=0.65,
                )
            )
        return facts

    @classmethod
    def _is_safe_scalar(cls, parameter: object, value: object) -> bool:
        key = str(parameter).strip().lower()
        if any(sensitive in key for sensitive in cls.SENSITIVE_KEYS):
            return False
        return isinstance(value, (str, int, float, bool)) and 0 < len(str(value)) <= 200

    @staticmethod
    def _slug(value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
        return slug[:80] or "unknown"

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"signals": {}, "seen_event_ids": []}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"signals": {}, "seen_event_ids": []}
        return raw if isinstance(raw, dict) else {"signals": {}, "seen_event_ids": []}

    def _save(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
