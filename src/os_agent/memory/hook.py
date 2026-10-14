"""Event-to-Markdown memory hook."""

from __future__ import annotations

import re

from os_agent.events.schemas import AgentEvent, MemoryCandidate
from os_agent.memory.markdown_store import MarkdownMemoryStore
from os_agent.memory.policy import MemoryPolicy


class MemoryHook:
    """Extract first-stage candidate memories from structured events.

    Observes tool_result events and user message_start events, producing
    MemoryCandidate records that flow through MemoryPolicy into stable,
    pending, or discarded states.
    """

    # ------------------------------------------------------------------
    # preference / safety detection patterns
    # ------------------------------------------------------------------
    _PREFERENCE_CN = re.compile(
        r"(我(?:喜欢|偏好|习惯|一般|通常|总是|每次都?|希望|想要|"
        r"一直|永远|从来|向来|每次都?会|每次都?要|每次都?想))"
    )
    _PREFERENCE_EN = re.compile(
        r"\b(I\s+(?:prefer|like|always|usually|typically|want|need|hate|love)|"
        r"(?:always|usually|typically)\b)",
        re.IGNORECASE,
    )
    _SAFETY_CN = re.compile(r"(不要|别|禁止|严禁|绝不|千万[别再]|避免|当心|小心)")
    _SAFETY_EN = re.compile(
        r"\b(?:don'?t\s|never\s|forbid|prohibit|avoid|beware|warning)",
        re.IGNORECASE,
    )

    def __init__(self, store: MarkdownMemoryStore, policy: MemoryPolicy | None = None) -> None:
        self.store = store
        self.policy = policy or MemoryPolicy()

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------
    def observe(self, event: AgentEvent) -> list[MemoryCandidate]:
        """Dispatch to the right handler based on event type."""
        if event.event_type == "tool_result":
            return self._observe_tool_result(event)
        if event.event_type == "message_start":
            payload = event.payload
            if payload.get("role") == "user":
                return self._observe_user_message(event)
        return []

    # ------------------------------------------------------------------
    # tool-result rules
    # ------------------------------------------------------------------
    def _observe_tool_result(self, event: AgentEvent) -> list[MemoryCandidate]:
        payload = event.payload
        tool_name = str(payload.get("tool_name", ""))
        status = str(payload.get("status", ""))
        candidates: list[MemoryCandidate] = []

        # ---- rule 1: read_file success → knowledge ----
        if tool_name == "filesystem.read_file" and status == "success":
            output = payload.get("output", {})
            path = output.get("path", "")
            content = str(output.get("content", "")).strip()
            if path and content:
                summary = self._summarize_text(content)
                candidates.append(
                    MemoryCandidate(
                        title=f"Knowledge from {path}",
                        memory_type="knowledge",
                        content=f"File `{path}` was read. Summary: {summary}",
                        source_event_id=event.event_id,
                        confidence=0.55,
                        tags=["tool_trace", "file"],
                    )
                )

        # ---- rule 2: list_dir success → knowledge (workspace structure) ----
        if tool_name == "filesystem.list_dir" and status == "success":
            output = payload.get("output", {})
            path = output.get("path", ".")
            entries = output.get("entries", [])
            if entries:
                names = [e.get("name", "") for e in entries[:20]]
                summary = ", ".join(names)
                candidates.append(
                    MemoryCandidate(
                        title=f"Directory listing: {path}",
                        memory_type="knowledge",
                        content=f"Directory `{path}` contains: {summary}",
                        source_event_id=event.event_id,
                        confidence=0.50,
                        tags=["tool_trace", "directory"],
                    )
                )

        # ---- rule 3: shell.run_safe success → tool_history ----
        if tool_name == "shell.run_safe" and status == "success":
            output = payload.get("output", {})
            returncode = output.get("returncode", -1)
            stdout = str(output.get("stdout", "")).strip()
            stderr = str(output.get("stderr", "")).strip()
            # reconstruct the command from the original input
            command = str(payload.get("input", {}).get("command", "")).strip()
            title_cmd = command[:60] if command else tool_name
            details_parts = [f"Command `{command}` completed (rc={returncode})."]
            if stdout:
                details_parts.append(f"stdout: {self._summarize_text(stdout)}")
            if stderr:
                details_parts.append(f"stderr: {self._summarize_text(stderr)}")
            candidates.append(
                MemoryCandidate(
                    title=f"Shell: {title_cmd}",
                    memory_type="tool_history",
                    content=" ".join(details_parts),
                    source_event_id=event.event_id,
                    confidence=0.55,
                    tags=["tool_trace", "shell"],
                )
            )

        # ---- rule 4: any tool error → tool_history (failures worth remembering) ----
        if tool_name and status != "success":
            error_raw = payload.get("error")
            error = str(error_raw) if error_raw else ""
            error_summary = self._summarize_text(error) if error else "unknown error"
            candidates.append(
                MemoryCandidate(
                    title=f"Tool failure: {tool_name}",
                    memory_type="tool_history",
                    content=f"Tool `{tool_name}` failed: {error_summary}",
                    source_event_id=event.event_id,
                    confidence=0.60,
                    tags=["tool_trace", "failure"],
                )
            )

        return self._route_candidates(candidates, event.session_id)

    # ------------------------------------------------------------------
    # user-message rules
    # ------------------------------------------------------------------
    def _observe_user_message(self, event: AgentEvent) -> list[MemoryCandidate]:
        payload = event.payload
        text = str(payload.get("content", "")).strip()
        if not text:
            return []
        candidates: list[MemoryCandidate] = []

        # ---- rule 5: preference patterns in user input ----
        pref_match_cn = self._PREFERENCE_CN.search(text)
        pref_match_en = self._PREFERENCE_EN.search(text)
        if pref_match_cn or pref_match_en:
            trigger = (pref_match_cn or pref_match_en).group(0)  # type: ignore[union-attr]
            candidates.append(
                MemoryCandidate(
                    title=f"User preference: {trigger}",
                    memory_type="preference",
                    content=f"User said: \"{text[:300]}\". Detected preference marker: `{trigger}`.",
                    source_event_id=event.event_id,
                    confidence=0.55,
                    tags=["user_message", "preference"],
                )
            )

        # ---- rule 6: safety / constraint patterns in user input ----
        safety_match_cn = self._SAFETY_CN.search(text)
        safety_match_en = self._SAFETY_EN.search(text)
        if safety_match_cn or safety_match_en:
            trigger = (safety_match_cn or safety_match_en).group(0)  # type: ignore[union-attr]
            candidates.append(
                MemoryCandidate(
                    title=f"Safety constraint: {trigger}",
                    memory_type="safety",
                    content=f"User said: \"{text[:300]}\". Detected constraint marker: `{trigger}`.",
                    source_event_id=event.event_id,
                    confidence=0.55,
                    tags=["user_message", "safety"],
                )
            )

        return self._route_candidates(candidates, event.session_id)

    # ------------------------------------------------------------------
    # shared routing
    # ------------------------------------------------------------------
    def _route_candidates(
        self,
        candidates: list[MemoryCandidate],
        session_id: str,
    ) -> list[MemoryCandidate]:
        new_candidates: list[MemoryCandidate] = []
        for candidate in candidates:
            if self.store.has_candidate(candidate):
                continue
            decision = self.policy.decide(
                candidate,
                has_conflict=self.store.has_conflict(candidate),
            )
            if decision.action == "stable":
                self.store.append_stable(candidate, decision)
                new_candidates.append(candidate)
            elif decision.action == "pending":
                self.store.append_pending_confirmation(candidate, decision, session_id)
                new_candidates.append(candidate)
            # discard → skip
        return new_candidates

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _summarize_text(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return ""
        joined = " ".join(lines[:3])
        return joined[:500]
