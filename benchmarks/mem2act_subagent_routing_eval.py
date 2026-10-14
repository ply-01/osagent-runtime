"""Measure whether the real LLM planner routes official Mem2Act queries to a subagent.

This is a routing diagnostic, not a memory-accuracy score. The planner sees each
official query and the normal OSagent tool catalog. The recording subagent returns
no evidence, so no benchmark labels or expected tool calls enter the prediction
path. Gold fields are never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from os_agent.agent.llm_planner import LLMPlanner
from os_agent.agent.loop import AgentLoop
from os_agent.events.logger import EventLogger
from os_agent.events.schemas import ToolResult
from os_agent.llm.provider import OpenAICompatibleProvider
from os_agent.memory.markdown_store import MarkdownMemoryStore
from os_agent.subagents.memory import MemorySubagentHit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    ROOT
    / "data"
    / "benchmark-sources"
    / "Mem2ActBench"
    / "toolmembench_small"
    / "qa_dataset.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "doc"
    / "evaluation"
    / "results"
    / "mem2act-subagent-routing.json"
)


class RoutingOnlyPlanner(LLMPlanner):
    """Use the real LLM for planning without spending another call on prose."""

    def respond(
        self,
        user_input: str,
        context: str,
        results: list[ToolResult],
        memory_titles: list[str],
    ) -> str:
        return "Routing evaluation completed."


class RecordingMemorySubagent:
    """Record explicit tool calls without injecting synthetic memory evidence."""

    name = "routing-eval-memory"

    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def recall(
        self,
        query: str,
        *,
        session_id: str,
    ) -> list[MemorySubagentHit]:
        self.calls.append({"query": query, "session_id": session_id})
        return []

    def observe_turn(
        self,
        *,
        session_id: str,
        user_input: str,
        assistant_output: str,
    ) -> dict[str, str]:
        return {"status": "not-persisted"}

    def render_context(self, hits: list[MemorySubagentHit]) -> str:
        return ""

    def health(self) -> dict[str, str]:
        return {"status": "healthy"}


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def read_tool_calls(event_root: Path, session_id: str) -> list[str]:
    tool_names: list[str] = []
    for event_file in event_root.glob("*.jsonl"):
        for line in event_file.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("session_id") != session_id:
                continue
            if event.get("event_type") == "tool_call":
                tool_names.append(str(event.get("payload", {}).get("tool_name", "")))
    return tool_names


def git_revision(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return completed.stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path, offset: int, limit: int) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows[offset : offset + limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    rows = load_rows(dataset, max(args.sample_offset, 0), max(args.limit, 0))
    provider = OpenAICompatibleProvider()
    if not provider.enabled:
        raise RuntimeError("An LLM API key is required for routing evaluation")

    started = time.perf_counter()
    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="osagent-mem2act-routing-") as tmp:
        temp_root = Path(tmp)
        for index, row in enumerate(rows):
            case_id = str(row.get("qa_id") or f"row-{args.sample_offset + index}")
            session_id = f"mem2act-routing-{args.sample_offset + index}"
            case_root = temp_root / f"case-{index}"
            event_root = case_root / "events"
            subagent = RecordingMemorySubagent()
            planner = RoutingOnlyPlanner(provider)
            loop = AgentLoop(
                workspace=case_root / "workspace",
                event_logger=EventLogger(event_root),
                memory_store=MarkdownMemoryStore(case_root / "memory"),
                llm_planner=planner,
                use_llm=True,
                use_mem0=False,
                max_steps=1,
                memory_subagent=subagent,
                memory_subagent_mode="tool",
            )
            case_started = time.perf_counter()
            error = ""
            try:
                loop.run(str(row.get("query", "")), session_id=session_id)
            except Exception as exc:  # noqa: BLE001 - retain every fixed-slice case
                error = f"{type(exc).__name__}: {exc}"
            latency_ms = (time.perf_counter() - case_started) * 1000
            tool_calls = read_tool_calls(event_root, session_id)
            cases.append(
                {
                    "case_id": case_id,
                    "planner_used_llm": bool(loop._last_plan_used_llm),
                    "tool_calls": tool_calls,
                    "subagent_selected": "subagent.memory_search" in tool_calls,
                    "subagent_called": bool(subagent.calls),
                    "latency_ms": round(latency_ms, 3),
                    "error": error,
                }
            )

    total = len(cases)
    latencies = [float(case["latency_ms"]) for case in cases]
    subagent_calls = sum(bool(case["subagent_called"]) for case in cases)
    subagent_selections = sum(bool(case["subagent_selected"]) for case in cases)
    local_memory_calls = sum(
        "memory.recall" in case["tool_calls"] for case in cases
    )
    no_tool_calls = sum(not case["tool_calls"] for case in cases)
    llm_plans = sum(bool(case["planner_used_llm"]) for case in cases)
    errors = sum(bool(case["error"]) for case in cases)
    source_root = dataset.parents[1]
    result = {
        "provenance": {
            "system": "OSagent AgentLoop",
            "benchmark": "Mem2ActBench subagent routing diagnostic",
            "dataset_revision": git_revision(source_root),
            "dataset_sha256": sha256(dataset),
            "sample_offset": max(args.sample_offset, 0),
            "sample_count": total,
            "evaluated_at": datetime.now(UTC).isoformat(),
            "planner_model": provider.config.model,
            "planner_endpoint_host": urlparse(provider.config.base_url).hostname,
            "labels_visible_to_agent": False,
            "result_contains_query_text": False,
            "memory_accuracy_scope": None,
            "notes": [
                "This diagnostic measures tool routing only.",
                "The recording subagent returns no evidence and persists nothing.",
                "Official gold tool calls and arguments are never read.",
            ],
        },
        "summary": {
            "cases": total,
            "llm_plan_success_rate": round(llm_plans / max(total, 1), 4),
            "subagent_selection_rate": round(
                subagent_selections / max(total, 1), 4
            ),
            "subagent_successful_call_rate": round(
                subagent_calls / max(total, 1), 4
            ),
            "local_memory_route_rate": round(local_memory_calls / max(total, 1), 4),
            "no_tool_route_rate": round(no_tool_calls / max(total, 1), 4),
            "error_rate": round(errors / max(total, 1), 4),
            "latency_ms": {
                "mean": round(statistics.fmean(latencies), 3) if latencies else 0.0,
                "p50": round(percentile(latencies, 0.50), 3),
                "p95": round(percentile(latencies, 0.95), 3),
                "max": round(max(latencies), 3) if latencies else 0.0,
            },
        },
        "cases": cases,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"output: {output}")


if __name__ == "__main__":
    main()
