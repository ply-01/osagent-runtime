"""Evaluate the OS Agent memory module on the public PERMA benchmark.

This adapter intentionally keeps benchmark labels outside the ingestion and
retrieval paths.  Gold labels are read only after a prediction is produced.
The reported MCQ metric follows PERMA's multiple-choice prompt, while the
retrieval metrics are local diagnostics and are labeled as such.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import tempfile
import time
import urllib.parse
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from os_agent.llm.provider import ChatMessage, LLMError, OpenAICompatibleProvider
from os_agent.memory.embedding_provider import (
    EmbeddingConfig,
    create_embedding_provider,
)
from os_agent.memory.interface import RecallRequest
from os_agent.memory.mem0_archival import ArchivalMemoryStore
from os_agent.memory.mem0_core import CoreMemoryStore
from os_agent.memory.mem0_pipeline import ExtractUpdatePipeline
from os_agent.memory.memory_config import MemoryConfig
from os_agent.memory.module import AgentMemoryModule

DATASET_REPO = "ustclsc/PERMA"
DATASET_REVISION = "440e64e4fb8baec6f7ad10c1de135505f93e7cb1"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_REPO}"
OFFICIAL_REPO_REVISION = "d678640987170e8cfbe9260b311e0493b9cd2c31"
SUBSET_DESCRIPTION = "clean, single-domain, Type 3 (post-intervention)"
IMPLEMENTATION_FILES = (
    "src/os_agent/memory/agentic_policy.py",
    "src/os_agent/memory/embedding_cache.py",
    "src/os_agent/memory/embedding_provider.py",
    "src/os_agent/memory/fact_vector_index.py",
    "src/os_agent/memory/mem0_archival.py",
    "src/os_agent/memory/mem0_core.py",
    "src/os_agent/memory/mem0_pipeline.py",
    "src/os_agent/memory/mem0_recall.py",
    "src/os_agent/memory/mem0_schemas.py",
    "src/os_agent/memory/memory_config.py",
    "src/os_agent/memory/preference_extractor.py",
    "src/os_agent/memory/retrieval_text.py",
    "src/os_agent/memory/session_vector_index.py",
    "src/os_agent/memory/module.py",
)

ANSWER_OPTIONAL_PROMPT = """You are an assistant specialized in answering multiple-choice questions.

## Your Memory
{context}

## User Task Query
{question}

## Options:
{options}

Your goal is to choose **the most appropriate answer option for the User Task Query** from the Options based on your memory. The output should be **ONLY the option key** without any additional explanation, e.g. `A`, etc.

Your response:
"""

_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_LABEL = re.compile(r"\b([A-H])\b", re.IGNORECASE)
_OPTION = re.compile(r"(?m)^([A-H]):\s*")
_META_NAME = re.compile(r"^SD-.+_3\.json$")
_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class PermaCase:
    """One official PERMA multiple-choice case."""

    user_id: str
    task_id: str
    question: str
    question_date: str
    options: str
    gold_label: str
    source_path: str


@dataclass(slots=True)
class CaseResult:
    """Prediction and retrieval diagnostics for one case."""

    user_id: str
    task_id: str
    question_date: str
    question: str
    source_path: str
    gold_label: str
    predicted_label: str = ""
    no_memory_label: str = ""
    recall_ms: float = 0.0
    semantic_index_ms: float = 0.0
    recalled_count: int = 0
    preference_count: int = 0
    raw_event_count: int = 0
    semantic_session_count: int = 0
    gold_support_rank: int | None = None
    correct: bool | None = None
    no_memory_correct: bool | None = None
    retrieved: list[str] | None = None
    error: str = ""


def public_case_result(row: CaseResult) -> dict[str, Any]:
    """Return diagnostics without publishing prompts, labels, or recalled text."""
    case_key = f"{row.user_id}\0{row.task_id}\0{row.source_path}"
    return {
        "case_id": hashlib.sha256(case_key.encode("utf-8")).hexdigest()[:16],
        "recall_ms": row.recall_ms,
        "semantic_index_ms": row.semantic_index_ms,
        "recalled_count": row.recalled_count,
        "preference_count": row.preference_count,
        "raw_event_count": row.raw_event_count,
        "semantic_session_count": row.semantic_session_count,
        "gold_support_rank": row.gold_support_rank,
        "correct": row.correct,
        "no_memory_correct": row.no_memory_correct,
        "error_type": row.error.split(":", 1)[0] if row.error else "",
    }


def _json_request(url: str, *, timeout: int = 90) -> Any:
    request = urllib.request.Request(
        url, headers={"User-Agent": "OSagent-PERMA-eval/1.0"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class PermaDataset:
    """Minimal, revision-pinned PERMA downloader with a local cache."""

    def __init__(self, cache_root: Path | str) -> None:
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def user_ids(self) -> list[str]:
        url = self._tree_url("profile")
        entries = _json_request(url)
        return sorted(
            Path(str(entry["path"])).name.removeprefix("user")
            for entry in entries
            if entry.get("type") == "directory"
        )

    def case_paths(self, user_id: str) -> list[str]:
        folder = f"evaluation/user{user_id}/meta/overall"
        entries = _json_request(self._tree_url(folder))
        return sorted(
            str(entry["path"])
            for entry in entries
            if entry.get("type") == "file"
            and _META_NAME.match(Path(str(entry["path"])).name)
        )

    def case(self, user_id: str, path: str) -> PermaCase | None:
        raw = self.load(path)
        gold = str(raw.get("gold_label", "")).strip().upper()
        if gold not in set("ABCDEFGH"):
            return None
        return PermaCase(
            user_id=user_id,
            task_id=str(raw.get("task_id", Path(path).stem)),
            question=str(raw.get("question", "")),
            question_date=str(raw.get("question_date", "")),
            options=str(raw.get("options", "")),
            gold_label=gold,
            source_path=path,
        )

    def dialogues(self, user_id: str) -> list[dict[str, Any]]:
        raw = self.load(f"tasks/user{user_id}/raw_dialogues_c.json")
        sessions: list[dict[str, Any]] = []
        for topic in raw if isinstance(raw, list) else []:
            for session in topic.get("dialogs", []):
                if isinstance(session, dict):
                    sessions.append(session)
        return sorted(
            sessions,
            key=lambda item: (
                parse_date(item.get("date", "")),
                str(item.get("task_id", "")),
            ),
        )

    def load(self, relative_path: str) -> Any:
        target = self.cache_root / Path(relative_path)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            quoted = urllib.parse.quote(relative_path.replace("\\", "/"), safe="/")
            url = f"{DATASET_URL}/resolve/{DATASET_REVISION}/{quoted}"
            payload = _json_request(url)
            target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return json.loads(target.read_text(encoding="utf-8"))

    @staticmethod
    def _tree_url(folder: str) -> str:
        quoted = urllib.parse.quote(folder, safe="/")
        return (
            f"https://huggingface.co/api/datasets/{DATASET_REPO}/tree/"
            f"{DATASET_REVISION}/{quoted}?recursive=false&expand=false"
        )


def parse_date(value: Any) -> date:
    """Parse the date prefix used by PERMA timeline and metadata files."""
    match = _DATE.search(str(value))
    return date.fromisoformat(match.group(0)) if match else date.min


def parse_label(value: str) -> str:
    """Extract a single A-H prediction without accepting arbitrary prose."""
    clean = value.strip().upper()
    if clean in set("ABCDEFGH"):
        return clean
    match = _LABEL.search(clean)
    return match.group(1).upper() if match else ""


def split_options(value: str) -> dict[str, str]:
    """Split PERMA's newline-delimited A-H option string."""
    matches = list(_OPTION.finditer(value))
    return {
        match.group(1): value[
            match.end() : matches[index + 1].start()
            if index + 1 < len(matches)
            else len(value)
        ].strip()
        for index, match in enumerate(matches)
    }


def select_cases(
    cases: Iterable[PermaCase], *, limit: int, seed: int
) -> list[PermaCase]:
    """Select a deterministic cross-user sample without using labels."""
    selected = list(cases)
    random.Random(seed).shuffle(selected)
    if limit > 0:
        selected = selected[:limit]
    return sorted(
        selected,
        key=lambda item: (item.user_id, parse_date(item.question_date), item.task_id),
    )


def session_events(session: dict[str, Any], user_id: str) -> list[dict[str, Any]]:
    """Map one official dialogue to the existing incremental event interface."""
    session_id = f"perma-user{user_id}-{session.get('task_id', 'session')}"
    timestamp = f"{parse_date(session.get('date', '')).isoformat()}T00:00:00+00:00"
    events: list[dict[str, Any]] = []
    for index, message in enumerate(session.get("conversation", [])):
        if not isinstance(message, dict):
            continue
        events.append(
            {
                "event_type": "message_start",
                "session_id": session_id,
                "task_id": str(session.get("task_id", "")),
                "event_id": f"{session_id}-{index}",
                "timestamp": timestamp,
                "payload": {
                    "role": str(message.get("role", "")),
                    "content": str(message.get("content", "")),
                },
            }
        )
    return events


def _context(records: list[Any]) -> str:
    return "\n".join(f"- {record.content}" for record in records)


def _gold_support_rank(context: str, options: str, gold_label: str) -> int | None:
    """Return a lexical evidence rank; this is not an official PERMA metric."""
    parsed = split_options(options)
    if gold_label not in parsed:
        return None
    context_tokens = set(_TOKEN.findall(context.lower()))
    scores: list[tuple[float, str]] = []
    for label, option in parsed.items():
        option_tokens = set(_TOKEN.findall(option.lower()))
        overlap = len(context_tokens & option_tokens) / math.sqrt(
            max(1, len(option_tokens))
        )
        scores.append((overlap, label))
    scores.sort(key=lambda item: (-item[0], item[1]))
    return next(
        index for index, (_, label) in enumerate(scores, start=1) if label == gold_label
    )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _predict(provider: OpenAICompatibleProvider, case: PermaCase, context: str) -> str:
    prompt = ANSWER_OPTIONAL_PROMPT.format(
        context=context,
        question=case.question,
        options=case.options,
    )
    return parse_label(provider.complete([ChatMessage(role="user", content=prompt)]))


def _implementation_sha256() -> str:
    project_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for relative in IMPLEMENTATION_FILES:
        digest.update(relative.encode("utf-8"))
        digest.update((project_root / relative).read_bytes())
    return digest.hexdigest()


def evaluate(
    *,
    cache_root: Path,
    max_tasks: int,
    sample_offset: int,
    seed: int,
    top_k: int,
    run_mcq: bool,
    run_no_memory: bool,
) -> dict[str, Any]:
    dataset = PermaDataset(cache_root)
    user_ids = dataset.user_ids()
    case_refs = [
        (user_id, path) for user_id in user_ids for path in dataset.case_paths(user_id)
    ]
    random.Random(seed).shuffle(case_refs)
    case_refs = case_refs[max(0, sample_offset) :]
    if max_tasks > 0:
        case_refs = case_refs[:max_tasks]
    selected = [
        case
        for user_id, path in case_refs
        if (case := dataset.case(user_id, path)) is not None
    ]
    selected.sort(
        key=lambda item: (item.user_id, parse_date(item.question_date), item.task_id)
    )
    selected_by_user: dict[str, list[PermaCase]] = {}
    for case in selected:
        selected_by_user.setdefault(case.user_id, []).append(case)

    provider = OpenAICompatibleProvider()
    embedding_config = EmbeddingConfig.from_env()
    embedding_provider = create_embedding_provider(embedding_config)
    if run_mcq and not provider.enabled:
        raise RuntimeError(
            "MCQ evaluation requested but the chat provider is not configured"
        )

    results: list[CaseResult] = []
    extraction = {
        "official_preference_sessions": 0,
        "covered_preference_sessions": 0,
        "stable_preference_sessions": 0,
        "observation_only_preference_sessions": 0,
        "user_utterances": 0,
        "extracted_preference_facts": 0,
        "extracted_preference_observations": 0,
        "reinforced_preference_facts": 0,
    }
    with tempfile.TemporaryDirectory(prefix="osagent-perma-run-") as run_dir:
        for user_id, user_cases in selected_by_user.items():
            root = Path(run_dir) / f"user{user_id}"
            config = MemoryConfig(root / "config.json")
            core = CoreMemoryStore(root / "memory", config=config)
            archival = ArchivalMemoryStore(root / "memory", config=config)
            pipeline = ExtractUpdatePipeline(core, archival, config=config)
            module = AgentMemoryModule(
                root / "memory",
                core=core,
                archival=archival,
                config=config,
                embedding_provider=embedding_provider,
            )
            sessions = dataset.dialogues(user_id)
            session_index = 0
            for case in user_cases:
                cutoff = parse_date(case.question_date)
                while (
                    session_index < len(sessions)
                    and parse_date(sessions[session_index].get("date", "")) <= cutoff
                ):
                    session = sessions[session_index]
                    events = session_events(session, user_id)
                    extraction["user_utterances"] += sum(
                        event["payload"]["role"] == "user" for event in events
                    )
                    preference_bearing = bool(
                        str(session.get("preferences", "")).strip()
                        or str(session.get("new_preferences", "")).strip()
                    )
                    extraction["official_preference_sessions"] += int(
                        preference_bearing
                    )
                    facts = pipeline.process_session(
                        events,
                        session_id=f"perma-user{user_id}-{session.get('task_id', session_index)}",
                    )
                    preference_facts = [
                        fact for fact in facts if "preference" in fact.tags
                    ]
                    stable_preference_facts = [
                        fact
                        for fact in preference_facts
                        if "preference_observation" not in fact.tags
                    ]
                    observation_facts = [
                        fact
                        for fact in preference_facts
                        if "preference_observation" in fact.tags
                    ]
                    extraction["extracted_preference_facts"] += len(preference_facts)
                    extraction["extracted_preference_observations"] += len(
                        observation_facts
                    )
                    extraction["reinforced_preference_facts"] += sum(
                        "preference_reinforced" in fact.tags
                        for fact in stable_preference_facts
                    )
                    extraction["covered_preference_sessions"] += int(
                        preference_bearing and bool(preference_facts)
                    )
                    extraction["stable_preference_sessions"] += int(
                        preference_bearing and bool(stable_preference_facts)
                    )
                    extraction["observation_only_preference_sessions"] += int(
                        preference_bearing
                        and bool(observation_facts)
                        and not stable_preference_facts
                    )
                    session_index += 1

                index_started = time.perf_counter()
                module.refresh_semantic_indexes()
                semantic_index_ms = (time.perf_counter() - index_started) * 1000
                started = time.perf_counter()
                recalled = module.recall(
                    RecallRequest(query=case.question, scope="ltm", limit=top_k)
                )
                recall_ms = (time.perf_counter() - started) * 1000
                context = _context(recalled.records)
                row = CaseResult(
                    user_id=user_id,
                    task_id=case.task_id,
                    question_date=case.question_date,
                    question=case.question,
                    source_path=case.source_path,
                    gold_label=case.gold_label,
                    recall_ms=round(recall_ms, 3),
                    semantic_index_ms=round(semantic_index_ms, 3),
                    recalled_count=len(recalled.records),
                    preference_count=sum(
                        "preference" in record.tags for record in recalled.records
                    ),
                    raw_event_count=sum(
                        "raw_event" in record.tags for record in recalled.records
                    ),
                    semantic_session_count=sum(
                        "semantic_session" in record.tags
                        for record in recalled.records
                    ),
                    gold_support_rank=_gold_support_rank(
                        context, case.options, case.gold_label
                    ),
                    retrieved=[record.content for record in recalled.records],
                )
                if run_mcq:
                    try:
                        row.predicted_label = _predict(provider, case, context)
                        row.correct = row.predicted_label == case.gold_label
                        if run_no_memory:
                            row.no_memory_label = _predict(provider, case, "")
                            row.no_memory_correct = (
                                row.no_memory_label == case.gold_label
                            )
                    except (LLMError, OSError, TimeoutError, ValueError) as exc:
                        row.error = f"{type(exc).__name__}: {exc}"
                results.append(row)

    extraction["preference_session_coverage"] = round(
        extraction["covered_preference_sessions"]
        / max(1, extraction["official_preference_sessions"]),
        4,
    )
    extraction["stable_preference_session_coverage"] = round(
        extraction["stable_preference_sessions"]
        / max(1, extraction["official_preference_sessions"]),
        4,
    )
    latencies = [row.recall_ms for row in results]
    indexing_latencies = [row.semantic_index_ms for row in results]
    mcq_rows = [row for row in results if row.correct is not None]
    no_memory_rows = [row for row in results if row.no_memory_correct is not None]
    summary = {
        "cases": len(results),
        "users": len(selected_by_user),
        "nonempty_recall_rate": round(
            sum(row.recalled_count > 0 for row in results) / max(1, len(results)), 4
        ),
        "preference_recall_rate": round(
            sum(row.preference_count > 0 for row in results) / max(1, len(results)), 4
        ),
        "raw_event_recall_rate": round(
            sum(row.raw_event_count > 0 for row in results) / max(1, len(results)), 4
        ),
        "semantic_session_recall_rate": round(
            sum(row.semantic_session_count > 0 for row in results)
            / max(1, len(results)),
            4,
        ),
        "gold_option_lexical_support_at_1": round(
            sum(row.gold_support_rank == 1 for row in results) / max(1, len(results)), 4
        ),
        "gold_option_lexical_support_at_3": round(
            sum(
                row.gold_support_rank is not None and row.gold_support_rank <= 3
                for row in results
            )
            / max(1, len(results)),
            4,
        ),
        "gold_option_lexical_support_at_5": round(
            sum(
                row.gold_support_rank is not None and row.gold_support_rank <= 5
                for row in results
            )
            / max(1, len(results)),
            4,
        ),
        "recall_latency_ms": {
            "p50": round(statistics.median(latencies), 3) if latencies else 0.0,
            "p95": round(_percentile(latencies, 0.95), 3),
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "semantic_index_latency_ms": {
            "p50": (
                round(statistics.median(indexing_latencies), 3)
                if indexing_latencies
                else 0.0
            ),
            "p95": round(_percentile(indexing_latencies, 0.95), 3),
            "max": round(max(indexing_latencies), 3) if indexing_latencies else 0.0,
        },
        "mcq_accuracy": (
            round(sum(bool(row.correct) for row in mcq_rows) / len(mcq_rows), 4)
            if mcq_rows
            else None
        ),
        "mcq_scored_cases": len(mcq_rows),
        "no_memory_mcq_accuracy": (
            round(
                sum(bool(row.no_memory_correct) for row in no_memory_rows)
                / len(no_memory_rows),
                4,
            )
            if no_memory_rows
            else None
        ),
        "no_memory_scored_cases": len(no_memory_rows),
    }
    summary["memory_mcq_delta"] = (
        round(summary["mcq_accuracy"] - summary["no_memory_mcq_accuracy"], 4)
        if summary["mcq_accuracy"] is not None
        and summary["no_memory_mcq_accuracy"] is not None
        else None
    )
    return {
        "provenance": {
            "dataset": DATASET_REPO,
            "dataset_url": DATASET_URL,
            "dataset_revision": DATASET_REVISION,
            "official_repo_revision": OFFICIAL_REPO_REVISION,
            "license": "Apache-2.0",
            "subset": SUBSET_DESCRIPTION,
            "sampling_seed": seed,
            "sample_offset": sample_offset,
            "requested_max_tasks": max_tasks,
            "top_k": top_k,
            "answer_prompt_sha256": hashlib.sha256(
                ANSWER_OPTIONAL_PROMPT.encode()
            ).hexdigest(),
            "implementation_sha256": _implementation_sha256(),
            "chat_model": provider.config.model if run_mcq else None,
            "chat_base_url": provider.config.base_url if run_mcq else None,
            "embedding_backend": embedding_config.backend,
            "embedding_model": embedding_config.model,
            "embedding_revision": embedding_config.revision,
        },
        "summary": summary,
        "extraction": extraction,
        "cases": [public_case_result(row) for row in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / f"osagent-perma-{DATASET_REVISION[:8]}",
    )
    parser.add_argument("--max-tasks", type=int, default=100)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--mcq", action="store_true", help="Run official-prompt MCQ scoring"
    )
    parser.add_argument(
        "--no-memory-baseline",
        action="store_true",
        help="Also query the same chat model with an empty memory context",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.no_memory_baseline and not args.mcq:
        parser.error("--no-memory-baseline requires --mcq")
    result = evaluate(
        cache_root=args.cache_dir,
        max_tasks=args.max_tasks,
        sample_offset=args.sample_offset,
        seed=args.seed,
        top_k=args.top_k,
        run_mcq=args.mcq,
        run_no_memory=args.no_memory_baseline,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(result["extraction"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
