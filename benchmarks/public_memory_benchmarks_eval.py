"""Blind retrieval evaluation over the five recommended public benchmarks.

Only benchmark history is inserted into the memory module. Gold answers,
tool calls, evidence annotations, and forgetting annotations remain in the
evaluator and are read only after recall returns. The result JSON contains no
benchmark text, so it can be retained without turning test data into memory.

These are memory-layer diagnostics. Metrics that require an answer model or
an LLM judge are deliberately reported as unavailable rather than replaced by
retrieval proxies with an official-looking name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from os_agent.memory.embedding_cache import SqliteEmbeddingCache
from os_agent.memory.embedding_provider import create_embedding_provider
from os_agent.memory.fact_vector_index import FactSemanticIndex
from os_agent.memory.interface import MemoryRecord, RecallRequest
from os_agent.memory.mem0_schemas import FactCard
from os_agent.memory.memory_config import MemoryConfig
from os_agent.memory.module import AgentMemoryModule
from os_agent.memory.session_vector_index import SessionSemanticIndex


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "doc" / "evaluation" / "results" / "public-benchmarks"
RUNTIME_ROOT = ROOT / "data" / "comparison-runtimes" / "public-benchmarks"
TOKEN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]", re.IGNORECASE)
SPACE = re.compile(r"\s+")


@dataclass(slots=True)
class RecallObservation:
    records: list[MemoryRecord]
    latency_ms: float


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "p50": round(percentile(values, 0.50), 3),
        "p95": round(percentile(values, 0.95), 3),
        "p99": round(percentile(values, 0.99), 3),
        "max": round(max(values), 3) if values else 0.0,
    }


def revision(path: Path) -> str:
    import subprocess

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


def normalize(text: object) -> str:
    return SPACE.sub(" ", str(text).casefold()).strip()


def tokens(text: object) -> set[str]:
    return set(TOKEN.findall(normalize(text)))


def text_supported(needle: str, haystack: str) -> bool:
    expected = normalize(needle)
    actual = normalize(haystack)
    if len(expected) < 2:
        return False
    if expected in actual:
        return True
    expected_tokens = tokens(expected)
    if len(expected_tokens) < 2:
        return False
    coverage = len(expected_tokens & tokens(actual)) / len(expected_tokens)
    return coverage >= 0.80


def support_rank(needles: Iterable[str], records: list[MemoryRecord]) -> int | None:
    expected = [str(item) for item in needles if len(normalize(item)) >= 2]
    for rank, record in enumerate(records, start=1):
        if any(text_supported(item, record.content) for item in expected):
            return rank
    return None


def flatten_strings(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, str):
        clean = normalize(value)
        if len(clean) >= 3:
            found.append(value)
    elif isinstance(value, dict):
        for nested in value.values():
            found.extend(flatten_strings(nested))
    elif isinstance(value, list):
        for nested in value:
            found.extend(flatten_strings(nested))
    return list(dict.fromkeys(found))


def chunk_text(text: str, *, max_chars: int = 1800, overlap: int = 180) -> list[str]:
    clean = text.strip()
    if not clean:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(start + max_chars, len(clean))
        if end < len(clean):
            boundary = max(clean.rfind("\n", start + max_chars // 2, end), clean.rfind(". ", start + max_chars // 2, end))
            if boundary > start:
                end = boundary + 1
        chunks.append(clean[start:end].strip())
        if end >= len(clean):
            break
        start = max(end - overlap, start + 1)
    return [chunk for chunk in chunks if chunk]


def fact(
    dataset: str,
    unit_id: str,
    index: int,
    text: str,
    *,
    session_id: str = "",
) -> FactCard:
    return FactCard(
        entity=dataset,
        attribute=f"{unit_id}:{index:05d}",
        value=text,
        evidence=text[:500],
        source_session_id=session_id or unit_id,
        source_event_id=f"{unit_id}:{index}",
        confidence=1.0,
        importance=0.5,
        tags=["public_benchmark", dataset],
    )


class BlindMemoryHarness:
    """Build isolated memory banks and expose a query-only recall method."""

    def __init__(self) -> None:
        config_path = ROOT / "config" / "memory_config.json"
        self.config = MemoryConfig(config_path)
        self.provider = create_embedding_provider()
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        self.cache = SqliteEmbeddingCache(RUNTIME_ROOT / "embeddings.sqlite3")

    def recall_bank(
        self,
        facts: list[FactCard],
        queries: list[str],
        *,
        top_k: int,
    ) -> tuple[list[RecallObservation], float]:
        with tempfile.TemporaryDirectory(prefix="osagent-public-bench-") as tmp:
            memory_root = Path(tmp) / "memory"
            memory_root.mkdir(parents=True, exist_ok=True)
            (memory_root / "core_facts.json").write_text(
                json.dumps([row.to_dict() for row in facts], ensure_ascii=False),
                encoding="utf-8",
            )
            semantic_index = FactSemanticIndex(
                provider=self.provider,
                cache=self.cache,
            )
            session_index = SessionSemanticIndex(
                provider=self.provider,
                cache=self.cache,
            )
            module = AgentMemoryModule(
                memory_root,
                config=self.config,
                semantic_index=semantic_index,
                session_semantic_index=session_index,
                embedding_provider=self.provider,
            )
            started = time.perf_counter()
            module.refresh_semantic_indexes()
            index_ms = (time.perf_counter() - started) * 1000
            observations = [blind_recall(module, query, top_k) for query in queries]
        return observations, index_ms


def blind_recall(module: AgentMemoryModule, query: str, top_k: int) -> RecallObservation:
    """The only function on the prediction path: query in, records out."""
    started = time.perf_counter()
    result = module.recall(RecallRequest(query=query, scope="ltm", limit=top_k))
    latency = (time.perf_counter() - started) * 1000
    return RecallObservation(records=result.records, latency_ms=latency)


def result_envelope(
    *,
    benchmark: str,
    dataset_revision: str,
    dataset_sha256: str,
    protocol: str,
    summary: dict[str, Any],
    cases: list[dict[str, Any]],
    started: float,
) -> dict[str, Any]:
    return {
        "provenance": {
            "system": "OS Agent explicit memory",
            "benchmark": benchmark,
            "dataset_revision": dataset_revision,
            "dataset_sha256": dataset_sha256,
            "protocol": protocol,
            "evaluated_at": datetime.now(UTC).isoformat(),
            "embedding_backend": type(create_embedding_provider()).__name__,
            "labels_visible_to_memory": False,
            "result_contains_benchmark_text": False,
            "answer_model": None,
            "official_generative_score_available": False,
        },
        "summary": summary,
        "cases": cases,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def evaluate_mem2act(harness: BlindMemoryHarness, top_k: int) -> dict[str, Any]:
    started = time.perf_counter()
    source = ROOT / "data" / "benchmark-sources" / "Mem2ActBench"
    data_root = source / "toolmembench_small"
    history_path = data_root / "toolmem_conversation.jsonl"
    qa_path = data_root / "qa_dataset.jsonl"
    sessions = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    facts: list[FactCard] = []
    source_to_sessions: dict[str, set[str]] = defaultdict(set)
    for session in sessions:
        session_id = str(session["session_id"])
        for source_id in session.get("original_conversation_ids", []):
            source_to_sessions[str(source_id)].add(session_id)
        visible = "\n".join(
            f"{turn.get('role', 'unknown')}: {turn.get('content', '')}"
            for turn in session.get("turns", [])
        )
        for chunk_index, chunk in enumerate(chunk_text(visible)):
            facts.append(
                fact(
                    "mem2act",
                    session_id,
                    chunk_index,
                    chunk,
                    session_id=session_id,
                )
            )
    rows = [json.loads(line) for line in qa_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observations, index_ms = harness.recall_bank(
        facts,
        [str(row.get("query", "")) for row in rows],
        top_k=top_k,
    )
    cases: list[dict[str, Any]] = []
    for row, observation in zip(rows, observations):
        gold_sessions = {
            session_id
            for source_id in row.get("source_conversation_ids", [])
            for session_id in source_to_sessions.get(str(source_id), set())
        }
        session_rank = next(
            (
                rank
                for rank, record in enumerate(observation.records, start=1)
                if str(record.metadata.get("source_session_id", "")) in gold_sessions
            ),
            None,
        )
        tool_call = row.get("tool_call") or {}
        argument_values = flatten_strings(tool_call.get("arguments") or {})
        supported_arguments = sum(
            support_rank([value], observation.records) is not None
            for value in argument_values
        )
        tool_rank = support_rank([str(tool_call.get("name", ""))], observation.records)
        cases.append(
            {
                "case_id": str(row.get("qa_id", "")),
                "complexity": str((row.get("complexity_metadata") or {}).get("level", "")),
                "source_session_rank": session_rank,
                "tool_name_support_rank": tool_rank,
                "argument_count": len(argument_values),
                "argument_supported": supported_arguments,
                "recall_ms": round(observation.latency_ms, 3),
            }
        )
    latencies = [float(row["recall_ms"]) for row in cases]
    summary = {
        "sessions": len(sessions),
        "history_facts": len(facts),
        "cases": len(cases),
        "source_session_recall_at_5": round(sum(row["source_session_rank"] is not None and row["source_session_rank"] <= 5 for row in cases) / max(1, len(cases)), 4),
        "tool_name_support_at_5": round(sum(row["tool_name_support_rank"] is not None and row["tool_name_support_rank"] <= 5 for row in cases) / max(1, len(cases)), 4),
        "argument_value_coverage_at_5": round(sum(row["argument_supported"] for row in cases) / max(1, sum(row["argument_count"] for row in cases)), 4),
        "all_arguments_supported_at_5": round(sum(row["argument_count"] > 0 and row["argument_supported"] == row["argument_count"] for row in cases) / max(1, len(cases)), 4),
        "index_ms": round(index_ms, 3),
        "recall_latency_ms": latency_summary(latencies),
        "official_tool_selection_accuracy": None,
        "official_argument_accuracy": None,
        "official_end_to_end_success": None,
    }
    return result_envelope(
        benchmark="Mem2ActBench",
        dataset_revision=revision(source),
        dataset_sha256=sha256(qa_path),
        protocol="official small tier: 429-session history; 400 queries; retrieval-only proxy; gold read after recall",
        summary=summary,
        cases=cases,
        started=started,
    )


def evaluate_memory_agent_bench(
    harness: BlindMemoryHarness, top_k: int
) -> dict[str, Any]:
    from datasets import Dataset

    started = time.perf_counter()
    source = ROOT / "data" / "benchmark-sources" / "MemoryAgentBench"
    data_root = source / "hf-data"
    selected = {
        "Accurate_Retrieval.parquet": {"eventqa_full"},
    }
    cases: list[dict[str, Any]] = []
    index_times: list[float] = []
    fact_count = 0
    for filename, sources in selected.items():
        dataset = Dataset.from_parquet(str(data_root / filename))
        for row_index, row in enumerate(dataset):
            metadata = row.get("metadata") or {}
            source_name = str(metadata.get("source", ""))
            if source_name not in sources:
                continue
            facts = [
                fact("memory_agent_bench", f"{source_name}-{row_index}", index, chunk)
                for index, chunk in enumerate(chunk_text(str(row.get("context", ""))))
            ]
            fact_count += len(facts)
            questions = [str(item) for item in row.get("questions", [])]
            observations, index_ms = harness.recall_bank(facts, questions, top_k=top_k)
            index_times.append(index_ms)
            answers = row.get("answers", [])
            previous_events = metadata.get("previous_events") or []
            qa_ids = metadata.get("qa_pair_ids") or []
            for question_index, observation in enumerate(observations):
                answer_values = flatten_strings(answers[question_index] if question_index < len(answers) else [])
                answer_rank = support_rank(answer_values, observation.records)
                evidence_values = flatten_strings(previous_events[question_index] if question_index < len(previous_events) else [])
                evidence_rank = support_rank(evidence_values, observation.records)
                cases.append(
                    {
                        "case_id": str(qa_ids[question_index] if question_index < len(qa_ids) else f"{source_name}:{row_index}:{question_index}"),
                        "benchmark_area": "conflict_resolution" if filename.startswith("Conflict") else "accurate_retrieval",
                        "source": source_name,
                        "answer_support_rank": answer_rank,
                        "evidence_support_rank": evidence_rank,
                        "has_evidence_label": bool(evidence_values),
                        "recall_ms": round(observation.latency_ms, 3),
                    }
                )
    def area_rows(area: str) -> list[dict[str, Any]]:
        return [row for row in cases if row["benchmark_area"] == area]

    areas: dict[str, Any] = {}
    for area in ("accurate_retrieval",):
        rows = area_rows(area)
        areas[area] = {
            "cases": len(rows),
            "answer_support_at_5": round(sum(row["answer_support_rank"] is not None and row["answer_support_rank"] <= 5 for row in rows) / max(1, len(rows)), 4),
            "evidence_support_at_5": round(sum(row["evidence_support_rank"] is not None and row["evidence_support_rank"] <= 5 for row in rows if row["has_evidence_label"]) / max(1, sum(row["has_evidence_label"] for row in rows)), 4) if any(row["has_evidence_label"] for row in rows) else None,
            "recall_latency_ms": latency_summary([float(row["recall_ms"]) for row in rows]),
        }
    summary = {
        "cases": len(cases),
        "history_facts": fact_count,
        "areas": areas,
        "index_latency_ms": latency_summary(index_times),
        "official_answer_accuracy": None,
        "official_conflict_accuracy": None,
    }
    combined_hash = hashlib.sha256()
    for filename in sorted(selected):
        combined_hash.update(bytes.fromhex(sha256(data_root / filename)))
    return result_envelope(
        benchmark="MemoryAgentBench",
        dataset_revision="7ea066982b140a19337e17e60d45d4076e042faf",
        dataset_sha256=combined_hash.hexdigest(),
        protocol="EventQA-full 500-question subset; retrieval-only proxy; gold read after recall",
        summary=summary,
        cases=cases,
        started=started,
    )


def _memora_evidence_score(
    evidence: Any, records: list[MemoryRecord]
) -> tuple[int, int]:
    values = flatten_strings(evidence)
    supported = sum(support_rank([value], records) is not None for value in values)
    return supported, len(values)


def evaluate_memora(harness: BlindMemoryHarness, top_k: int) -> dict[str, Any]:
    started = time.perf_counter()
    source = ROOT / "data" / "benchmark-sources" / "Memora"
    period_root = source / "data" / "weekly"
    cases: list[dict[str, Any]] = []
    index_times: list[float] = []
    session_count = 0
    fact_count = 0
    for persona_root in sorted(path for path in period_root.iterdir() if path.is_dir()):
        question_path = next(persona_root.glob("evaluation_questions_*.json"))
        payload = json.loads(question_path.read_text(encoding="utf-8"))
        category_rows: list[tuple[str, dict[str, Any]]] = []
        for category in ("remembering", "reasoning", "recommending"):
            category_rows.extend((category, row) for row in payload["questions"][category])
        cutoff = max(str(row.get("question_date", "")) for _, row in category_rows)
        facts: list[FactCard] = []
        for session_path in sorted((persona_root / "conversations").glob("session_*.json")):
            session = json.loads(session_path.read_text(encoding="utf-8"))
            if str(session.get("date", "")) > cutoff:
                continue
            session_count += 1
            session_id = f"{persona_root.name}:{session.get('session_id', session_path.stem)}"
            visible = "\n".join(
                f"{turn.get('speaker', 'unknown')}: {turn.get('message', '')}"
                for turn in session.get("conversation", [])
            )
            for index, chunk in enumerate(chunk_text(visible)):
                facts.append(fact("memora", session_id, index, chunk, session_id=session_id))
        fact_count += len(facts)
        observations, index_ms = harness.recall_bank(
            facts,
            [str(row.get("question", "")) for _, row in category_rows],
            top_k=top_k,
        )
        index_times.append(index_ms)
        for (category, row), observation in zip(category_rows, observations):
            presence_supported, presence_total = _memora_evidence_score(
                row.get("memory_evidence") or {}, observation.records
            )
            forgotten_leaked, forgotten_total = _memora_evidence_score(
                row.get("forgetting_evidence") or {}, observation.records
            )
            presence = presence_supported / presence_total if presence_total else 0.0
            forgetting_absence = 1.0 - (forgotten_leaked / forgotten_total) if forgotten_total else 1.0
            weight = forgotten_total / max(1, presence_total + forgotten_total)
            proxy_fama = max(0.0, presence - weight * (1.0 - forgetting_absence))
            cases.append(
                {
                    "case_id": str(row.get("question_id", "")),
                    "persona": persona_root.name,
                    "task": category,
                    "presence_supported": presence_supported,
                    "presence_total": presence_total,
                    "forgotten_leaked": forgotten_leaked,
                    "forgotten_total": forgotten_total,
                    "retrieval_proxy_fama": round(proxy_fama, 4),
                    "recall_ms": round(observation.latency_ms, 3),
                }
            )
    presence_supported = sum(row["presence_supported"] for row in cases)
    presence_total = sum(row["presence_total"] for row in cases)
    forgotten_leaked = sum(row["forgotten_leaked"] for row in cases)
    forgotten_total = sum(row["forgotten_total"] for row in cases)
    summary = {
        "period": "weekly",
        "personas": 10,
        "sessions": session_count,
        "history_facts": fact_count,
        "cases": len(cases),
        "memory_evidence_coverage_at_5": round(presence_supported / max(1, presence_total), 4),
        "forgotten_evidence_leakage_at_5": round(forgotten_leaked / max(1, forgotten_total), 4),
        "retrieval_proxy_fama": round(statistics.fmean(row["retrieval_proxy_fama"] for row in cases), 4),
        "index_latency_ms": latency_summary(index_times),
        "recall_latency_ms": latency_summary([float(row["recall_ms"]) for row in cases]),
        "official_fama": None,
        "official_remembering_accuracy": None,
        "official_reasoning_accuracy": None,
        "official_recommending_accuracy": None,
    }
    return result_envelope(
        benchmark="Memora",
        dataset_revision=revision(source),
        dataset_sha256=sha256(next(period_root.glob("*/evaluation_questions_*.json"))),
        protocol="complete official weekly tier: 10 personas, 1,551 sessions, 150 questions; text retrieval proxy; annotations read after recall",
        summary=summary,
        cases=cases,
        started=started,
    )


def trajectory_summary(row: dict[str, Any], *, max_chars: int = 16000) -> str:
    parts = [
        f"Goal: {row.get('goal', '')}",
        f"Outcome: {row.get('outcome', '')}",
        f"Start URL: {row.get('start_url', '')}",
    ]
    tree_budget = 4000
    for state in row.get("states", []):
        parts.append(
            " | ".join(
                str(value)
                for value in (
                    state.get("step", ""),
                    state.get("url", ""),
                    state.get("action", ""),
                    state.get("thought", ""),
                )
                if value
            )
        )
        tree = str(state.get("accessibility_tree", ""))
        if tree and tree_budget > 0:
            excerpt = tree[: min(500, tree_budget)]
            parts.append(excerpt)
            tree_budget -= len(excerpt)
        if sum(len(part) for part in parts) >= max_chars:
            break
    return "\n".join(part for part in parts if part)[:max_chars]


def evaluate_longmemeval_v2(
    harness: BlindMemoryHarness, top_k: int
) -> dict[str, Any]:
    started = time.perf_counter()
    source = ROOT / "data" / "benchmark-sources" / "LongMemEval-V2"
    data_root = source / "data" / "public"
    trajectories_path = data_root / "trajectories.jsonl"
    if not trajectories_path.exists():
        raise FileNotFoundError(
            f"missing LongMemEval-V2 text trajectories: {trajectories_path}"
        )
    questions = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in (data_root / "questions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    haystacks = json.loads(
        (data_root / "haystacks" / "lme_v2_small.json").read_text(encoding="utf-8")
    )
    required_ids = {item for values in haystacks.values() for item in values}
    trajectories: dict[str, str] = {}
    with trajectories_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("id") in required_ids:
                trajectories[str(row["id"])] = trajectory_summary(row)
    missing = required_ids - set(trajectories)
    if missing:
        raise ValueError(f"missing {len(missing)} required trajectories")
    cases: list[dict[str, Any]] = []
    index_times: list[float] = []
    for question_id, allowed_ids in haystacks.items():
        question = questions[question_id]
        facts = [
            fact("longmemeval_v2", trajectory_id, 0, trajectories[trajectory_id], session_id=trajectory_id)
            for trajectory_id in allowed_ids
        ]
        observations, index_ms = harness.recall_bank(
            facts,
            [str(question.get("question", ""))],
            top_k=top_k,
        )
        index_times.append(index_ms)
        observation = observations[0]
        answer_rank = support_rank(
            flatten_strings(question.get("answer", "")), observation.records
        )
        cases.append(
            {
                "case_id": question_id,
                "domain": str(question.get("domain", "")),
                "question_type": str(question.get("question_type", "")),
                "has_question_image": bool(question.get("image")),
                "answer_support_rank": answer_rank,
                "recall_ms": round(observation.latency_ms, 3),
            }
        )
    by_type: dict[str, Any] = {}
    for question_type in sorted({row["question_type"] for row in cases}):
        rows = [row for row in cases if row["question_type"] == question_type]
        by_type[question_type] = {
            "cases": len(rows),
            "answer_support_at_5": round(sum(row["answer_support_rank"] is not None and row["answer_support_rank"] <= 5 for row in rows) / max(1, len(rows)), 4),
        }
    summary = {
        "tier": "small",
        "questions": len(cases),
        "haystack_trajectories_per_question": 100,
        "unique_trajectories": len(trajectories),
        "text_only_questions": sum(not row["has_question_image"] for row in cases),
        "answer_support_at_5": round(sum(row["answer_support_rank"] is not None and row["answer_support_rank"] <= 5 for row in cases) / max(1, len(cases)), 4),
        "by_question_type": by_type,
        "index_latency_ms": latency_summary(index_times),
        "recall_latency_ms": latency_summary([float(row["recall_ms"]) for row in cases]),
        "official_answer_accuracy": None,
        "official_lafs": None,
    }
    return result_envelope(
        benchmark="LongMemEval-V2",
        dataset_revision="f152293e235517d504809563c833d7190b8c713b",
        dataset_sha256=sha256(data_root / "questions.jsonl"),
        protocol="complete public Small tier: 451 questions x 100 allowed trajectories; deterministic text compaction; gold read after recall; screenshots excluded",
        summary=summary,
        cases=cases,
        started=started,
    )


def evaluate_perma(top_k: int) -> dict[str, Any]:
    from benchmarks.perma_memory_eval import evaluate

    started = time.perf_counter()
    cache_root = RUNTIME_ROOT / "perma-440e64e4"
    raw = evaluate(
        cache_root=cache_root,
        max_tasks=0,
        sample_offset=0,
        seed=20260728,
        top_k=top_k,
        run_mcq=False,
        run_no_memory=False,
    )
    cases = [
        {
            "case_id": str(row.get("task_id", "")),
            "user_id": str(row.get("user_id", "")),
            "gold_support_rank": row.get("gold_support_rank"),
            "recall_ms": row.get("recall_ms"),
            "semantic_index_ms": row.get("semantic_index_ms"),
        }
        for row in raw.get("cases", [])
    ]
    return result_envelope(
        benchmark="PERMA",
        dataset_revision=str(raw["provenance"]["dataset_revision"]),
        dataset_sha256=str(raw["provenance"]["implementation_sha256"]),
        protocol="all public clean single-domain Type-3 cases; incremental history; gold labels read after recall; retrospective because prior slices were consumed",
        summary={
            **raw["summary"],
            "extraction": raw["extraction"],
            "official_mcq_accuracy": None,
            "official_memory_score": None,
        },
        cases=cases,
        started=started,
    )


EVALUATORS = {
    "mem2act": lambda harness, top_k: evaluate_mem2act(harness, top_k),
    "perma": lambda harness, top_k: evaluate_perma(top_k),
    "memory-agent-bench": lambda harness, top_k: evaluate_memory_agent_bench(harness, top_k),
    "memora": lambda harness, top_k: evaluate_memora(harness, top_k),
    "longmemeval-v2": lambda harness, top_k: evaluate_longmemeval_v2(harness, top_k),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=["all", *EVALUATORS],
        default="all",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=RESULT_ROOT)
    args = parser.parse_args()
    os.environ.setdefault(
        "OS_AGENT_EMBEDDING_CONFIG",
        str(ROOT / "config" / "embedding.local.json"),
    )
    selected = list(EVALUATORS) if args.benchmark == "all" else [args.benchmark]
    harness = BlindMemoryHarness()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in selected:
        result = EVALUATORS[name](harness, max(args.top_k, 1))
        output = args.output_dir / f"{name}.json"
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"benchmark": name, "summary": result["summary"]}, ensure_ascii=False, indent=2))
        print(f"result: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
