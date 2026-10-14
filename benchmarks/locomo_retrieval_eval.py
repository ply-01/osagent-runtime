"""Evaluate explicit-memory retrieval on LoCoMo dialogue evidence labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from os_agent.memory.interface import RecallRequest
from os_agent.memory.mem0_schemas import FactCard
from os_agent.memory.memory_config import MemoryConfig
from os_agent.memory.module import AgentMemoryModule

DATASET_NAME = "snap-research/locomo"
DATASET_REVISION = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
SEED = 20260726


@dataclass(frozen=True, slots=True)
class RetrievalCase:
    sample_id: str
    question_index: int
    question: str
    evidence_ids: tuple[str, ...]
    category: int


@dataclass(slots=True)
class CaseResult:
    sample_id: str
    question_index: int
    category: int
    evidence_count: int
    retrieved_event_ids: list[str]
    first_evidence_rank: int | None
    recall_ms: float


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize_results(
    results: list[CaseResult],
    indexing_ms: list[float],
    conversations: int,
) -> dict[str, Any]:
    """Normalize retrieval metrics shared by every LoCoMo backend adapter."""
    latencies = [row.recall_ms for row in results]
    by_category: dict[str, dict[str, Any]] = {}
    for category in sorted({row.category for row in results}):
        rows = [row for row in results if row.category == category]
        by_category[str(category)] = {
            "cases": len(rows),
            "recall_at_5": round(
                sum(
                    row.first_evidence_rank is not None
                    and row.first_evidence_rank <= 5
                    for row in rows
                )
                / max(1, len(rows)),
                4,
            ),
        }
    return {
        "cases": len(results),
        "conversations": conversations,
        "recall_at_1": round(
            sum(row.first_evidence_rank == 1 for row in results)
            / max(1, len(results)),
            4,
        ),
        "recall_at_3": round(
            sum(
                row.first_evidence_rank is not None and row.first_evidence_rank <= 3
                for row in results
            )
            / max(1, len(results)),
            4,
        ),
        "recall_at_5": round(
            sum(
                row.first_evidence_rank is not None and row.first_evidence_rank <= 5
                for row in results
            )
            / max(1, len(results)),
            4,
        ),
        "mrr_at_5": round(
            statistics.fmean(
                1 / row.first_evidence_rank
                if row.first_evidence_rank is not None
                and row.first_evidence_rank <= 5
                else 0.0
                for row in results
            )
            if results
            else 0.0,
            4,
        ),
        "recall_latency_ms": {
            "p50": round(percentile(latencies, 0.5), 3),
            "p95": round(percentile(latencies, 0.95), 3),
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "indexing_ms": {
            "p50": round(percentile(indexing_ms, 0.5), 3),
            "p95": round(percentile(indexing_ms, 0.95), 3),
            "max": round(max(indexing_ms), 3) if indexing_ms else 0.0,
        },
        "by_category": by_category,
    }


def embedding_provenance() -> dict[str, Any]:
    """Return a secret-free description of the embedding config used."""
    path = Path(
        os.getenv("OS_AGENT_EMBEDDING_CONFIG", "config/embedding.local.json")
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"embedding_config": str(path), "embedding_config_readable": False}
    allowed = (
        "embedding_backend",
        "embedding_model",
        "embedding_revision",
        "embedding_device",
        "embedding_onnx_filename",
        "embedding_onnx_threads",
    )
    return {key: payload.get(key) for key in allowed if key in payload}


def load_dataset(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"LoCoMo dataset is empty or invalid: {path}")
    return payload


def dialogue_facts(sample: dict[str, Any]) -> list[FactCard]:
    conversation = sample.get("conversation", {})
    sample_id = str(sample.get("sample_id", "unknown"))
    facts: list[FactCard] = []
    session_keys = sorted(
        (
            key
            for key, value in conversation.items()
            if key.startswith("session_")
            and not key.endswith("_date_time")
            and isinstance(value, list)
        ),
        key=lambda key: int(key.removeprefix("session_")),
    )
    for session_key in session_keys:
        for turn in conversation[session_key]:
            event_id = str(turn.get("dia_id", "")).strip()
            text = str(turn.get("text", "")).strip()
            caption = str(turn.get("blip_caption", "")).strip()
            if caption:
                text = f"{text}\nImage: {caption}" if text else f"Image: {caption}"
            if not event_id or not text:
                continue
            speaker = str(turn.get("speaker", "speaker")).strip() or "speaker"
            facts.append(
                FactCard(
                    entity=speaker,
                    attribute=event_id,
                    value=text,
                    evidence=text,
                    source_session_id=f"{sample_id}:{session_key}",
                    source_event_id=event_id,
                    confidence=1.0,
                    tags=["benchmark", "locomo", "dialogue_turn"],
                    importance=0.5,
                )
            )
    return facts


def select_cases(
    samples: list[dict[str, Any]], *, limit: int, seed: int
) -> list[RetrievalCase]:
    cases: list[RetrievalCase] = []
    for sample in samples:
        sample_id = str(sample.get("sample_id", "unknown"))
        for index, row in enumerate(sample.get("qa", [])):
            evidence = tuple(
                str(item).strip()
                for item in row.get("evidence", [])
                if str(item).strip()
            )
            question = str(row.get("question", "")).strip()
            if not evidence or not question:
                continue
            cases.append(
                RetrievalCase(
                    sample_id=sample_id,
                    question_index=index,
                    question=question,
                    evidence_ids=evidence,
                    category=int(row.get("category", 0)),
                )
            )
    random.Random(seed).shuffle(cases)
    return cases[:limit] if limit > 0 else cases


def _write_fixture(path: Path, facts: list[FactCard]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "core_facts.json").write_text(
        json.dumps([fact.to_dict() for fact in facts], ensure_ascii=False),
        encoding="utf-8",
    )


def _git_revision(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.strip()


def evaluate(
    dataset_path: Path, *, limit: int, seed: int, top_k: int
) -> dict[str, Any]:
    started = time.perf_counter()
    samples = load_dataset(dataset_path)
    selected = select_cases(samples, limit=limit, seed=seed)
    by_sample: dict[str, list[RetrievalCase]] = {}
    for case in selected:
        by_sample.setdefault(case.sample_id, []).append(case)
    sample_map = {str(sample.get("sample_id", "unknown")): sample for sample in samples}
    results: list[CaseResult] = []
    indexing_ms: list[float] = []

    with tempfile.TemporaryDirectory(prefix="osagent-locomo-") as tmp:
        for sample_id, cases in by_sample.items():
            root = Path(tmp) / sample_id
            memory_root = root / "memory"
            _write_fixture(memory_root, dialogue_facts(sample_map[sample_id]))
            module = AgentMemoryModule(
                memory_root,
                config=MemoryConfig(root / "memory-config.json"),
            )
            index_started = time.perf_counter()
            module.refresh_semantic_indexes()
            indexing_ms.append((time.perf_counter() - index_started) * 1000)
            for case in cases:
                recall_started = time.perf_counter()
                recalled = module.recall(
                    RecallRequest(query=case.question, scope="ltm", limit=top_k)
                )
                recall_ms = (time.perf_counter() - recall_started) * 1000
                event_ids = [
                    str(record.metadata.get("source_event_id", ""))
                    for record in recalled.records
                ]
                evidence = set(case.evidence_ids)
                first_rank = next(
                    (
                        rank
                        for rank, event_id in enumerate(event_ids, start=1)
                        if event_id in evidence
                    ),
                    None,
                )
                results.append(
                    CaseResult(
                        sample_id=sample_id,
                        question_index=case.question_index,
                        category=case.category,
                        evidence_count=len(case.evidence_ids),
                        retrieved_event_ids=event_ids,
                        first_evidence_rank=first_rank,
                        recall_ms=round(recall_ms, 3),
                    )
                )

    summary = summarize_results(results, indexing_ms, len(by_sample))
    return {
        "provenance": {
            "system": "OS Agent explicit memory",
            "dataset": DATASET_NAME,
            "dataset_revision": _git_revision(dataset_path.parent.parent),
            "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "sample_seed": seed,
            "sample_limit": limit,
            "top_k": top_k,
            "evaluated_at": datetime.now(UTC).isoformat(),
            "llm_used": False,
            **embedding_provenance(),
            "metric_scope": "evidence-labelled retrieval; not preference extraction or conflict handling",
        },
        "summary": summary,
        "cases": [asdict(row) for row in results],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/benchmark-sources/locomo/data/locomo10.json"),
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("doc/evaluation/results/locomo-os-agent-retrieval.json"),
    )
    args = parser.parse_args()
    result = evaluate(
        args.dataset_path,
        limit=max(args.limit, 0),
        seed=args.seed,
        top_k=max(args.top_k, 1),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
