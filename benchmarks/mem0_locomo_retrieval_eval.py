"""Evaluate Mem0 OSS retrieval on the same LoCoMo evidence slice as OS Agent."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__:
    from benchmarks.locomo_retrieval_eval import (
        DATASET_NAME,
        DATASET_REVISION,
        SEED,
        CaseResult,
        dialogue_facts,
        load_dataset,
        select_cases,
        summarize_results,
    )
else:
    from locomo_retrieval_eval import (
        DATASET_NAME,
        DATASET_REVISION,
        SEED,
        CaseResult,
        dialogue_facts,
        load_dataset,
        select_cases,
        summarize_results,
    )

MEM0_VERSION = "2.0.14"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
EMBEDDING_DIMENSIONS = 384


def _memory_config() -> dict[str, Any]:
    return {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "locomo_comparison",
                "path": ":memory:",
                "embedding_model_dims": EMBEDDING_DIMENSIONS,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": EMBEDDING_MODEL,
                "embedding_dims": EMBEDDING_DIMENSIONS,
                "model_kwargs": {"revision": EMBEDDING_REVISION},
            },
        },
        # Mem0 constructs an LLM even when infer=False. This adapter never calls it.
        "llm": {
            "provider": "deepseek",
            "config": {
                "model": "deepseek-chat",
                "api_key": "unused-for-infer-false",
            },
        },
        "history_db_path": ":memory:",
    }


def evaluate(
    dataset_path: Path, *, limit: int, seed: int, top_k: int
) -> dict[str, Any]:
    from mem0 import Memory

    started = time.perf_counter()
    samples = load_dataset(dataset_path)
    selected = select_cases(samples, limit=limit, seed=seed)
    by_sample: dict[str, list[Any]] = {}
    for case in selected:
        by_sample.setdefault(case.sample_id, []).append(case)
    sample_map = {str(sample.get("sample_id", "unknown")): sample for sample in samples}

    memory = Memory.from_config(_memory_config())
    results: list[CaseResult] = []
    indexing_ms: list[float] = []
    try:
        for sample_id, cases in by_sample.items():
            index_started = time.perf_counter()
            for fact in dialogue_facts(sample_map[sample_id]):
                memory.add(
                    str(fact.value),
                    user_id=sample_id,
                    metadata={
                        "source_event_id": fact.source_event_id,
                        "source_session_id": fact.source_session_id,
                    },
                    infer=False,
                )
            indexing_ms.append((time.perf_counter() - index_started) * 1000)

            for case in cases:
                recall_started = time.perf_counter()
                response = memory.search(
                    case.question,
                    filters={"user_id": sample_id},
                    top_k=top_k,
                    threshold=0.0,
                )
                recall_ms = (time.perf_counter() - recall_started) * 1000
                hits = response.get("results", [])
                event_ids = [
                    str((row.get("metadata") or {}).get("source_event_id", ""))
                    for row in hits
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
    finally:
        client = getattr(memory.vector_store, "client", None)
        if client is not None and hasattr(client, "close"):
            client.close()

    summary = summarize_results(results, indexing_ms, len(by_sample))
    return {
        "provenance": {
            "system": "Mem0 OSS",
            "system_version": MEM0_VERSION,
            "dataset": DATASET_NAME,
            "dataset_revision": DATASET_REVISION,
            "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "sample_seed": seed,
            "sample_limit": limit,
            "top_k": top_k,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_revision": EMBEDDING_REVISION,
            "llm_used": False,
            "optional_capabilities": {
                "spacy": importlib.util.find_spec("spacy") is not None,
                "fastembed_bm25": importlib.util.find_spec("fastembed") is not None,
            },
            "ingestion": "public Memory.add with infer=False; one dialogue turn per memory",
            "evaluated_at": datetime.now(UTC).isoformat(),
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
        default=Path("doc/evaluation/results/locomo-mem0-retrieval.json"),
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
