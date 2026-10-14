"""Blind runner for the sealed KylinMemoryEval-v1 competition protocol."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from os_agent.memory.embedding_cache import SqliteEmbeddingCache
from os_agent.memory.embedding_provider import create_embedding_provider
from os_agent.memory.fact_vector_index import FactSemanticIndex
from os_agent.memory.interface import RecallRequest, RememberRequest, UpdateRequest
from os_agent.memory.mem0_archival import ArchivalMemoryStore
from os_agent.memory.mem0_core import CoreMemoryStore
from os_agent.memory.mem0_pipeline import ExtractUpdatePipeline
from os_agent.memory.mem0_schemas import FactCard
from os_agent.memory.memory_config import MemoryConfig
from os_agent.memory.module import AgentMemoryModule
from os_agent.memory.session_vector_index import SessionSemanticIndex

DEFAULT_DATASET = ROOT / "data" / "kylin-memory-eval-v1"
DEFAULT_OUTPUT = ROOT / "doc" / "evaluation" / "results" / "kylin-memory-eval-v1.json"
EXTRACTION_DIAGNOSTICS: dict[str, dict[str, Any]] = {}


class TimedEmbeddingProvider:
    """Record provider latency without retaining input text or vectors."""

    def __init__(self, delegate: object) -> None:
        self.delegate = delegate
        self.events: list[dict[str, float | int | str]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._measure("embed_queries", texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._measure("embed_documents", texts)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._measure("embed_texts", texts)

    def _measure(self, operation: str, texts: list[str]) -> list[list[float]]:
        started = time.perf_counter()
        try:
            method = getattr(self.delegate, operation)
            return method(texts)
        finally:
            self.events.append(
                {
                    "operation": operation,
                    "text_count": len(texts),
                    "latency_ms": (time.perf_counter() - started) * 1000,
                }
            )


def timed_event_total(events: list[dict[str, float | int | str]], start: int = 0) -> float:
    return sum(float(event["latency_ms"]) for event in events[start:])


def run_prediction_cases(rows, base, worker, section):
    """Parallelize isolated cases while preserving each case's chronology."""

    workers = min(max(int(os.getenv("OS_AGENT_EVAL_WORKERS", "1")), 1), 16)
    totals = Counter()
    predictions = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, row, base) for row in rows]
        for index, future in enumerate(as_completed(futures), 1):
            case_id, prediction, stats = future.result()
            predictions[case_id] = prediction
            totals.update(stats)
            if index % 100 == 0:
                print(
                    f"{section}: {index}/{len(rows)} predictions completed",
                    flush=True,
                )
    EXTRACTION_DIAGNOSTICS[section] = {"workers": workers, **dict(totals)}
    return predictions


def process_natural_sessions(pipeline, sessions):
    for session in sessions:
        session_id = str(session[0].get("session_id", "")) if session else ""
        pipeline.process_session(session, session_id)
    return pipeline.runtime_stats


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "mean": round(statistics.fmean(values), 3) if values else 0.0,
        "p50": round(percentile(values, 0.50), 3),
        "p95": round(percentile(values, 0.95), 3),
        "p99": round(percentile(values, 0.99), 3),
        "max": round(max(values), 3) if values else 0.0,
    }


def classification_metrics(labels: list[bool], predictions: list[bool]) -> dict[str, Any]:
    tp = sum(label and prediction for label, prediction in zip(labels, predictions))
    tn = sum(not label and not prediction for label, prediction in zip(labels, predictions))
    fp = sum(not label and prediction for label, prediction in zip(labels, predictions))
    fn = sum(label and not prediction for label, prediction in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "cases": len(labels),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": round((tp + tn) / max(1, len(labels)), 4),
        "balanced_accuracy": round((recall + specificity) / 2, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
    }


def verify_dataset(dataset_root: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != "KylinMemoryEval-v1" or not manifest.get("sealed"):
        raise ValueError("dataset is not a sealed KylinMemoryEval-v1 release")
    files: dict[str, Path] = {}
    for name, metadata in manifest["files"].items():
        path = dataset_root / metadata["path"]
        if not path.is_file() or sha256(path) != metadata["sha256"]:
            raise ValueError(f"dataset integrity check failed: {name}")
        files[name] = path
    return manifest, files


def make_pipeline(root: Path) -> tuple[ExtractUpdatePipeline, CoreMemoryStore]:
    config = MemoryConfig(root / "config.json")
    core = CoreMemoryStore(root / "memory", config=config)
    archival = ArchivalMemoryStore(root / "memory", config=config)
    return ExtractUpdatePipeline(core, archival, config=config), core


def predict_preferences(path: Path) -> tuple[dict[str, dict[str, Any]], float]:
    rows = read_jsonl(path)
    started = time.perf_counter()

    def worker(row, base):
        pipeline, core = make_pipeline(base / row["case_id"])
        stats = process_natural_sessions(pipeline, row["sessions"])
        facts = [fact for fact in core.active_facts() if "preference" in fact.tags]
        return (
            row["case_id"],
            {
                "predicted": bool(facts),
                "attributes": [fact.attribute for fact in facts],
                "values": [str(fact.value) for fact in facts],
            },
            stats,
        )

    with tempfile.TemporaryDirectory(prefix="kme-pref-") as tmp:
        predictions = run_prediction_cases(
            rows, Path(tmp), worker, "preferences"
        )
    return predictions, time.perf_counter() - started


def score_preferences(
    predictions: dict[str, dict[str, Any]],
    labels_path: Path,
    inputs_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = {row["case_id"]: row for row in read_jsonl(labels_path)}
    inputs = {row["case_id"]: row for row in read_jsonl(inputs_path)}
    expected: list[bool] = []
    actual: list[bool] = []
    by_category: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    failures: list[dict[str, Any]] = []
    grounded = 0
    positive = 0
    normalized_attribute_hits = 0
    normalized_attribute_total = 0
    for case_id, label in labels.items():
        prediction = predictions[case_id]
        truth = bool(label["expected_preference"])
        predicted = bool(prediction["predicted"])
        expected.append(truth)
        actual.append(predicted)
        category = str(inputs[case_id]["category"])
        by_category[category].append((truth, predicted))
        if truth:
            positive += 1
            value = str(label.get("expected_value") or "").lower()
            grounded += int(any(value in item.lower() for item in prediction["values"]))
            if category != "repeated_behavior":
                normalized_attribute_total += 1
                normalized_attribute_hits += int(
                    str(label["expected_attribute"]) in prediction["attributes"]
                )
        if truth != predicted:
            failures.append({"case_id": case_id, "category": category, "expected": truth, "predicted": predicted})
    metrics = classification_metrics(expected, actual)
    metrics.update(
        {
            "value_grounding_rate": round(grounded / max(1, positive), 4),
            "normalized_attribute_accuracy": round(
                normalized_attribute_hits / max(1, normalized_attribute_total), 4
            ),
            "by_category": {
                category: classification_metrics(
                    [pair[0] for pair in pairs], [pair[1] for pair in pairs]
                )
                for category, pairs in sorted(by_category.items())
            },
        }
    )
    return metrics, failures


def retrieval_fact(row: dict[str, Any]) -> FactCard:
    return FactCard(
        fact_id=str(row["fact_id"]),
        entity=str(row["entity"]),
        attribute=str(row["attribute"]),
        value=row["value"],
        evidence=str(row["evidence"]),
        source_session_id=f"corpus-{row['fact_id']}",
        source_event_id=f"event-{row['fact_id']}",
        confidence=0.8,
        importance=0.6,
        tags=["kylin_eval", str(row["source_type"])],
        created_at="2026-06-01T00:00:00+00:00",
        updated_at="2026-06-01T00:00:00+00:00",
    )


def predict_retrieval(
    corpus_path: Path,
    queries_path: Path,
    *,
    top_k: int,
    require_kylin_vector: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    corpus = read_jsonl(corpus_path)
    queries = read_jsonl(queries_path)
    with tempfile.TemporaryDirectory(prefix="kme-retrieval-") as tmp:
        runtime_root = Path(tmp)
        memory_root = runtime_root / "memory"
        memory_root.mkdir(parents=True, exist_ok=True)
        facts = [retrieval_fact(row) for row in corpus]
        (memory_root / "core_facts.json").write_text(
            json.dumps([fact.to_dict() for fact in facts], ensure_ascii=False),
            encoding="utf-8",
        )
        cache = SqliteEmbeddingCache(runtime_root / "embeddings.sqlite3")
        raw_provider = create_embedding_provider()
        provider = TimedEmbeddingProvider(raw_provider)
        vector_store_root = memory_root if require_kylin_vector else None
        semantic_index = FactSemanticIndex(
            provider=provider,
            cache=cache,
            vector_store_root=vector_store_root,
        )
        session_index = SessionSemanticIndex(
            provider=provider,
            cache=cache,
            vector_store_root=vector_store_root,
        )
        module = AgentMemoryModule(
            memory_root,
            config=MemoryConfig(runtime_root / "config.json"),
            semantic_index=semantic_index,
            session_semantic_index=session_index,
            embedding_provider=provider,
        )
        index_started = time.perf_counter()
        indexed = module.refresh_semantic_indexes()
        index_ms = (time.perf_counter() - index_started) * 1000
        require_complete_semantic_index(provider, indexed, len(facts))
        vector_store = semantic_index.vector_store
        if require_kylin_vector:
            require_native_vector_backend(provider, vector_store)

        native_search_latencies: list[float] = []
        if vector_store is not None:
            original_search = vector_store.search

            def timed_native_search(*args: Any, **kwargs: Any) -> Any:
                started = time.perf_counter()
                try:
                    return original_search(*args, **kwargs)
                finally:
                    native_search_latencies.append(
                        (time.perf_counter() - started) * 1000
                    )

            vector_store.search = timed_native_search

        index_embedding_ms = timed_event_total(provider.events)
        predictions: dict[str, dict[str, Any]] = {}
        query_latencies: list[float] = []
        query_embedding_latencies: list[float] = []
        native_query_latencies: list[float] = []
        other_query_latencies: list[float] = []
        query_embedding_texts = 0
        try:
            for row in queries:
                embedding_start = len(provider.events)
                native_start = len(native_search_latencies)
                started = time.perf_counter()
                result = module.recall(
                    RecallRequest(query=str(row["query"]), scope="ltm", limit=top_k)
                )
                latency = (time.perf_counter() - started) * 1000
                embedding_ms = timed_event_total(provider.events, embedding_start)
                native_ms = sum(native_search_latencies[native_start:])
                other_ms = max(0.0, latency - embedding_ms - native_ms)
                query_embedding_texts += sum(
                    int(event["text_count"])
                    for event in provider.events[embedding_start:]
                )
                query_latencies.append(latency)
                query_embedding_latencies.append(embedding_ms)
                native_query_latencies.append(native_ms)
                other_query_latencies.append(other_ms)
                predictions[row["case_id"]] = {
                    "returned_fact_ids": [record.memory_id for record in result.records],
                    "latency_ms": latency,
                }
        finally:
            close_vector_store = getattr(vector_store, "close", None)
            if callable(close_vector_store):
                close_vector_store()
            close_provider = getattr(raw_provider, "close", None)
            if callable(close_provider):
                close_provider()
    return predictions, {
        "corpus_facts": len(corpus),
        "queries": len(queries),
        "indexed": indexed,
        "embedding_backend": type(raw_provider).__name__,
        "vector_backend": type(vector_store).__name__ if vector_store is not None else "numpy",
        "native_vector_required": require_kylin_vector,
        "cold_index_ms": round(index_ms, 3),
        "cold_index_stage_ms": {
            "document_embedding": round(index_embedding_ms, 3),
            "non_embedding": round(max(0.0, index_ms - index_embedding_ms), 3),
        },
        "online_latency_ms": latency_summary(query_latencies),
        "online_stage_latency_ms": {
            "query_embedding": latency_summary(query_embedding_latencies),
            "native_vector_search": latency_summary(native_query_latencies),
            "other_recall": latency_summary(other_query_latencies),
        },
        "query_embedding_texts": query_embedding_texts,
        "native_search_calls": len(native_search_latencies),
    }


def require_complete_semantic_index(
    provider: object, indexed: dict[str, int], corpus_facts: int
) -> None:
    """Reject benchmark runs that silently lost the configured dense path."""
    if not bool(getattr(provider, "enabled", False)):
        return
    indexed_core = int(indexed.get("core_facts", 0))
    if indexed_core != corpus_facts:
        raise RuntimeError(
            "configured semantic retrieval did not index the complete corpus: "
            f"expected {corpus_facts}, indexed {indexed_core}"
        )


def require_native_vector_backend(provider: object, vector_store: object | None) -> None:
    """Reject target-machine runs that did not open the native vector engine."""
    if not bool(getattr(provider, "is_kylin_system", False)):
        raise RuntimeError("native vector evaluation requires the Kylin system provider")
    if vector_store is None or type(vector_store).__name__ != "KylinVectorEngine":
        actual = type(vector_store).__name__ if vector_store is not None else "none"
        raise RuntimeError(
            "native vector evaluation did not open KylinVectorEngine: "
            f"actual backend is {actual}"
        )


def score_retrieval(
    predictions: dict[str, dict[str, Any]],
    qrels_path: Path,
    inputs_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    qrels = {row["case_id"]: row for row in read_jsonl(qrels_path)}
    inputs = {row["case_id"]: row for row in read_jsonl(inputs_path)}
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    complete_hits = 0
    by_category: dict[str, list[float]] = defaultdict(list)
    failures: list[dict[str, Any]] = []
    for case_id, label in qrels.items():
        relevant = set(label["relevant_fact_ids"])
        returned = predictions[case_id]["returned_fact_ids"]
        found = relevant & set(returned)
        recall = len(found) / len(relevant)
        recalls.append(recall)
        complete_hits += int(found == relevant)
        ranks = [returned.index(fact_id) + 1 for fact_id in relevant if fact_id in returned]
        reciprocal_ranks.append(1 / min(ranks) if ranks else 0.0)
        category = str(inputs[case_id]["category"])
        by_category[category].append(recall)
        if recall < 1.0:
            failures.append(
                {
                    "case_id": case_id,
                    "category": category,
                    "relevant_count": len(relevant),
                    "retrieved_relevant": len(found),
                }
            )
    return (
        {
            "queries": len(recalls),
            "recall_at_5": round(statistics.fmean(recalls), 4),
            "complete_recall_at_5": round(complete_hits / max(1, len(recalls)), 4),
            "mrr": round(statistics.fmean(reciprocal_ranks), 4),
            "by_category": {
                category: {
                    "queries": len(values),
                    "recall_at_5": round(statistics.fmean(values), 4),
                }
                for category, values in sorted(by_category.items())
            },
        },
        failures,
    )


def predict_conflicts(path: Path, *, capture_natural_state: bool = False) -> tuple[dict[str, dict[str, Any]], float]:
    rows = read_jsonl(path)
    started = time.perf_counter()

    def worker(row, base):
        case_root = base / row["case_id"]
        module = AgentMemoryModule(
            case_root / "structured" / "memory",
            config=MemoryConfig(case_root / "structured" / "config.json"),
        )
        for operation in row["structured_operations"]:
            request_kwargs = {
                "content": str(operation["value"]),
                "entity": str(operation["entity"]),
                "attribute": str(operation["attribute"]),
                "value": operation["value"],
            }
            if operation["action"] == "update":
                module.update(UpdateRequest(**request_kwargs))
            else:
                module.remember(RememberRequest(**request_kwargs))
        structured_state = sorted(
            (
                {
                    "entity": fact.entity,
                    "attribute": fact.attribute,
                    "value": str(fact.value),
                }
                for fact in module.core.active_facts()
            ),
            key=lambda item: (item["entity"], item["attribute"]),
        )
        pipeline, core = make_pipeline(case_root / "natural")
        stats = process_natural_sessions(pipeline, row["natural_sessions"])
        natural_records = [
            f"{fact.attribute} {fact.value}"
            for fact in core.active_facts()
            if "preference" in fact.tags
        ]
        return (
            row["case_id"],
            {
                "structured_state": structured_state,
                "natural_records": natural_records,
                **({"natural_state": [fact.to_dict() for fact in core.active_facts()
                                       if "preference" in fact.tags]} if capture_natural_state else {}),
            },
            stats,
        )

    with tempfile.TemporaryDirectory(prefix="kme-conflict-") as tmp:
        predictions = run_prediction_cases(
            rows, Path(tmp), worker, "conflicts"
        )
    return predictions, time.perf_counter() - started


def record_matches(tokens: list[str], records: list[str]) -> bool:
    return any(all(token.lower() in record.lower() for token in tokens) for record in records)


def score_conflicts(
    predictions: dict[str, dict[str, Any]],
    labels_path: Path,
    inputs_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = {row["case_id"]: row for row in read_jsonl(labels_path)}
    inputs = {row["case_id"]: row for row in read_jsonl(inputs_path)}
    structured_correct = 0
    natural_correct = 0
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"cases": 0, "structured": 0, "natural": 0})
    failures: list[dict[str, Any]] = []
    for case_id, label in labels.items():
        prediction = predictions[case_id]
        expected_state = [
            {**row, "value": str(row["value"])} for row in label["expected_state"]
        ]
        structured = prediction["structured_state"] == expected_state
        records = prediction["natural_records"]
        natural = all(
            record_matches(tokens, records) for tokens in label["expected_records"]
        ) and not any(
            record_matches(tokens, records) for tokens in label["forbidden_records"]
        )
        structured_correct += int(structured)
        natural_correct += int(natural)
        category = str(inputs[case_id]["category"])
        bucket = by_category[category]
        bucket["cases"] += 1
        bucket["structured"] += int(structured)
        bucket["natural"] += int(natural)
        if not structured or not natural:
            failures.append(
                {
                    "case_id": case_id,
                    "category": category,
                    "structured_correct": structured,
                    "natural_correct": natural,
                }
            )
    total = len(labels)
    return (
        {
            "cases": total,
            "structured_update_accuracy": round(structured_correct / max(1, total), 4),
            "natural_ingress_accuracy": round(natural_correct / max(1, total), 4),
            "by_category": {
                category: {
                    "cases": values["cases"],
                    "structured_update_accuracy": round(values["structured"] / values["cases"], 4),
                    "natural_ingress_accuracy": round(values["natural"] / values["cases"], 4),
                }
                for category, values in sorted(by_category.items())
            },
        },
        failures,
    )


def git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def evaluate(dataset_root: Path, *, require_kylin_vector: bool = False) -> dict[str, Any]:
    EXTRACTION_DIAGNOSTICS.clear()
    manifest, files = verify_dataset(dataset_root)
    started = time.perf_counter()

    preference_predictions, preference_seconds = predict_preferences(files["preference_inputs"])
    preference, preference_failures = score_preferences(
        preference_predictions,
        files["preference_labels"],
        files["preference_inputs"],
    )

    retrieval_predictions, retrieval_runtime = predict_retrieval(
        files["retrieval_corpus"],
        files["retrieval_queries"],
        top_k=5,
        require_kylin_vector=require_kylin_vector,
    )
    retrieval, retrieval_failures = score_retrieval(
        retrieval_predictions,
        files["retrieval_qrels"],
        files["retrieval_queries"],
    )

    conflict_predictions, conflict_seconds = predict_conflicts(files["conflict_inputs"])
    conflict, conflict_failures = score_conflicts(
        conflict_predictions,
        files["conflict_labels"],
        files["conflict_inputs"],
    )

    latency = retrieval_runtime["online_latency_ms"]
    thresholds = {
        "preference_accuracy": {
            "target": 0.85,
            "value": preference["accuracy"],
            "passed": preference["accuracy"] >= 0.85,
        },
        "knowledge_recall_at_5": {
            "target": 0.85,
            "value": retrieval["recall_at_5"],
            "passed": retrieval["recall_at_5"] >= 0.85,
        },
        "online_recall_p95_ms": {
            "target": 500.0,
            "value": latency["p95"],
            "passed": latency["p95"] <= 500.0,
        },
        "natural_conflict_accuracy": {
            "target": 0.88,
            "value": conflict["natural_ingress_accuracy"],
            "passed": conflict["natural_ingress_accuracy"] >= 0.88,
        },
    }
    return {
        "protocol": {
            "dataset": manifest["dataset"],
            "manifest_sha256": sha256(dataset_root / "manifest.json"),
            "seed_commitment": manifest["seed_commitment"],
            "evaluated_at": datetime.now(UTC).isoformat(),
            "agent_commit_before_run": git_commit(),
            "labels_visible_to_agent": False,
            "labels_loaded_after_prediction": True,
            "top_k": 5,
            "embedding_backend": retrieval_runtime["embedding_backend"],
            "vector_backend": retrieval_runtime["vector_backend"],
            "native_vector_required": require_kylin_vector,
            "vector_collection_prefix": (
                os.getenv("OS_AGENT_KYLIN_VECTOR_COLLECTION_PREFIX", "")
                if require_kylin_vector
                else ""
            ),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "memory_extractor": os.getenv("OS_AGENT_MEMORY_EXTRACTOR", "rules"),
        },
        "extraction_runtime": dict(EXTRACTION_DIAGNOSTICS),
        "competition_thresholds": thresholds,
        "preference_extraction": {
            **preference,
            "elapsed_seconds": round(preference_seconds, 3),
            "failure_count": len(preference_failures),
            "failure_ids": [row["case_id"] for row in preference_failures],
        },
        "knowledge_retrieval": {
            **retrieval,
            **retrieval_runtime,
            "failure_count": len(retrieval_failures),
            "failures": retrieval_failures,
        },
        "conflict_handling": {
            **conflict,
            "elapsed_seconds": round(conflict_seconds, 3),
            "failure_count": len(conflict_failures),
            "failures": conflict_failures,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def redact_case_details(result: dict[str, Any]) -> dict[str, Any]:
    """Return a publishable result without case identifiers or failure payloads."""
    public = copy.deepcopy(result)
    for section_name in (
        "preference_extraction",
        "knowledge_retrieval",
        "conflict_handling",
    ):
        section = public.get(section_name)
        if not isinstance(section, dict):
            continue
        section.pop("failure_ids", None)
        section.pop("failures", None)
    return public


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--require-kylin-vector",
        action="store_true",
        help="fail unless retrieval uses KylinSystemEmbeddingProvider and KylinVectorEngine",
    )
    args = parser.parse_args()
    os.environ.setdefault(
        "OS_AGENT_EMBEDDING_CONFIG", str(ROOT / "config" / "embedding.local.json")
    )
    result = evaluate(
        args.dataset_root,
        require_kylin_vector=args.require_kylin_vector,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(redact_case_details(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"thresholds": result["competition_thresholds"], "output": str(args.output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
