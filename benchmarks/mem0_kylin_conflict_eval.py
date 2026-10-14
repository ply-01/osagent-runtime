"""Evaluate Mem0 OSS on the sealed Kylin natural-language conflict track.

The backend receives only the conflict input sessions.  Labels are loaded only
after every prediction has been materialized, and the published artifact never
contains case identifiers, prompts, memories, or failure examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

if __package__:
    from benchmarks.kylin_memory_eval import (
        read_jsonl,
        record_matches,
        sha256,
        verify_dataset,
    )
else:
    from kylin_memory_eval import read_jsonl, record_matches, sha256, verify_dataset


MEM0_VERSION = "2.0.14"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
EMBEDDING_DIMENSIONS = 384
DEFAULT_DATASET = ROOT / "data" / "benchmarks" / "kylin-memory-eval-v5"
DEFAULT_OUTPUT = (
    ROOT
    / "doc"
    / "evaluation"
    / "results"
    / "external-memory-conflict"
    / "mem0-2.0.14.json"
)


def session_messages(session: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Map OSagent events to the public Mem0 conversation input contract."""

    messages: list[dict[str, str]] = []
    for event in session:
        if event.get("event_type") != "message_start":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict) or payload.get("role") != "user":
            continue
        content = str(payload.get("content", "")).strip()
        if content:
            messages.append({"role": "user", "content": content})
    return messages


def _case_user_id(case_id: str) -> str:
    return "kme-" + hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:24]


def _load_llm_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = ("api_key", "base_url", "model")
    missing = [key for key in required if not str(payload.get(key, "")).strip()]
    if missing:
        raise ValueError(f"LLM config is missing: {', '.join(missing)}")
    return payload


def _memory_config(root: Path, llm: dict[str, Any], worker: int) -> dict[str, Any]:
    return {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": f"kylin_conflict_{worker}",
                "path": str(root / f"qdrant-{worker}"),
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
        "llm": {
            "provider": "deepseek",
            "config": {
                "model": str(llm["model"]),
                "api_key": str(llm["api_key"]),
                "deepseek_base_url": str(llm["base_url"]),
                "temperature": 0.0,
                "max_tokens": 2000,
            },
        },
        "history_db_path": str(root / f"history-{worker}.db"),
    }


def _close_memory(memory: Any) -> None:
    close = getattr(memory, "close", None)
    if callable(close):
        close()
    clients: dict[int, Any] = {}
    for store_name in ("vector_store", "_entity_store"):
        client = getattr(getattr(memory, store_name, None), "client", None)
        if client is not None and hasattr(client, "close"):
            clients[id(client)] = client
    for client in clients.values():
        client.close()


def _predict_partition(
    rows: list[dict[str, Any]],
    *,
    root: Path,
    llm: dict[str, Any],
    worker: int,
) -> dict[str, list[str]]:
    from mem0 import Memory
    from mem0 import __version__ as installed_version

    if installed_version != MEM0_VERSION:
        raise RuntimeError(
            f"Mem0 runtime is {installed_version}, expected {MEM0_VERSION}"
        )
    memory = Memory.from_config(_memory_config(root, llm, worker))
    predictions: dict[str, list[str]] = {}
    try:
        for row in rows:
            case_id = str(row["case_id"])
            user_id = _case_user_id(case_id)
            for session in row["natural_sessions"]:
                messages = session_messages(session)
                if messages:
                    # Use Mem0's stock inference prompt and memory policy.  No
                    # benchmark-specific instruction or expected state is supplied.
                    memory.add(messages, user_id=user_id, infer=True)
            response = memory.get_all(filters={"user_id": user_id}, top_k=100)
            predictions[case_id] = [
                str(item.get("memory", ""))
                for item in response.get("results", [])
                if str(item.get("memory", "")).strip()
            ]
    finally:
        _close_memory(memory)
    return predictions


def predict_conflicts(
    inputs_path: Path,
    *,
    llm_config_path: Path,
    workers: int,
) -> tuple[dict[str, list[str]], float]:
    """Finish all backend predictions before a label path is accepted anywhere."""

    rows = read_jsonl(inputs_path)
    llm = _load_llm_config(llm_config_path)
    started = time.perf_counter()
    partitions = [rows[index::workers] for index in range(workers)]
    predictions: dict[str, list[str]] = {}
    with tempfile.TemporaryDirectory(prefix="mem0-kme-conflict-") as tmp:
        base = Path(tmp)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _predict_partition,
                    partition,
                    root=base,
                    llm=llm,
                    worker=index,
                )
                for index, partition in enumerate(partitions)
                if partition
            ]
            for future in as_completed(futures):
                predictions.update(future.result())
                print(
                    f"Mem0 conflict partitions completed: {len(predictions)}/{len(rows)}",
                    flush=True,
                )
    if len(predictions) != len(rows):
        raise RuntimeError(
            f"Mem0 returned {len(predictions)} of {len(rows)} conflict predictions"
        )
    return predictions, time.perf_counter() - started


def score_natural_conflicts(
    predictions: dict[str, list[str]],
    *,
    labels_path: Path,
    inputs_path: Path,
) -> dict[str, Any]:
    """Apply the unchanged natural-record matcher and return aggregates only."""

    labels = {row["case_id"]: row for row in read_jsonl(labels_path)}
    inputs = {row["case_id"]: row for row in read_jsonl(inputs_path)}
    correct = 0
    by_category: dict[str, dict[str, int]] = defaultdict(
        lambda: {"cases": 0, "correct": 0}
    )
    for case_id, label in labels.items():
        records = predictions[case_id]
        passed = all(
            record_matches(tokens, records) for tokens in label["expected_records"]
        ) and not any(
            record_matches(tokens, records) for tokens in label["forbidden_records"]
        )
        correct += int(passed)
        category = str(inputs[case_id]["category"])
        by_category[category]["cases"] += 1
        by_category[category]["correct"] += int(passed)
    total = len(labels)
    return {
        "cases": total,
        "natural_ingress_accuracy": round(correct / max(1, total), 4),
        "by_category": {
            category: {
                "cases": values["cases"],
                "natural_ingress_accuracy": round(
                    values["correct"] / max(1, values["cases"]), 4
                ),
            }
            for category, values in sorted(by_category.items())
        },
    }


def evaluate(
    dataset_root: Path,
    *,
    llm_config_path: Path,
    workers: int,
) -> dict[str, Any]:
    manifest, files = verify_dataset(dataset_root)
    predictions, prediction_seconds = predict_conflicts(
        files["conflict_inputs"],
        llm_config_path=llm_config_path,
        workers=workers,
    )
    # This is deliberately the first point at which labels are opened.
    score = score_natural_conflicts(
        predictions,
        labels_path=files["conflict_labels"],
        inputs_path=files["conflict_inputs"],
    )
    llm = _load_llm_config(llm_config_path)
    return {
        "protocol": {
            "dataset": manifest["dataset"],
            "manifest_sha256": sha256(dataset_root / "manifest.json"),
            "seed_commitment": manifest["seed_commitment"],
            "evaluated_at": datetime.now(UTC).isoformat(),
            "labels_visible_to_backend": False,
            "labels_loaded_after_prediction": True,
            "case_isolation": "one hashed Mem0 user_id per case",
            "workers": workers,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "system": {
            "name": "Mem0 OSS",
            "version": MEM0_VERSION,
            "memory_policy": "stock infer=True additive extraction",
            "llm_provider": "deepseek",
            "llm_model": str(llm["model"]),
            "temperature": 0.0,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_revision": EMBEDDING_REVISION,
        },
        "conflict_handling": {
            **score,
            "elapsed_seconds": round(prediction_seconds, 3),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--llm-config",
        type=Path,
        default=ROOT / "config" / "llm.local.json",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    workers = min(max(args.workers, 1), 4)
    os.environ.setdefault("MEM0_TELEMETRY", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    result = evaluate(
        args.dataset_root,
        llm_config_path=args.llm_config,
        workers=workers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "system": result["system"],
                "conflict_handling": result["conflict_handling"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
