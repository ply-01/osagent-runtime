"""Evaluate Oh My Pi's Mnemopi backend on the shared LoCoMo slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
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

OH_MY_PI_VERSION = "17.3.4"
OH_MY_PI_REVISION = "de6b7974a0658e1fae8fac584368a33021ae668f"
MNEMOPI_PACKAGE_INTEGRITY = (
    "sha512-ihQ8CfyE5W5VOJ1r55X/wLMX/3uGfUhc9nAJ8taJHn4dXlQkAViBukKphbe8+"
    "bQy6LCk4GvpiMqvU5d/RF0zwA=="
)


def _git_revision(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _package_metadata(runtime_root: Path) -> tuple[str, str]:
    package = runtime_root / "node_modules/@oh-my-pi/pi-mnemopi/package.json"
    payload = json.loads(package.read_text(encoding="utf-8"))
    lock_path = runtime_root / "package-lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock_entry = lock.get("packages", {}).get(
        "node_modules/@oh-my-pi/pi-mnemopi",
        {},
    )
    package_version = str(payload.get("version", ""))
    lock_version = str(lock_entry.get("version", ""))
    if package_version != lock_version:
        raise RuntimeError(
            "Mnemopi package.json and package-lock.json versions do not match"
        )
    return package_version, str(lock_entry.get("integrity", ""))


def _input_payload(
    samples: list[dict[str, Any]],
    selected: list[Any],
    *,
    top_k: int,
    retrieval_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_sample: dict[str, list[Any]] = {}
    for case in selected:
        by_sample.setdefault(case.sample_id, []).append(case)
    sample_map = {str(sample.get("sample_id", "unknown")): sample for sample in samples}
    groups = []
    cases_by_id = {}
    for sample_id, cases in by_sample.items():
        case_rows = []
        for case in cases:
            case_id = f"{sample_id}:{case.question_index}"
            cases_by_id[case_id] = case
            case_rows.append({"case_id": case_id, "question": case.question})
        groups.append(
            {
                "sample_id": sample_id,
                "memories": [
                    {
                        "content": str(fact.value),
                        "source_event_id": fact.source_event_id,
                        "source_session_id": fact.source_session_id,
                    }
                    for fact in dialogue_facts(sample_map[sample_id])
                ],
                "cases": case_rows,
            }
        )
    return (
        {"top_k": top_k, "retrieval_mode": retrieval_mode, "groups": groups},
        cases_by_id,
    )


def evaluate(
    dataset_path: Path,
    *,
    limit: int,
    seed: int,
    top_k: int,
    bun: Path,
    runtime_root: Path,
    source_root: Path,
    runner: Path,
    retrieval_mode: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    source_revision = _git_revision(source_root)
    if source_revision != OH_MY_PI_REVISION:
        raise RuntimeError(
            f"Oh My Pi source is at {source_revision}, expected {OH_MY_PI_REVISION}"
        )
    package_version, package_integrity = _package_metadata(runtime_root)
    if package_version != OH_MY_PI_VERSION:
        raise RuntimeError(
            f"Mnemopi runtime is {package_version}, expected {OH_MY_PI_VERSION}"
        )
    if package_integrity != MNEMOPI_PACKAGE_INTEGRITY:
        raise RuntimeError(
            "Mnemopi package integrity does not match the pinned artifact"
        )

    samples = load_dataset(dataset_path)
    selected = select_cases(samples, limit=limit, seed=seed)
    input_payload, cases_by_id = _input_payload(
        samples,
        selected,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
    )
    results: list[CaseResult] = []
    indexing_ms: list[float] = []
    with tempfile.TemporaryDirectory(prefix="oh-my-pi-locomo-") as tmp:
        temp_root = Path(tmp)
        input_path = temp_root / "input.json"
        raw_output_path = temp_root / "output.json"
        input_path.write_text(
            json.dumps(input_payload, ensure_ascii=False), encoding="utf-8"
        )
        completed = subprocess.run(
            [
                str(bun),
                str(runner),
                "--input",
                str(input_path),
                "--output",
                str(raw_output_path),
                "--runtime-root",
                str(runtime_root),
                "--db-root",
                str(temp_root / "db"),
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1800,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout)[-4000:]
            raise RuntimeError(f"Oh My Pi runner failed: {detail}")
        raw = json.loads(raw_output_path.read_text(encoding="utf-8"))
        groups = raw.get("groups", [])
        for group in groups:
            indexing_ms.append(float(group.get("indexing_ms", 0.0)))
            for raw_case in group.get("cases", []):
                case_id = str(raw_case.get("case_id", ""))
                case = cases_by_id[case_id]
                event_ids = [
                    str(item) for item in raw_case.get("retrieved_event_ids", [])
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
                        sample_id=case.sample_id,
                        question_index=case.question_index,
                        category=case.category,
                        evidence_count=len(case.evidence_ids),
                        retrieved_event_ids=event_ids,
                        first_evidence_rank=first_rank,
                        recall_ms=round(float(raw_case.get("recall_ms", 0.0)), 3),
                    )
                )
    if len(results) != len(selected):
        raise RuntimeError(f"runner returned {len(results)} of {len(selected)} cases")

    return {
        "provenance": {
            "system": "Oh My Pi / Mnemopi",
            "system_version": OH_MY_PI_VERSION,
            "source_revision": source_revision,
            "package_integrity": package_integrity,
            "dataset": DATASET_NAME,
            "dataset_revision": DATASET_REVISION,
            "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "sample_seed": seed,
            "sample_limit": limit,
            "top_k": top_k,
            "retrieval_mode": retrieval_mode,
            "embedding_backend": None if retrieval_mode == "fts" else "fastembed",
            "llm_used": False,
            "ingestion": "public Mnemopi.remember; one dialogue turn per memory",
            "evaluated_at": datetime.now(UTC).isoformat(),
            "metric_scope": (
                "evidence-labelled retrieval; Mnemopi backend only, not full-agent "
                "task completion"
            ),
        },
        "summary": summarize_results(
            results, indexing_ms, len(input_payload["groups"])
        ),
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
    parser.add_argument("--bun", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("benchmarks/oh_my_pi_mnemopi_runner.ts"),
    )
    parser.add_argument(
        "--retrieval-mode", choices=("auto", "fts"), default="fts"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("doc/evaluation/results/locomo-oh-my-pi-retrieval.json"),
    )
    args = parser.parse_args()
    result = evaluate(
        args.dataset_path,
        limit=max(args.limit, 0),
        seed=args.seed,
        top_k=max(args.top_k, 1),
        bun=args.bun,
        runtime_root=args.runtime_root,
        source_root=args.source_root,
        runner=args.runner,
        retrieval_mode=args.retrieval_mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
