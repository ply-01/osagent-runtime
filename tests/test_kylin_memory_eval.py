from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from benchmarks.generate_kylin_memory_eval import generate
from benchmarks.kylin_memory_eval import (
    TimedEmbeddingProvider,
    classification_metrics,
    record_matches,
    redact_case_details,
    require_complete_semantic_index,
    require_native_vector_backend,
    verify_dataset,
)


def test_published_result_omits_case_level_failure_details() -> None:
    result = {
        "preference_extraction": {"accuracy": 0.5, "failure_ids": ["private"]},
        "knowledge_retrieval": {"recall_at_5": 0.5, "failures": [{"private": 1}]},
        "conflict_handling": {"accuracy": 0.5, "failures": [{"private": 2}]},
    }

    public = redact_case_details(result)

    assert public["preference_extraction"] == {"accuracy": 0.5}
    assert public["knowledge_retrieval"] == {"recall_at_5": 0.5}
    assert public["conflict_handling"] == {"accuracy": 0.5}
    assert result["preference_extraction"]["failure_ids"] == ["private"]


def test_generator_seals_balanced_inputs_and_separates_labels(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    manifest = generate(root, seed="unit-test-seed")
    verified, files = verify_dataset(root)

    assert verified == manifest
    assert manifest["protocol"]["preference_cases"] == 500
    assert manifest["protocol"]["retrieval_corpus_facts"] == 10000
    assert manifest["protocol"]["retrieval_queries"] == 500
    assert manifest["protocol"]["conflict_cases"] == 500

    preference_inputs = [
        json.loads(line)
        for line in files["preference_inputs"].read_text(encoding="utf-8").splitlines()
    ]
    assert Counter(row["category"] for row in preference_inputs) == {
        "explicit_zh": 50,
        "explicit_en": 50,
        "explicit_avoid": 50,
        "configuration": 50,
        "repeated_behavior": 50,
        "question": 50,
        "quoted_or_reported": 50,
        "one_off_request": 50,
        "uncertain": 50,
        "system_statement": 50,
    }
    all_inputs = "\n".join(
        files[name].read_text(encoding="utf-8")
        for name in (
            "preference_inputs",
            "retrieval_corpus",
            "retrieval_queries",
            "conflict_inputs",
        )
    )
    for label_key in (
        "expected_preference",
        "expected_state",
        "relevant_fact_ids",
        "forbidden_records",
    ):
        assert label_key not in all_inputs


def test_generator_content_is_deterministic_for_same_secret_seed(tmp_path: Path) -> None:
    first = generate(tmp_path / "first", seed="same-seed")
    second = generate(tmp_path / "second", seed="same-seed")

    assert {
        name: row["sha256"] for name, row in first["files"].items()
    } == {
        name: row["sha256"] for name, row in second["files"].items()
    }


def test_classification_metrics_are_balanced_and_auditable() -> None:
    result = classification_metrics(
        [True, True, False, False],
        [True, False, True, False],
    )

    assert result["accuracy"] == 0.5
    assert result["balanced_accuracy"] == 0.5
    assert result["tp"] == result["tn"] == result["fp"] == result["fn"] == 1


def test_benchmark_rejects_silent_semantic_index_fallback() -> None:
    provider = type("Provider", (), {"enabled": True})()

    try:
        require_complete_semantic_index(provider, {"core_facts": 0}, 10000)
    except RuntimeError as exc:
        assert "expected 10000, indexed 0" in str(exc)
    else:
        raise AssertionError("incomplete semantic indexing must fail the benchmark")

    require_complete_semantic_index(
        provider, {"core_facts": 10000}, 10000
    )


def test_native_vector_guard_rejects_non_native_backend() -> None:
    provider = type("Provider", (), {"is_kylin_system": True})()

    try:
        require_native_vector_backend(provider, None)
    except RuntimeError as exc:
        assert "actual backend is none" in str(exc)
    else:
        raise AssertionError("missing native vector backend must fail the benchmark")

    vector_store = type("KylinVectorEngine", (), {})()
    require_native_vector_backend(provider, vector_store)


def test_timed_embedding_provider_does_not_retain_text() -> None:
    class Provider:
        enabled = True
        cache_namespace = "test"

        @staticmethod
        def embed_queries(texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    provider = TimedEmbeddingProvider(Provider())

    assert provider.embed_queries(["private query"]) == [[1.0, 0.0]]
    assert provider.events[0]["operation"] == "embed_queries"
    assert provider.events[0]["text_count"] == 1
    assert "private query" not in json.dumps(provider.events)


def test_record_matching_requires_all_tokens_in_one_memory() -> None:
    records = ["输出格式 PDF", "项目甲 uses ODT"]

    assert record_matches(["输出格式", "PDF"], records)
    assert not record_matches(["项目甲", "PDF"], records)
