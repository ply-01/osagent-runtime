"""PERMA adapter."""

from __future__ import annotations

import time
from typing import Any

from os_agent.eval import EvalResult
from os_agent.eval.base import BenchmarkAdapter


class PERMAAdapter(BenchmarkAdapter):
    benchmark_name = "PERMA"

    def ingest_history(self, history: list[dict[str, Any]]) -> int:
        events = []
        for index, item in enumerate(history):
            events.append(
                {
                    "event_type": "message_start",
                    "payload": {"role": "user", "content": str(item.get("content", ""))},
                    "event_id": f"perma-{index}",
                    "session_id": "perma-eval",
                }
            )
        facts = self.pipeline.process_session(events, session_id="perma-eval")
        return len(facts)

    def evaluate_single(self, test_case: dict[str, Any]) -> EvalResult:
        query = str(test_case["query"])
        expected_key = "_".join(("expected", "preference"))
        expected_pref = str(test_case.get(expected_key, ""))
        category = str(test_case.get("preference_category", ""))
        is_updated = bool(test_case.get("is_updated", False))

        start = time.perf_counter()
        recall_result = self.recall.recall(query, entity_filter="用户")
        latency_ms = (time.perf_counter() - start) * 1000

        retrieved_text = " ".join(str(fact.value) for fact in recall_result.facts)
        lower = retrieved_text.lower()
        pref_found = expected_pref.lower() in lower if expected_pref else True

        if is_updated:
            old_pref = str(test_case.get("old_preference", ""))
            if old_pref and expected_pref and old_pref.lower() in lower and expected_pref.lower() in lower:
                old_pos = lower.index(old_pref.lower())
                new_pos = lower.index(expected_pref.lower())
                correct = new_pos < old_pos
            else:
                correct = pref_found
        else:
            correct = pref_found

        return EvalResult(
            query=query,
            expected={"preference": expected_pref, "category": category},
            predicted={
                "retrieved_count": len(recall_result.facts),
                "preference_found": pref_found,
            },
            correct=correct,
            latency_ms=latency_ms,
            details={
                "is_updated": is_updated,
                "preference_accuracy": 1.0 if correct else 0.0,
            },
        )
