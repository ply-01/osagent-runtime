"""MemoryAgentBench adapter."""

from __future__ import annotations

import time
from typing import Any

from os_agent.eval import EvalResult
from os_agent.eval.base import BenchmarkAdapter


class MemoryAgentBenchAdapter(BenchmarkAdapter):
    benchmark_name = "MemoryAgentBench"

    def ingest_history(self, history: list[dict[str, Any]]) -> int:
        total_facts = 0
        for chunk in history:
            chunk_id = chunk.get("chunk_id", 0)
            events = [
                {
                    "event_type": "message_start",
                    "payload": {"role": "user", "content": str(chunk.get("content", ""))},
                    "event_id": f"mab-chunk-{chunk_id}",
                    "session_id": f"mab-eval-{chunk_id}",
                }
            ]
            facts = self.pipeline.process_session(events, session_id=f"mab-eval-{chunk_id}")
            total_facts += len(facts)
        return total_facts

    def evaluate_single(self, test_case: dict[str, Any]) -> EvalResult:
        query = str(test_case["query"])
        expected = str(test_case.get("expected_answer", ""))
        task_type = str(test_case.get("task_type", "accurate_retrieval"))
        old_answer = str(test_case.get("old_answer", ""))

        start = time.perf_counter()
        recall_result = self.recall.recall(query)
        latency_ms = (time.perf_counter() - start) * 1000

        retrieved_text = " ".join(str(fact.value) for fact in recall_result.facts)
        lower = retrieved_text.lower()
        answer_found = expected.lower() in lower if expected else True

        if task_type == "conflict_resolution" and old_answer:
            old_found = old_answer.lower() in lower
            if old_found and answer_found:
                correct = lower.index(expected.lower()) < lower.index(old_answer.lower())
            else:
                correct = answer_found
        else:
            correct = answer_found

        return EvalResult(
            query=query,
            expected={"answer": expected, "task_type": task_type},
            predicted={
                "retrieved_count": len(recall_result.facts),
                "answer_found": answer_found,
            },
            correct=correct,
            latency_ms=latency_ms,
            details={
                "task_type": task_type,
                "old_answer_still_present": old_answer.lower() in lower if old_answer else False,
            },
        )

