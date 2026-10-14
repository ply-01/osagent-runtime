"""Memora adapter."""

from __future__ import annotations

import time
from typing import Any

from os_agent.eval import EvalResult
from os_agent.eval.base import BenchmarkAdapter


class MemoraAdapter(BenchmarkAdapter):
    benchmark_name = "Memora"

    def ingest_history(self, history: list[dict[str, Any]]) -> int:
        events = []
        for index, item in enumerate(history):
            events.append(
                {
                    "event_type": "message_start",
                    "payload": {"role": "user", "content": str(item.get("content", ""))},
                    "event_id": f"memora-{index}",
                    "session_id": "memora-eval",
                }
            )
        facts = self.pipeline.process_session(events, session_id="memora-eval")
        return len(facts)

    def execute_forget(self, instruction: str) -> bool:
        results = self.core.search(instruction, top_k=5)
        deleted = 0
        for fact in results:
            if instruction_matches_fact(instruction, fact):
                if self.core.remove(fact.entity, fact.attribute):
                    deleted += 1
        return deleted > 0

    def evaluate_single(self, test_case: dict[str, Any]) -> EvalResult:
        task_type = str(test_case.get("task_type", "forgetting"))
        query = str(test_case.get("query_after_forget", test_case.get("query", "")))
        expected = str(test_case.get("expected_answer", ""))
        forgotten_info = str(test_case.get("forgotten_info", ""))
        forget_instruction = str(test_case.get("forget_instruction", ""))

        if forget_instruction:
            self.execute_forget(forget_instruction)

        start = time.perf_counter()
        recall_result = self.recall.recall(query)
        latency_ms = (time.perf_counter() - start) * 1000

        retrieved_text = " ".join(str(fact.value) for fact in recall_result.facts)
        lower = retrieved_text.lower()

        if task_type == "forgetting":
            leaked = forgotten_info.lower() in lower if forgotten_info else False
            expected_found = expected.lower() in lower if expected else True
            correct = (not leaked) and expected_found
            details = {
                "leaked": leaked,
                "expected_found": expected_found,
                "leakage_rate": 1.0 if leaked else 0.0,
            }
        else:
            correct = expected.lower() in lower if expected else True
            details = {"answer_found": correct}

        return EvalResult(
            query=query,
            expected={"answer": expected, "forgotten": forgotten_info},
            predicted={
                "retrieved_count": len(recall_result.facts),
                "leaked": forgotten_info.lower() in lower if forgotten_info else False,
            },
            correct=correct,
            latency_ms=latency_ms,
            details=details,
        )


def instruction_matches_fact(instruction: str, fact: Any) -> bool:
    inst_lower = instruction.lower()
    return (
        fact.attribute.lower() in inst_lower
        or fact.entity.lower() in inst_lower
        or any(tag in inst_lower for tag in fact.tags)
    )

