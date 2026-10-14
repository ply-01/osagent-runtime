"""LongMemEval-V2 adapter."""

from __future__ import annotations

import time
from typing import Any

from os_agent.eval import EvalResult
from os_agent.eval.base import BenchmarkAdapter


class LongMemEvalV2Adapter(BenchmarkAdapter):
    benchmark_name = "LongMemEval-V2"

    def ingest_history(self, history: list[dict[str, Any]]) -> int:
        total_facts = 0
        for index, trajectory in enumerate(history):
            events = []
            actions = trajectory.get("actions", [])
            observations = trajectory.get("observations", [])
            result = str(trajectory.get("result", "success"))
            for step, (action, observation) in enumerate(zip(actions, observations)):
                events.append(
                    {
                        "event_type": "tool_result",
                        "payload": {
                            "tool_name": action.get("tool", "unknown"),
                            "status": "success" if result == "success" else "error",
                            "input": action.get("arguments", {}),
                            "output": {"content": observation},
                        },
                        "event_id": f"lme-{index}-{step}",
                        "session_id": f"lme-eval-{index}",
                    }
                )
            facts = self.pipeline.process_session(events, session_id=f"lme-eval-{index}")
            total_facts += len(facts)
        return total_facts

    def evaluate_single(self, test_case: dict[str, Any]) -> EvalResult:
        query = str(test_case["query"])
        expected = str(test_case.get("expected_answer", ""))
        task_type = str(test_case.get("task_type", "workflow_knowledge"))

        start = time.perf_counter()
        recall_result = self.recall.recall(query)
        latency_ms = (time.perf_counter() - start) * 1000

        retrieved_text = " ".join(str(fact.value) for fact in recall_result.facts)
        correct = expected.lower() in retrieved_text.lower() if expected else True

        return EvalResult(
            query=query,
            expected={"answer": expected, "task_type": task_type},
            predicted={
                "retrieved_count": len(recall_result.facts),
                "answer_found": correct,
            },
            correct=correct,
            latency_ms=latency_ms,
            details={
                "task_type": task_type,
                "latency_within_500ms": latency_ms <= 500,
            },
        )

