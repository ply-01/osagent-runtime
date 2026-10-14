"""Mem2ActBench adapter."""

from __future__ import annotations

import time
from typing import Any

from os_agent.eval import EvalResult
from os_agent.eval.base import BenchmarkAdapter


class Mem2ActAdapter(BenchmarkAdapter):
    benchmark_name = "Mem2ActBench"

    def ingest_history(self, history: list[dict[str, Any]]) -> int:
        events: list[dict[str, Any]] = []
        for index, turn in enumerate(history):
            role = str(turn.get("role", "user"))
            content = str(turn.get("content", ""))
            if role == "assistant" and isinstance(turn.get("tool_calls"), list):
                for tool_index, tool_call in enumerate(turn["tool_calls"]):
                    events.append(
                        {
                            "event_type": "tool_result",
                            "payload": {
                                "tool_name": tool_call.get("name", ""),
                                "status": "success",
                                "input": tool_call.get("arguments", {}),
                                "output": tool_call.get("result", {}),
                            },
                            "event_id": f"mem2act-tool-{index}-{tool_index}",
                            "session_id": "mem2act-eval",
                        }
                    )
            events.append(
                {
                    "event_type": "message_start" if role == "user" else "tool_result",
                    "payload": {"role": role, "content": content},
                    "event_id": f"mem2act-{index}",
                    "session_id": "mem2act-eval",
                }
            )
        facts = self.pipeline.process_session(events, session_id="mem2act-eval")
        return len(facts)

    def evaluate_single(self, test_case: dict[str, Any]) -> EvalResult:
        query = str(test_case["query"])
        expected_tool = str(test_case.get("expected_tool", ""))
        expected_args = test_case.get("expected_args", {})
        expected_constraints = set(test_case.get("expected_constraints", []))

        start = time.perf_counter()
        recall_result = self.recall.recall(query)
        latency_ms = (time.perf_counter() - start) * 1000

        retrieved_text = " ".join(str(fact.value) for fact in recall_result.facts)
        lower = retrieved_text.lower()

        constraints_met = {item for item in expected_constraints if item.lower() in lower}
        constraint_rate = (
            len(constraints_met) / len(expected_constraints)
            if expected_constraints
            else 1.0
        )
        tool_hint_found = expected_tool.lower() in lower if expected_tool else True
        arg_hits = sum(1 for value in expected_args.values() if str(value).lower() in lower)
        arg_rate = arg_hits / len(expected_args) if expected_args else 1.0
        correct = constraint_rate >= 0.8 and (tool_hint_found or arg_rate >= 0.5)

        return EvalResult(
            query=query,
            expected={"tool": expected_tool, "args": expected_args, "constraints": list(expected_constraints)},
            predicted={
                "retrieved_count": len(recall_result.facts),
                "constraints_met": sorted(constraints_met),
            },
            correct=correct,
            latency_ms=latency_ms,
            details={
                "constraint_satisfaction_rate": constraint_rate,
                "tool_hint_found": tool_hint_found,
                "argument_accuracy": arg_rate,
            },
        )

