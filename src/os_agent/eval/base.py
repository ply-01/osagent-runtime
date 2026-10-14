"""Base adapter interface for benchmark evaluation."""

from __future__ import annotations

import math
import statistics
from abc import ABC, abstractmethod
from typing import Any

from os_agent.eval import BenchmarkReport, EvalResult
from os_agent.memory.mem0_core import CoreMemoryStore
from os_agent.memory.mem0_pipeline import ExtractUpdatePipeline
from os_agent.memory.mem0_recall import RecallStrategy


class BenchmarkAdapter(ABC):
    """Base class for benchmark adapters."""

    benchmark_name: str = "base"

    def __init__(
        self,
        core: CoreMemoryStore,
        recall: RecallStrategy,
        pipeline: ExtractUpdatePipeline,
    ) -> None:
        self.core = core
        self.recall = recall
        self.pipeline = pipeline

    @abstractmethod
    def ingest_history(self, history: list[dict[str, Any]]) -> int:
        """Load historical data into the memory stack."""

    @abstractmethod
    def evaluate_single(self, test_case: dict[str, Any]) -> EvalResult:
        """Evaluate a single test case."""

    def evaluate(self, test_data: list[dict[str, Any]]) -> BenchmarkReport:
        """Run all test cases and aggregate the report."""
        results = [self.evaluate_single(case) for case in test_data]
        latencies = [result.latency_ms for result in results]
        correct = sum(1 for result in results if result.correct)
        return BenchmarkReport(
            benchmark_name=self.benchmark_name,
            total=len(results),
            correct=correct,
            accuracy=correct / len(results) if results else 0.0,
            p50_latency_ms=statistics.median(latencies) if latencies else 0.0,
            p95_latency_ms=_percentile(latencies, 0.95),
            results=results,
        )


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(min(math.ceil(len(ordered) * fraction) - 1, len(ordered) - 1), 0)
    return ordered[index]

