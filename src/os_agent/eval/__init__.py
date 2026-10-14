"""Benchmark evaluation interfaces for OS Agent memory."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class EvalResult:
    """Single benchmark case result."""

    query: str
    expected: dict[str, Any]
    predicted: dict[str, Any]
    correct: bool
    latency_ms: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BenchmarkReport:
    """Aggregated benchmark report."""

    benchmark_name: str
    total: int
    correct: int
    accuracy: float
    p50_latency_ms: float
    p95_latency_ms: float
    metrics: dict[str, float] = field(default_factory=dict)
    results: list[EvalResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["results"] = [result.to_dict() for result in self.results]
        return payload

