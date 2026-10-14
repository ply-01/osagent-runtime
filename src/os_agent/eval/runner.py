"""Unified benchmark runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from os_agent.eval import BenchmarkReport
from os_agent.eval.base import BenchmarkAdapter
from os_agent.eval.longmemeval import LongMemEvalV2Adapter
from os_agent.eval.mem2act import Mem2ActAdapter
from os_agent.eval.memora import MemoraAdapter
from os_agent.eval.memory_agent_bench import MemoryAgentBenchAdapter
from os_agent.eval.perma import PERMAAdapter
from os_agent.memory.mem0_pipeline import ExtractUpdatePipeline
from os_agent.memory.module import AgentMemoryModule

ADAPTERS: dict[str, type[BenchmarkAdapter]] = {
    "mem2act": Mem2ActAdapter,
    "perma": PERMAAdapter,
    "memory_agent_bench": MemoryAgentBenchAdapter,
    "memora": MemoraAdapter,
    "longmemeval_v2": LongMemEvalV2Adapter,
}


def run_benchmark(
    benchmark_name: str,
    test_data_path: str | Path,
    memory_root: str = "memory",
) -> BenchmarkReport:
    """Run one benchmark end-to-end."""
    path = Path(test_data_path)
    if not path.exists():
        raise FileNotFoundError(f"Test data not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    history = _as_list(data.get("history", []))
    test_cases = _as_list(data.get("test_cases", []))

    module = AgentMemoryModule(memory_root)
    pipeline = ExtractUpdatePipeline(module.core, module.archival, module.markdown_store, module.config)

    adapter_cls = ADAPTERS.get(benchmark_name)
    if adapter_cls is None:
        raise ValueError(f"Unknown benchmark: {benchmark_name}. Available: {sorted(ADAPTERS)}")

    adapter = adapter_cls(core=module.core, recall=module.recall_strategy, pipeline=pipeline)
    facts_created = adapter.ingest_history(history)
    module.refresh_semantic_indexes()
    report = adapter.evaluate(test_cases)
    report.metrics["facts_created"] = float(facts_created)
    report.metrics["history_items"] = float(len(history))
    report.metrics["test_cases"] = float(len(test_cases))
    return report


def _as_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []

