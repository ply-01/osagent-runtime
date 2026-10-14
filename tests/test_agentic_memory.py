"""Integration tests for agentic memory dispatch, recall, and traces."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from os_agent.memory.agentic_policy import MemoryDistiller
from os_agent.memory.interface import (
    ForgetRequest,
    MemoryActionRequest,
    RecallRequest,
    RememberRequest,
    UpdateRequest,
)
from os_agent.memory.mem0_schemas import FactCard, RecallResult
from os_agent.memory.memory_config import MemoryConfig
from os_agent.memory.module import AgentMemoryModule


class _RecordingRecallStrategy:
    max_total = 8

    def __init__(self, facts: list[FactCard]) -> None:
        self.facts = facts
        self.requested_limit: int | None = None

    def recall(
        self,
        query: str,
        entity_filter: str | None = None,
        *,
        max_total: int | None = None,
        **_: object,
    ) -> RecallResult:
        self.requested_limit = max_total
        limit = len(self.facts) if max_total is None else max(max_total, 0)
        return RecallResult(
            facts=self.facts[:limit],
            source_layer="core",
            query=query,
            total_core_hits=min(len(self.facts), limit),
        )


class _RecordingDistiller(MemoryDistiller):
    def __init__(self) -> None:
        self.candidate_count = 0

    def distill(self, records, query: str, limit: int):  # type: ignore[no-untyped-def]
        self.candidate_count = len(records)
        return list(reversed(records))[:limit]


class AgenticMemoryTests(unittest.TestCase):
    def test_recall_expands_candidates_before_distillation(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = MemoryConfig(root / "memory_config.json")
            facts = [
                FactCard(
                    entity="project",
                    attribute=f"note_{index}",
                    value=f"memory candidate {index}",
                    evidence="test",
                    source_session_id="session",
                    source_event_id=f"event-{index}",
                )
                for index in range(5)
            ]
            distiller = _RecordingDistiller()
            strategy = _RecordingRecallStrategy(facts)
            module = AgentMemoryModule(root / "memory", config=config, distiller=distiller)
            module.recall_strategy = strategy  # type: ignore[assignment]

            result = module.recall(RecallRequest(query="candidate", scope="ltm", limit=2))

            self.assertEqual(config.agentic_policy_distillation_candidate_limit, 24)
            self.assertEqual(strategy.requested_limit, 24)
            self.assertEqual(distiller.candidate_count, 5)
            self.assertEqual(len(result.records), 2)
            self.assertEqual(result.records[0].attribute, "note_4")

    def test_zero_limit_skips_candidate_recall(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            strategy = _RecordingRecallStrategy([])
            module = AgentMemoryModule(root / "memory")
            module.recall_strategy = strategy  # type: ignore[assignment]

            result = module.recall(RecallRequest(query="anything", scope="ltm", limit=0))

            self.assertFalse(result.ok)
            self.assertEqual(result.records, [])
            self.assertIsNone(strategy.requested_limit)

    def test_legacy_methods_share_dispatch_and_explicit_trace(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = MemoryConfig(root / "memory_config.json")
            config.set_value("agentic_policy_trace_enabled", True)
            trace_file = root / "memory" / "policy-traces.jsonl"
            module = AgentMemoryModule(root / "memory", config=config)

            module.remember(
                RememberRequest(
                    content="Use concise answers.",
                    entity="user",
                    attribute="reply_style",
                    value="concise",
                )
            )
            module.update(
                UpdateRequest(
                    content="Use concise answers with examples.",
                    entity="user",
                    attribute="reply_style",
                    value="concise with examples",
                )
            )
            module.recall(RecallRequest(query="concise examples", scope="ltm", limit=1))
            module.forget(
                ForgetRequest(scope="ltm", entity="user", attribute="reply_style")
            )

            traces = [json.loads(line) for line in trace_file.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(
                [trace["action"] for trace in traces],
                ["ltm.add", "ltm.update", "ltm.retrieve", "ltm.delete"],
            )
            self.assertTrue(all(trace["arguments"]["scope"] == "ltm" for trace in traces))

    def test_trace_is_disabled_by_default(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            module = AgentMemoryModule(root / "memory", config=MemoryConfig(root / "config.json"))

            module.execute_action(MemoryActionRequest(action="noop", reason="nothing to store"))

            self.assertIsNone(module.trace_store)
            self.assertFalse((root / "memory" / "policy-traces.jsonl").exists())

    def test_stm_policy_operations_run_through_dispatcher(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = MemoryConfig(root / "config.json")
            config.set_value("agentic_policy_trace_enabled", True)
            module = AgentMemoryModule(root / "memory", config=config)
            for memory_id, content in (("one", "first task note"), ("two", "second task note")):
                module.execute_action(
                    MemoryActionRequest(
                        action="remember",
                        scope="stm",
                        memory_id=memory_id,
                        content=content,
                    )
                )

            summary = module.execute_action(
                MemoryActionRequest(
                    action="stm.summary",
                    memory_ids=["one", "two"],
                    replace_sources=True,
                )
            )
            summary_id = summary.records[0].memory_id
            filtered = module.execute_action(
                MemoryActionRequest(
                    action="stm.filter",
                    memory_ids=[summary_id],
                    remove_irrelevant=True,
                )
            )
            recalled = module.execute_action(
                MemoryActionRequest(action="stm.retrieve", query="task note", limit=1)
            )

            self.assertTrue(summary.ok)
            self.assertEqual(summary.records[0].metadata["source_memory_ids"], ["one", "two"])
            self.assertEqual([record.memory_id for record in filtered.records], [summary_id])
            self.assertEqual([record.memory_id for record in recalled.records], [summary_id])
            traces = [
                json.loads(line)
                for line in (root / "memory" / "policy-traces.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertTrue(all(trace["arguments"]["scope"] == "stm" for trace in traces))

    def test_mutations_reject_all_scope(self) -> None:
        with TemporaryDirectory() as tmp:
            module = AgentMemoryModule(Path(tmp) / "memory")

            with self.assertRaisesRegex(ValueError, "requires scope"):
                module.execute_action(
                    MemoryActionRequest(action="remember", scope="all", content="invalid")
                )


if __name__ == "__main__":
    unittest.main()
