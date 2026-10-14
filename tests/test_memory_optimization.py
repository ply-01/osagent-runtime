"""Generalization tests for preference, retrieval, conflict, and deletion."""

from __future__ import annotations

import gzip
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from os_agent.memory import embedding_provider
from os_agent.memory.agentic_policy import MemoryDistiller
from os_agent.memory.embedding_cache import SqliteEmbeddingCache
from os_agent.memory.embedding_provider import (
    EmbeddingConfig,
    LocalOnnxEmbeddingProvider,
    LocalTransformerEmbeddingProvider,
    create_embedding_provider,
)
from os_agent.memory.fact_vector_index import FactSemanticHit, FactSemanticIndex
from os_agent.memory.interface import (
    ForgetRequest,
    MemoryRecord,
    RecallRequest,
    RememberRequest,
    UpdateRequest,
)
from os_agent.memory.mem0_archival import ArchivalMemoryStore
from os_agent.memory.mem0_core import CoreMemoryStore
from os_agent.memory.mem0_pipeline import ExtractUpdatePipeline
from os_agent.memory.mem0_recall import RecallStrategy
from os_agent.memory.mem0_schemas import FactCard, SessionDocument
from os_agent.memory.memory_config import MemoryConfig
from os_agent.memory.module import AgentMemoryModule
from os_agent.memory.preference_extractor import PreferenceExtractor
from os_agent.memory.retrieval_text import tokenize_search_text
from os_agent.memory.session_vector_index import SessionSemanticIndex


def _event(text: str, event_id: str) -> dict:
    return {
        "event_type": "message_start",
        "session_id": event_id,
        "task_id": f"task-{event_id}",
        "event_id": event_id,
        "payload": {"role": "user", "content": text},
    }


def _tool_event(event_id: str, inputs: dict) -> dict:
    return {
        "event_type": "tool_result",
        "session_id": event_id,
        "task_id": f"task-{event_id}",
        "event_id": event_id,
        "payload": {
            "tool_name": "document.export",
            "status": "success",
            "input": inputs,
            "output": {"ok": True},
        },
    }


class _SemanticStub:
    enabled = True

    def __init__(self, target) -> None:  # type: ignore[no-untyped-def]
        self.target = target

    def search(self, query: str, facts: list, limit: int):  # type: ignore[no-untyped-def]
        return [FactSemanticHit(self.target, 0.93)][:limit]


class _DirectionalEmbeddingProvider:
    enabled = True
    cache_namespace = "directional-test"

    def __init__(self) -> None:
        self.query_items = 0
        self.document_items = 0

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        self.query_items += len(texts)
        return [[1.0, 0.0] for _ in texts]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_items += len(texts)
        return [
            [1.0, 0.0] if "安静" in text or "quiet" in text.lower() else [0.0, 1.0]
            for text in texts
        ]


class _CountingCache:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], list[float]] = {}
        self.transient: dict[tuple[str, str], list[float]] = {}
        self.document_reads = 0

    def get_many(self, namespace: str, texts: list[str]):  # type: ignore[no-untyped-def]
        self.document_reads += 1
        return [self.values.get((namespace, text)) for text in texts]

    def set_many(self, namespace: str, rows: dict[str, list[float]]) -> None:
        self.values.update({(namespace, text): vector for text, vector in rows.items()})

    def get_transient(self, namespace: str, text: str) -> list[float] | None:
        return self.transient.get((namespace, text))

    def set_many_transient(
        self, namespace: str, rows: dict[str, list[float]]
    ) -> None:
        self.transient.update(
            {(namespace, text): vector for text, vector in rows.items()}
        )

    def clear(self) -> None:
        self.values.clear()
        self.transient.clear()


class _CapturingLocalProvider(LocalTransformerEmbeddingProvider):
    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__(config)
        self.batches: list[list[str]] = []

    def _encode(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(texts)
        return [[1.0] for _ in texts]


class MemoryOptimizationTests(unittest.TestCase):
    def test_distillation_preserves_cross_session_evidence_diversity(self) -> None:
        records = [
            MemoryRecord(
                memory_id=f"same-{index}",
                scope="ltm",
                content=f"quiet hotel near transit evidence {index}",
                tags=["preference"],
                metadata={
                    "source_session_id": "same-session",
                    "semantic_score": 0.9 - index * 0.01,
                    "search_score": 0.02,
                },
            )
            for index in range(3)
        ]
        records.append(
            MemoryRecord(
                memory_id="other-session",
                scope="ltm",
                content="quiet hotel near transit corroborating evidence",
                tags=["semantic_session"],
                metadata={
                    "source_session_id": "other-session",
                    "semantic_score": 0.82,
                    "search_score": 0.02,
                },
            )
        )

        selected = MemoryDistiller().distill(records, "quiet hotel near transit", 3)

        self.assertIn("other-session", [record.memory_id for record in selected])
        self.assertEqual(
            sum(
                record.metadata["source_session_id"] == "same-session"
                for record in selected
            ),
            2,
        )

    def test_semantic_indexes_reuse_resident_document_snapshots(self) -> None:
        provider = _DirectionalEmbeddingProvider()
        fact_cache = _CountingCache()
        fact_index = FactSemanticIndex(
            provider=provider, cache=fact_cache  # type: ignore[arg-type]
        )
        fact = FactCard(
            entity="user",
            attribute="hotel",
            value="quiet room",
            evidence="quiet room",
            source_session_id="hotel-session",
            source_event_id="hotel-event",
        )
        fact_index.refresh([fact])
        reads_after_refresh = fact_cache.document_reads

        fact_hits = fact_index.search("quiet hotel", [fact], 1)

        self.assertEqual(fact_cache.document_reads, reads_after_refresh)
        self.assertEqual(fact_hits[0].fact, fact)

        session_cache = _CountingCache()
        session_index = SessionSemanticIndex(
            provider=provider, cache=session_cache  # type: ignore[arg-type]
        )
        document = SessionDocument(
            document_id="session-document",
            session_id="hotel-session",
            task_id="hotel",
            text="A complete conversation about a quiet room.",
            embedding_text="quiet room",
        )
        session_index.refresh([document])
        reads_after_refresh = session_cache.document_reads

        session_hits = session_index.search("quiet hotel", [document], 1)

        self.assertEqual(session_cache.document_reads, reads_after_refresh)
        self.assertEqual(session_hits[0].document, document)

    def test_repeated_implicit_preferences_are_reinforced_across_sessions(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = MemoryConfig(root / "config.json")
            core = CoreMemoryStore(root / "memory", config=config)
            pipeline = ExtractUpdatePipeline(
                core,
                ArchivalMemoryStore(root / "memory", config=config),
                config=config,
            )

            first = pipeline.process_session(
                [_event("I'm interested in quiet boutique hotels near transit.", "one")],
                session_id="session-one",
            )
            second = pipeline.process_session(
                [_event("I'm curious about quiet boutique hotels near transit.", "two")],
                session_id="session-two",
            )

            self.assertFalse(any("preference_reinforced" in fact.tags for fact in first))
            reinforced = [
                fact for fact in second if "preference_reinforced" in fact.tags
            ]
            self.assertEqual(len(reinforced), 1)
            self.assertGreaterEqual(reinforced[0].confidence, 0.72)
            self.assertEqual(
                len(
                    {
                        tag
                        for tag in reinforced[0].tags
                        if tag.startswith("evidence_session:")
                    }
                ),
                2,
            )

    def test_conflicting_observation_polarity_is_not_reinforced(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = MemoryConfig(root / "config.json")
            pipeline = ExtractUpdatePipeline(
                CoreMemoryStore(root / "memory", config=config),
                ArchivalMemoryStore(root / "memory", config=config),
                config=config,
            )

            pipeline.process_session(
                [_event("I'm interested in quiet hotels near transit.", "positive")],
                session_id="positive-session",
            )
            result = pipeline.process_session(
                [
                    _event(
                        "I'm interested in avoiding quiet hotels near transit.",
                        "negative",
                    )
                ],
                session_id="negative-session",
            )

            self.assertFalse(
                any("preference_reinforced" in fact.tags for fact in result)
            )

    def test_rrf_tie_prefers_session_summary_over_raw_event(self) -> None:
        core = FactCard(
            entity="user",
            attribute="preference",
            value="quiet hotels",
            evidence="quiet hotels",
            source_session_id="core-session",
            source_event_id="core-event",
        )
        session = FactCard(
            entity="session",
            attribute="trip",
            value="full session",
            evidence="full session",
            source_session_id="trip-session",
            source_event_id="session:trip-session",
            tier="archival",
            tags=["semantic_session"],
        )
        raw = FactCard(
            entity="event",
            attribute="message",
            value="one event",
            evidence="one event",
            source_session_id="raw-session",
            source_event_id="raw-event",
            tier="archival",
            tags=["raw_event"],
        )

        merged = RecallStrategy._rrf_merge_rankings([[core], [session], [raw]], 3)

        self.assertEqual(merged, [core, session, raw])

    def test_sqlite_embedding_cache_persists_documents_but_not_queries(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "vectors.sqlite3"
            cache = SqliteEmbeddingCache(path)
            cache.set_many("documents", {"stored": [1.0, 2.0]})
            cache.set_many_transient("queries", {"ephemeral": [3.0, 4.0]})
            cache.close()

            reopened = SqliteEmbeddingCache(path)

            self.assertEqual(reopened.get("documents", "stored"), [1.0, 2.0])
            self.assertIsNone(reopened.get_transient("queries", "ephemeral"))
            reopened.clear()
            self.assertFalse(path.exists())

    def test_local_e5_provider_uses_asymmetric_retrieval_prefixes(self) -> None:
        provider = _CapturingLocalProvider(
            EmbeddingConfig(
                backend="local_transformers",
                model="intfloat/multilingual-e5-small",
                revision="fixed-revision",
            )
        )

        provider.embed_queries(["安静的酒店"])
        provider.embed_documents(["The user prefers quiet rooms."])

        self.assertEqual(provider.batches[0], ["query: 安静的酒店"])
        self.assertEqual(
            provider.batches[1], ["passage: The user prefers quiet rooms."]
        )
        self.assertIn("fixed-revision", provider.cache_namespace)

    def test_embedding_factory_selects_quantized_onnx_backend(self) -> None:
        provider = create_embedding_provider(
            EmbeddingConfig(
                backend="local_onnx",
                model="intfloat/multilingual-e5-small",
            )
        )

        self.assertIsInstance(provider, LocalOnnxEmbeddingProvider)

    def test_dedicated_embedding_config_is_discovered_without_env_var(self) -> None:
        env_keys = (
            "OS_AGENT_EMBEDDING_CONFIG",
            "OS_AGENT_LLM_CONFIG",
            "OS_AGENT_EMBEDDING_BACKEND",
            "OS_AGENT_EMBEDDING_MODEL",
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config").mkdir()
            (root / "config" / "embedding.local.json").write_text(
                json.dumps(
                    {
                        "embedding_backend": "local_onnx",
                        "embedding_model": "intfloat/multilingual-e5-base",
                    }
                ),
                encoding="utf-8",
            )
            previous_cwd = Path.cwd()
            previous_env = {key: os.environ.pop(key, None) for key in env_keys}
            try:
                os.chdir(root)
                config = EmbeddingConfig.from_env()
            finally:
                os.chdir(previous_cwd)
                for key, value in previous_env.items():
                    if value is not None:
                        os.environ[key] = value

        self.assertEqual(config.backend, "local_onnx")
        self.assertEqual(config.model, "intfloat/multilingual-e5-base")
        self.assertTrue(create_embedding_provider(config).enabled)

    def test_explicit_embedding_config_env_var_wins_over_dedicated_file(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config").mkdir()
            (root / "config" / "embedding.local.json").write_text(
                json.dumps({"embedding_model": "from-dedicated-file"}),
                encoding="utf-8",
            )
            explicit = root / "explicit.json"
            explicit.write_text(
                json.dumps({"embedding_model": "from-explicit-env"}),
                encoding="utf-8",
            )
            previous_cwd = Path.cwd()
            previous = os.environ.get("OS_AGENT_EMBEDDING_CONFIG")
            try:
                os.chdir(root)
                os.environ["OS_AGENT_EMBEDDING_CONFIG"] = str(explicit)
                config = EmbeddingConfig.from_env()
            finally:
                os.chdir(previous_cwd)
                if previous is None:
                    os.environ.pop("OS_AGENT_EMBEDDING_CONFIG", None)
                else:
                    os.environ["OS_AGENT_EMBEDDING_CONFIG"] = previous

        self.assertEqual(config.model, "from-explicit-env")

    def test_llm_config_still_supplies_embedding_model_as_fallback(self) -> None:
        env_keys = ("OS_AGENT_EMBEDDING_CONFIG", "OS_AGENT_EMBEDDING_MODEL")
        with TemporaryDirectory() as tmp, mock.patch.object(
            embedding_provider,
            "DEFAULT_EMBEDDING_CONFIG_PATH",
            "config/absent-embedding-config.json",
        ):
            root = Path(tmp)
            llm_config = root / "llm.local.json"
            llm_config.write_text(
                json.dumps(
                    {
                        "embedding_model": "text-embedding-3-small",
                        "embedding_api_key": "test-key",
                    }
                ),
                encoding="utf-8",
            )
            previous_cwd = Path.cwd()
            previous_env = {key: os.environ.pop(key, None) for key in env_keys}
            previous_llm = os.environ.get("OS_AGENT_LLM_CONFIG")
            try:
                os.chdir(root)
                os.environ["OS_AGENT_LLM_CONFIG"] = str(llm_config)
                config = EmbeddingConfig.from_env()
            finally:
                os.chdir(previous_cwd)
                for key, value in previous_env.items():
                    if value is not None:
                        os.environ[key] = value
                if previous_llm is None:
                    os.environ.pop("OS_AGENT_LLM_CONFIG", None)
                else:
                    os.environ["OS_AGENT_LLM_CONFIG"] = previous_llm

        self.assertEqual(config.model, "text-embedding-3-small")
        self.assertTrue(create_embedding_provider(config).enabled)

    def test_session_summary_ignores_evaluation_only_metadata(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            archival = ArchivalMemoryStore(
                root / "memory", config=MemoryConfig(root / "config.json")
            )
            event = _event("I prefer quiet rooms near transit.", "session-a")
            event.update(
                {
                    "gold_label": "DO_NOT_INDEX",
                    "options": "DO_NOT_INDEX_OPTIONS",
                    "preferences": "DO_NOT_INDEX_PREFERENCES",
                }
            )
            archival.append_raw_events([event])

            documents = archival.session_documents()

            self.assertEqual(len(documents), 1)
            self.assertIn("quiet rooms", documents[0].text)
            self.assertNotIn("DO_NOT_INDEX", documents[0].text)

    def test_session_summary_prioritizes_later_preference_evidence(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            archival = ArchivalMemoryStore(
                root / "memory", config=MemoryConfig(root / "config.json")
            )
            events = [
                _event("Could you explain the available travel options?", "turn-a"),
                _event(
                    "I prefer non-stop morning flights with a window seat.",
                    "turn-b",
                ),
            ]
            for event in events:
                event["session_id"] = "travel-session"
            archival.append_raw_events(events)

            document = archival.session_documents(
                max_chars=900, embedding_max_chars=240
            )[0]

            self.assertLess(
                document.embedding_text.index("I prefer"),
                document.embedding_text.index("explain"),
            )

    def test_raw_events_compact_to_cold_archive_and_build_global_summary(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = MemoryConfig(root / "memory-config.json")
            config.set_value("archival_raw_event_hot_max_events", 2)
            config.set_value("archival_global_summary_max_chars", 500)
            config.set_value("archival_global_summary_max_sessions", 10)
            archival = ArchivalMemoryStore(root / "memory", config=config)
            events = [
                _event("alpha historical decision", "session-a"),
                _event("beta historical decision", "session-b"),
                {
                    **_event("gamma current decision", "event-c-1"),
                    "session_id": "session-c",
                },
                {
                    **_event("delta current decision", "event-c-2"),
                    "session_id": "session-c",
                },
            ]

            self.assertEqual(archival.append_raw_events(events), 4)
            hot = (root / "memory" / "raw_events.jsonl").read_text(encoding="utf-8")
            self.assertEqual(len(hot.splitlines()), 2)
            cold_path = root / "memory" / "raw_events.archive.jsonl.gz"
            self.assertTrue(cold_path.exists())
            with gzip.open(cold_path, "rt", encoding="utf-8") as handle:
                cold_text = handle.read()
            self.assertIn("alpha historical decision", cold_text)

            documents = archival.session_documents()
            self.assertIn("__global__", {document.session_id for document in documents})
            global_document = next(
                document
                for document in documents
                if document.session_id == "__global__"
            )
            self.assertIn("session-a", global_document.text)
            self.assertIn("session-b", global_document.text)
            self.assertIn("alpha historical decision", global_document.text)

            matches = archival.search_raw_events("alpha historical decision")
            self.assertTrue(matches)

    def test_raw_event_search_does_not_scan_ambient_session_directories(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_root = root / "memory"
            ambient = root / "unrelated-sessions"
            ambient.mkdir(parents=True)
            (ambient / "foreign.jsonl").write_text(
                json.dumps(_event("foreign private session", "foreign")) + "\n",
                encoding="utf-8",
            )
            archival = ArchivalMemoryStore(memory_root)

            self.assertEqual(archival.search_raw_events("foreign private session"), [])
            self.assertTrue(
                archival.search_raw_events(
                    "foreign private session",
                    session_dir=ambient,
                )
            )

    def test_core_and_session_semantics_share_query_cache(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = MemoryConfig(root / "config.json")
            core = CoreMemoryStore(root / "memory", config=config)
            archival = ArchivalMemoryStore(root / "memory", config=config)
            core.set("project", "note", "unrelated build output")
            archival.append_raw_events(
                [_event("我喜欢安静并且靠近公共交通的房间。", "lodging-session")]
            )
            provider = _DirectionalEmbeddingProvider()
            module = AgentMemoryModule(
                root / "memory",
                core=core,
                archival=archival,
                config=config,
                embedding_provider=provider,  # type: ignore[arg-type]
            )
            indexed = module.refresh_semantic_indexes()
            document_items_after_refresh = provider.document_items

            result = module.recall(
                RecallRequest(
                    query="Which lodging suits a calm stay near a station?",
                    scope="ltm",
                    limit=2,
                )
            )

            self.assertEqual(indexed, {"core_facts": 1, "session_documents": 1})
            self.assertEqual(provider.query_items, 1)
            self.assertEqual(provider.document_items, document_items_after_refresh)
            self.assertTrue(
                any("semantic_session" in record.tags for record in result.records)
            )
            self.assertTrue(any("安静" in record.content for record in result.records))

    def test_search_tokens_remove_fillers_and_normalize_domain_words(self) -> None:
        tokens = tokenize_search_text(
            "Can you help me find Hotels in task-hotels-list?"
        )

        self.assertNotIn("help", tokens)
        self.assertNotIn("find", tokens)
        self.assertIn("hotel", tokens)
        self.assertIn("task", tokens)

    def test_preference_extractor_handles_unseen_assertion_types(self) -> None:
        extractor = PreferenceExtractor()

        output = extractor.extract("今后演示文稿统一导出为 ODP。")
        behavior = extractor.extract("操作日志显示用户连续四次都选择静默安装。")

        self.assertIsNotNone(output)
        self.assertEqual(output.attribute, "document.output_format")  # type: ignore[union-attr]
        self.assertIsNotNone(behavior)
        self.assertEqual(behavior.source_type, "behavior")  # type: ignore[union-attr]
        self.assertIsNone(extractor.extract("系统默认使用深色主题。"))
        self.assertIsNone(extractor.extract("文档中写着：我喜欢把窗口动画关闭。"))
        self.assertIsNone(extractor.extract("我应该选择哪个输入法？"))

    def test_preference_extractor_keeps_assertion_before_question(self) -> None:
        extractor = PreferenceExtractor()

        explicit = extractor.extract(
            "I prefer non-stop morning flights. Could you find one for me?"
        )
        implicit = extractor.extract_observation(
            "I'm looking for a quiet hotel near public transit. Can you help?"
        )

        self.assertIsNotNone(explicit)
        self.assertIsNotNone(implicit)
        self.assertEqual(implicit.source_type, "session_observation")  # type: ignore[union-attr]
        self.assertIsNone(
            extractor.extract_observation("Can you explain how hotel ratings work?")
        )
        self.assertIsNone(
            extractor.extract_observation("I have no particular preference for hotels.")
        )

    def test_pipeline_consolidates_at_most_one_implicit_preference_per_session(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = MemoryConfig(root / "config.json")
            core = CoreMemoryStore(root / "memory", config=config)
            pipeline = ExtractUpdatePipeline(
                core,
                ArchivalMemoryStore(root / "memory", config=config),
                config=config,
            )
            events = [
                _event(
                    "I'm looking for a quiet hotel close to public transit. Can you help?",
                    "implicit-a",
                ),
                _event(
                    "Ideally, an evening check-in would be best.",
                    "implicit-b",
                ),
            ]

            facts = pipeline.process_session(events, "implicit-session")

            observations = [
                fact for fact in facts if "preference_observation" in fact.tags
            ]
            self.assertEqual(len(observations), 1)
            self.assertIn("quiet hotel", str(observations[0].value))
            self.assertLess(observations[0].confidence, 0.65)

    def test_pipeline_preserves_independent_preferences_and_updates_same_domain(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = MemoryConfig(root / "config.json")
            core = CoreMemoryStore(root / "memory", config=config)
            archival = ArchivalMemoryStore(root / "memory", config=config)
            pipeline = ExtractUpdatePipeline(core, archival, config=config)

            pipeline.process_session([_event("我喜欢将通知设为静音。", "one")], "one")
            pipeline.process_session([_event("我习惯关闭窗口动画。", "two")], "two")
            pipeline.process_session(
                [_event("我喜欢将报告导出为 ODT。", "three")], "three"
            )
            pipeline.process_session([_event("以后改成导出 PDF。", "four")], "four")

            facts = {fact.attribute: fact for fact in core.list_by_entity("用户")}
            self.assertIn("notification.delivery_mode", facts)
            self.assertIn("ui.window_animation", facts)
            self.assertEqual(
                facts["document.output_format"].value, "以后改成导出 PDF。"
            )
            history = archival.get_fact_history("用户", "document.output_format")
            self.assertEqual(
                [fact.value for fact in history], ["我喜欢将报告导出为 ODT。"]
            )

    def test_chinese_recall_generalizes_to_unseen_entity_and_paraphrase(self) -> None:
        with TemporaryDirectory() as tmp:
            module = AgentMemoryModule(Path(tmp) / "memory")
            module.remember(
                RememberRequest(
                    content="北极星项目的网络代理地址是 gateway.internal",
                    entity="北极星项目",
                    attribute="网络代理地址",
                    value="gateway.internal",
                )
            )
            for index in range(30):
                module.remember(
                    RememberRequest(
                        content=f"无关记录 {index}",
                        entity=f"其他项目{index}",
                        attribute="构建编号",
                        value=f"build-{index}",
                    )
                )

            result = module.recall(
                RecallRequest(query="北极星工程通过哪个代理联网", scope="ltm", limit=5)
            )

            self.assertTrue(result.ok)
            self.assertTrue(
                any(record.value == "gateway.internal" for record in result.records)
            )

    def test_repeated_tool_parameters_become_preferences_without_sensitive_values(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = MemoryConfig(root / "config.json")
            core = CoreMemoryStore(root / "memory", config=config)
            pipeline = ExtractUpdatePipeline(
                core,
                ArchivalMemoryStore(root / "memory", config=config),
                config=config,
            )
            for index in range(3):
                pipeline.process_session(
                    [
                        _tool_event(
                            f"tool-{index}",
                            {"quality": "print", "api_token": "must-not-persist"},
                        )
                    ],
                    f"session-{index}",
                )

            facts = core.list_by_entity("用户")
            self.assertEqual(len(facts), 1)
            self.assertEqual(
                facts[0].attribute,
                "tool.parameter.document_export.quality",
            )
            self.assertEqual(facts[0].value, "print")
            evidence = (root / "memory" / "behavior_evidence.json").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("must-not-persist", evidence)

    def test_semantic_candidates_are_merged_when_provider_is_available(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = MemoryConfig(root / "config.json")
            core = CoreMemoryStore(root / "memory", config=config)
            archival = ArchivalMemoryStore(root / "memory", config=config)
            target = core.set("user", "locale", "zh_CN", evidence="preferred locale")
            strategy = RecallStrategy(
                core,
                archival,
                config=config,
                semantic_index=_SemanticStub(target),  # type: ignore[arg-type]
            )

            result = strategy.recall("completely unrelated wording", max_total=5)

            self.assertEqual([fact.fact_id for fact in result.facts], [target.fact_id])

    def test_implicit_raw_event_competes_with_weak_core_hits(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = MemoryConfig(root / "config.json")
            core = CoreMemoryStore(root / "memory", config=config)
            archival = ArchivalMemoryStore(root / "memory", config=config)
            pipeline = ExtractUpdatePipeline(core, archival, config=config)
            event = _event("A quiet room near transit works best.", "implicit")
            event["task_id"] = "lodging-hotels-reservation"
            pipeline.process_session([event], "implicit-session")
            core.set(
                "unrelated-trip",
                "note",
                "A trip checklist",
                evidence="weak lexical distractor",
            )
            module = AgentMemoryModule(
                root / "memory",
                core=core,
                archival=archival,
                config=config,
            )

            result = module.recall(
                RecallRequest(
                    query="help me find a hotel for my trip", scope="ltm", limit=3
                )
            )

            self.assertTrue(
                any(
                    "quiet room near transit" in record.content.lower()
                    for record in result.records
                )
            )

    def test_hard_delete_removes_versions_sources_and_caches(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_root = root / "memory"
            module = AgentMemoryModule(
                memory_root, config=MemoryConfig(root / "config.json")
            )
            old_value = "private-token-old"
            new_value = "private-token-new"
            module.remember(
                RememberRequest(
                    content=old_value,
                    entity="account",
                    attribute="access_token",
                    value=old_value,
                    source_event_id="evt-old",
                )
            )
            module.update(
                UpdateRequest(
                    content=new_value,
                    entity="account",
                    attribute="access_token",
                    value=new_value,
                    source_event_id="evt-new",
                )
            )
            event_dir = root / "data" / "events"
            event_dir.mkdir(parents=True)
            event_file = event_dir / "events.jsonl"
            event_file.write_text(
                "\n".join(
                    json.dumps({"event_id": event_id, "payload": {"value": value}})
                    for event_id, value in (
                        ("evt-old", old_value),
                        ("evt-new", new_value),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            cache_dir = root / "data" / "vector-cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / "derived.json"
            cache_file.write_text('{"cached": true}', encoding="utf-8")
            module.archival.append_raw_events(
                [
                    {
                        "event_id": "evt-new",
                        "event_type": "message_start",
                        "payload": {"content": new_value},
                    }
                ]
            )

            deleted = module.forget(
                ForgetRequest(
                    scope="ltm",
                    entity="account",
                    attribute="access_token",
                    hard=True,
                )
            )

            self.assertTrue(deleted.ok)
            self.assertTrue(deleted.records[0].metadata["hard_deleted"])
            searchable_files = list(memory_root.glob("*.json")) + list(
                memory_root.glob("*.md")
            )
            residual = "\n".join(
                path.read_text(encoding="utf-8", errors="replace")
                for path in searchable_files
            )
            self.assertNotIn(old_value, residual)
            self.assertNotIn(new_value, residual)
            self.assertEqual(event_file.read_text(encoding="utf-8"), "")
            self.assertEqual(
                (memory_root / "raw_events.jsonl").read_text(encoding="utf-8"), ""
            )
            self.assertFalse(cache_file.exists())


if __name__ == "__main__":
    unittest.main()
