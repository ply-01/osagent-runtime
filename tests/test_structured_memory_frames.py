from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from os_agent.memory.fact_vector_index import FactSemanticHit
from os_agent.memory.interface import RecallRequest, RememberRequest, UpdateRequest
from os_agent.memory.mem0_archival import ArchivalMemoryStore
from os_agent.memory.mem0_core import CoreMemoryStore
from os_agent.memory.mem0_pipeline import ExtractUpdatePipeline
from os_agent.memory.memory_config import MemoryConfig
from os_agent.memory.memory_frame import MemoryFrameParser
from os_agent.memory.module import AgentMemoryModule


def event(text: str, event_id: str, **metadata: str) -> dict:
    return {
        "event_type": "message_start",
        "session_id": event_id,
        "task_id": f"task-{event_id}",
        "event_id": event_id,
        "payload": {"role": "user", "content": text, **metadata},
    }


def pipeline(root: Path) -> tuple[ExtractUpdatePipeline, CoreMemoryStore, ArchivalMemoryStore]:
    config = MemoryConfig(root / "config.json")
    core = CoreMemoryStore(root / "memory", config=config)
    archival = ArchivalMemoryStore(root / "memory", config=config)
    return ExtractUpdatePipeline(core, archival, config=config), core, archival


class CapturingSemanticIndex:
    enabled = True

    def __init__(self) -> None:
        self.candidates = []

    def search(self, query, facts, limit):  # type: ignore[no-untyped-def]
        self.candidates = list(facts)
        return [FactSemanticHit(fact, 0.9) for fact in facts[:limit]]

    def clear_cache(self) -> None:
        return None


def test_parser_models_certainty_temporality_scope_and_intent() -> None:
    parser = MemoryFrameParser()

    scoped = parser.parse("在曙光工程中，我希望文档导出为 PDF。")
    uncertain = parser.parse("也许以后使用高对比度主题，但还没有确认。")
    reported = parser.parse(
        "External source statement.",
        metadata={
            "certainty": "reported",
            "attribute": "application.preferred_tool",
        },
    )
    temporary = parser.parse("这次临时改成导出 DOCX，不要保存为长期设置。")
    avoid = parser.parse(
        "Persistent notification restriction.",
        metadata={
            "intent": "avoid",
            "attribute": "notification.delivery_mode",
            "value": "sound",
        },
    )

    assert scoped is not None
    assert scoped.scope == "曙光工程"
    assert scoped.storage_entity == "用户::曙光工程"
    assert scoped.attribute == "document.output_format"
    assert uncertain is not None and uncertain.intent == "uncertain"
    assert not uncertain.should_persist
    assert reported is not None and reported.intent == "reported"
    assert not reported.should_persist
    assert temporary is not None and temporary.temporal_status == "temporary"
    assert not temporary.should_persist
    assert avoid is not None and avoid.intent == "avoid"
    assert avoid.polarity == "avoid"


def test_attribute_normalization_does_not_confuse_keyboard_with_language() -> None:
    frame = MemoryFrameParser().parse("我希望以后输入法使用英文键盘。")

    assert frame is not None
    assert frame.attribute == "input.preferred_method"


def test_natural_pipeline_updates_same_attribute_and_preserves_scopes() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        memory_pipeline, core, archival = pipeline(root)

        memory_pipeline.process_session(
            [event("我希望文档导出为 ODT。", "initial")], "initial"
        )
        memory_pipeline.process_session(
            [
                event(
                    "更新文档设置。",
                    "update",
                    intent="update",
                    attribute="document.output_format",
                    value="PDF",
                )
            ],
            "update",
        )
        memory_pipeline.process_session(
            [event("在曙光工程中，我希望文档导出为 PDF。", "scope-a")],
            "scope-a",
        )
        memory_pipeline.process_session(
            [event("在远帆工程中，我希望文档导出为 DOCX。", "scope-b")],
            "scope-b",
        )

        global_fact = core.get("用户", "document.output_format")
        assert global_fact is not None and "PDF" in str(global_fact.value)
        assert core.get("用户::曙光工程", "document.output_format") is not None
        assert core.get("用户::远帆工程", "document.output_format") is not None
        history = archival.get_fact_history("用户", "document.output_format")
        assert len(history) == 1 and "ODT" in str(history[0].value)


def test_temporary_instruction_does_not_replace_stable_value_and_revert_is_stable() -> None:
    with TemporaryDirectory() as tmp:
        memory_pipeline, core, _ = pipeline(Path(tmp))
        memory_pipeline.process_session(
            [event("我希望文档导出为 PDF。", "stable")], "stable"
        )
        memory_pipeline.process_session(
            [event("本次临时改成导出 DOCX。", "temporary")], "temporary"
        )
        memory_pipeline.process_session(
            [event("最终恢复为文档导出 PDF。", "revert")], "revert"
        )

        fact = core.get("用户", "document.output_format")
        assert fact is not None
        assert "PDF" in str(fact.value)
        assert "DOCX" not in str(fact.value)
        assert fact.temporal_status == "reverted"
        assert fact.intent == "revert"


def test_recall_filters_semantic_candidates_by_known_entity_and_attribute() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        config = MemoryConfig(root / "config.json")
        core = CoreMemoryStore(root / "memory", config=config)
        archival = ArchivalMemoryStore(root / "memory", config=config)
        for index in range(40):
            core.set(
                f"workspace_{index:03d}",
                "network_profile",
                f"profile_{index:03d}",
            )
            core.set(f"workspace_{index:03d}", "theme", "dark")
        semantic = CapturingSemanticIndex()
        module = AgentMemoryModule(
            root / "memory",
            core=core,
            archival=archival,
            config=config,
            semantic_index=semantic,  # type: ignore[arg-type]
        )

        result = module.recall(
            RecallRequest(
                query="workspace_027 network_profile current value",
                scope="ltm",
                limit=5,
            )
        )

        assert semantic.candidates
        assert {fact.entity for fact in semantic.candidates} == {"workspace_027"}
        assert {fact.attribute for fact in semantic.candidates} == {"network_profile"}
        assert result.records[0].value == "profile_027"


def test_scoped_query_and_time_window_become_structured_constraints() -> None:
    parser = MemoryFrameParser()
    constraints = parser.query_constraints(
        "查找最近 30 天曙光工程的 document.output_format",
        known_entities=["用户::曙光工程", "用户::远帆工程"],
        known_attributes=["document.output_format"],
    )

    assert constraints.entity == "用户::曙光工程"
    assert constraints.attribute == "document.output_format"
    assert constraints.days == 30


def test_canonical_attribute_key_shares_one_version_chain() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        module = AgentMemoryModule(root / "memory")
        module.remember(
            RememberRequest(
                content="Initial document setting",
                entity="用户",
                attribute="document.output_format",
                value="ODT",
            )
        )
        module.remember(
            RememberRequest(
                content="Updated document setting",
                entity="用户",
                attribute="document.output_format",
                value="PDF",
            )
        )

        current = module.core.get("用户", "document.output_format")
        history = module.archival.get_fact_history(
            "用户", "document.output_format"
        )
        assert current is not None and current.value == "PDF"
        assert current.version == 2
        assert len(history) == 1 and history[0].value == "ODT"


def test_recall_returns_only_current_version_for_structured_updates() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        module = AgentMemoryModule(root / "memory")
        module.remember(
            RememberRequest(
                content="Initial document setting",
                entity="用户",
                attribute="document.output_format",
                value="ODT",
            )
        )
        module.update(
            UpdateRequest(
                content="Updated document setting",
                entity="用户",
                attribute="document.output_format",
                value="PDF",
            )
        )

        result = module.recall(
            RecallRequest(
                query="用户 document.output_format",
                scope="ltm",
                limit=5,
            )
        )

        values = [record.value for record in result.records if record.entity == "用户"]
        assert values == ["PDF"]


def test_natural_language_assignment_updates_same_ltm_fact() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        module = AgentMemoryModule(root / "memory")

        first = module.remember(
            RememberRequest(content="我希望文档导出为 Markdown。")
        )
        second = module.update(
            UpdateRequest(content="把文档导出格式改成 PDF。")
        )

        current = module.core.get("用户", "document.output_format")
        history = module.archival.get_fact_history("用户", "document.output_format")

        assert first.ok and second.ok
        assert current is not None
        assert current.value == "PDF"
        assert current.version == 2
        assert len(history) == 1
        assert history[0].value == "Markdown"


def test_semantic_attribute_alignment_shares_version_chain() -> None:
    with TemporaryDirectory() as tmp:
        module = AgentMemoryModule(Path(tmp) / "memory")
        first = module.remember(
            RememberRequest(
                content="The response style is concise.",
                entity="user",
                attribute="response_style",
                value="concise",
                source_session_id="first",
            )
        )
        second = module.update(
            UpdateRequest(
                content="Change the response style to detailed.",
                entity="user",
                attribute="response style",
                value="detailed",
                source_session_id="second",
            )
        )

        current = module.core.get("user", "response_style")
        history = module.archival.get_fact_history("user", "response_style")
        assert first.ok and second.ok
        assert current is not None and current.value == "detailed"
        assert current.version == 2
        assert len(history) == 1 and history[0].value == "concise"


def test_explicit_fact_scope_filters_before_semantic_search() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        config = MemoryConfig(root / "config.json")
        core = CoreMemoryStore(root / "memory", config=config)
        archival = ArchivalMemoryStore(root / "memory", config=config)
        core.set(
            "用户::曙光工程",
            "document.output_format",
            "PDF",
            scope="曙光工程",
        )
        core.set(
            "用户::远帆工程",
            "document.output_format",
            "DOCX",
            scope="远帆工程",
        )
        semantic = CapturingSemanticIndex()
        module = AgentMemoryModule(
            root / "memory",
            core=core,
            archival=archival,
            config=config,
            semantic_index=semantic,  # type: ignore[arg-type]
        )

        result = module.recall(
            RecallRequest(
                query="文档输出格式",
                scope="ltm",
                fact_scope="曙光工程",
                limit=5,
            )
        )

        assert {fact.scope for fact in semantic.candidates} == {"曙光工程"}
        assert result.records[0].value == "PDF"


def test_explicit_attribute_filter_is_not_relaxed_on_empty_result() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        config = MemoryConfig(root / "config.json")
        core = CoreMemoryStore(root / "memory", config=config)
        archival = ArchivalMemoryStore(root / "memory", config=config)
        core.set("用户", "ui.theme", "dark")
        semantic = CapturingSemanticIndex()
        module = AgentMemoryModule(
            root / "memory",
            core=core,
            archival=archival,
            config=config,
            semantic_index=semantic,  # type: ignore[arg-type]
        )

        result = module.recall(
            RecallRequest(
                query="theme",
                scope="ltm",
                attribute="document.output_format",
            )
        )

        assert not result.ok
        assert not semantic.candidates
