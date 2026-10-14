from __future__ import annotations

import json
from collections import deque
from typing import Any

import pytest

from os_agent.llm.provider import LLMConfig, LLMError
from os_agent.memory.mem0_archival import ArchivalMemoryStore
from os_agent.memory.mem0_core import CoreMemoryStore
from os_agent.memory.mem0_pipeline import ExtractUpdatePipeline
from os_agent.memory.memory_config import MemoryConfig
from os_agent.memory.semantic_extractor import SemanticMemoryExtractor
from os_agent.memory.state_assembler import StateAssembler


class _RoutedProvider:
    """Deterministic fake for independently authored synthetic examples."""

    enabled = True

    def __init__(
        self,
        mention_results: list[dict[str, Any] | Exception],
        relations: list[str | dict[str, str]] | None = None,
    ) -> None:
        self.mention_results = deque(mention_results)
        self.relations = deque(relations or [])
        self.relation_payloads: list[dict[str, Any]] = []
        self.last_usage: dict[str, int] = {}

    def complete(self, messages, response_format: str = "text") -> str:
        assert response_format == "json"
        system = messages[0].content
        if "property identity" in system:
            payload = json.loads(messages[1].content)
            self.relation_payloads.append(payload)
            relation_spec = self.relations.popleft()
            if isinstance(relation_spec, dict):
                return json.dumps(relation_spec)
            relation = relation_spec
            candidate_id = (
                payload["candidates"][0]["candidate_id"]
                if relation == "same_slot"
                else ""
            )
            return json.dumps(
                {"relation": relation, "candidate_id": candidate_id}
            )
        result = self.mention_results.popleft()
        if isinstance(result, Exception):
            raise result
        return json.dumps(result)


def _fact(
    attribute: str,
    attribute_evidence: str,
    value: str,
    evidence: str,
    *,
    intent: str = "set",
    scope: str = "",
    certainty: str = "certain",
    temporal_status: str = "stable",
) -> dict[str, str]:
    return {
        "attribute": attribute,
        "attribute_evidence": attribute_evidence,
        "value": value,
        "scope": scope,
        "evidence": evidence,
        "intent": intent,
        "certainty": certainty,
        "temporal_status": temporal_status,
    }


def _result(*facts: dict[str, str], index: int = 0) -> dict[str, Any]:
    return {"results": [{"index": index, "facts": list(facts)}]}


def _event(
    text: str,
    event_id: str,
    *,
    scope: str = "",
    user_id: str = "operator",
) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": "user", "content": text}
    if scope:
        payload["scope"] = scope
    return {
        "event_type": "message_start",
        "event_id": event_id,
        "session_id": f"session-{event_id}",
        "user_id": user_id,
        "payload": payload,
    }


def _pipeline(tmp_path, provider: _RoutedProvider):
    config = MemoryConfig(tmp_path / "config.json")
    config.set_value("pipeline_behavior_enabled", False)
    config.set_value("pipeline_session_preference_consolidation_enabled", False)
    config.set_value("pipeline_preference_reinforcement_enabled", False)
    core = CoreMemoryStore(tmp_path / "memory", config=config)
    archival = ArchivalMemoryStore(tmp_path / "memory", config=config)
    extractor = SemanticMemoryExtractor(provider)
    pipeline = ExtractUpdatePipeline(
        core,
        archival,
        config=config,
        semantic_extractor=extractor,
    )
    return pipeline, core, archival


def test_semantic_paraphrase_updates_one_version_chain_without_values_in_relation(
    tmp_path,
) -> None:
    first = "I prefer the orbital retry window at seven seconds."
    second = "Please change the retry delay for orbital jobs to nine seconds."
    provider = _RoutedProvider(
        [
            _result(
                _fact(
                    "orbital retry window",
                    "orbital retry window",
                    "seven seconds",
                    first,
                )
            ),
            _result(
                _fact(
                    "retry delay for orbital jobs",
                    "retry delay for orbital jobs",
                    "nine seconds",
                    second,
                    intent="update",
                )
            ),
        ],
        ["same_slot"],
    )
    pipeline, core, archival = _pipeline(tmp_path, provider)

    pipeline.process_session([_event(first, "one")], "session-one")
    pipeline.process_session([_event(second, "two")], "session-two")

    active = core.active_facts()
    assert len(active) == 1
    assert active[0].value == "nine seconds"
    assert active[0].version == 2
    assert len(archival.get_fact_history(active[0].entity, active[0].attribute)) == 1
    assert provider.relation_payloads
    assert "value" not in json.dumps(provider.relation_payloads[0]).casefold()


def test_different_properties_with_the_same_value_coexist(tmp_path) -> None:
    first = "I prefer seven levels for review font size."
    second = "I prefer seven levels for review contrast."
    provider = _RoutedProvider(
        [
            _result(
                _fact("review font size", "review font size", "seven", first)
            ),
            _result(
                _fact("review contrast", "review contrast", "seven", second)
            ),
        ],
        ["independent"],
    )
    pipeline, core, _ = _pipeline(tmp_path, provider)

    pipeline.process_session([_event(first, "font")], "session-font")
    pipeline.process_session([_event(second, "contrast")], "session-contrast")

    assert sorted(fact.value for fact in core.active_facts()) == ["seven", "seven"]
    assert len({fact.attribute for fact in core.active_facts()}) == 2


def test_identical_properties_in_distinct_scopes_coexist_without_relation_call(
    tmp_path,
) -> None:
    red = "I prefer the build channel at stable."
    blue = "I prefer the build channel at preview."
    provider = _RoutedProvider(
        [
            _result(
                _fact("build channel", "build channel", "stable", red, scope="red")
            ),
            _result(
                _fact("build channel", "build channel", "preview", blue, scope="blue")
            ),
        ]
    )
    pipeline, core, _ = _pipeline(tmp_path, provider)

    pipeline.process_session([_event(red, "red", scope="red")], "session-red")
    pipeline.process_session([_event(blue, "blue", scope="blue")], "session-blue")

    assert {(fact.scope, fact.value) for fact in core.active_facts()} == {
        ("red", "stable"),
        ("blue", "preview"),
    }
    assert provider.relation_payloads == []


def test_duplicate_surface_formatting_reinforces_without_new_version(tmp_path) -> None:
    first = "I prefer the orbital retry window at seven seconds."
    second = "I prefer the orbital retry window at seven seconds."
    provider = _RoutedProvider(
        [
            _result(
                _fact(
                    "orbital retry window",
                    "orbital retry window",
                    "seven seconds",
                    first,
                )
            ),
            _result(
                _fact(
                    "orbital retry window",
                    "orbital retry window",
                    "seven seconds.",
                    second,
                )
            ),
        ]
    )
    pipeline, core, archival = _pipeline(tmp_path, provider)

    pipeline.process_session([_event(first, "first")], "session-first")
    pipeline.process_session([_event(second, "second")], "session-second")

    active = core.active_facts()
    assert len(active) == 1
    assert active[0].version == 1
    assert archival.get_fact_history(active[0].entity, active[0].attribute) == []


def test_unvalued_revert_restores_previous_version_in_selected_slot(tmp_path) -> None:
    first = "Set the orbital retry window to seven seconds."
    second = "Change the orbital retry window to nine seconds."
    third = "Restore the orbital retry window to its previous value."
    provider = _RoutedProvider(
        [
            _result(
                _fact(
                    "orbital retry window",
                    "orbital retry window",
                    "seven seconds",
                    first,
                )
            ),
            _result(
                _fact(
                    "orbital retry window",
                    "orbital retry window",
                    "nine seconds",
                    second,
                    intent="update",
                )
            ),
            _result(
                _fact(
                    "orbital retry window",
                    "orbital retry window",
                    "",
                    third,
                    intent="revert",
                    temporal_status="reverted",
                )
            ),
        ]
    )
    pipeline, core, archival = _pipeline(tmp_path, provider)

    for index, text in enumerate((first, second, third)):
        pipeline.process_session(
            [_event(text, str(index))], f"session-{index}"
        )

    active = core.active_facts()
    assert len(active) == 1
    assert active[0].value == "seven seconds"
    assert active[0].version == 3
    assert [fact.value for fact in archival.get_fact_history(
        active[0].entity, active[0].attribute
    )] == ["nine seconds", "seven seconds"]


def test_ambiguous_property_relation_never_overwrites_existing_state(tmp_path) -> None:
    first = "I prefer the orbital retry window at seven seconds."
    second = "I prefer orbital timing at nine seconds."
    provider = _RoutedProvider(
        [
            _result(
                _fact(
                    "orbital retry window",
                    "orbital retry window",
                    "seven seconds",
                    first,
                )
            ),
            _result(
                _fact("orbital timing", "orbital timing", "nine seconds", second)
            ),
        ],
        ["ambiguous"],
    )
    pipeline, core, archival = _pipeline(tmp_path, provider)

    pipeline.process_session([_event(first, "one")], "session-one")
    pipeline.process_session([_event(second, "two")], "session-two")

    active = core.active_facts()
    assert {fact.value for fact in active} == {"seven seconds", "nine seconds"}
    uncertain = next(fact for fact in active if fact.value == "nine seconds")
    assert uncertain.custom_metadata["state_status"] == "needs_recheck"
    assert archival.get_fact_history(
        next(fact for fact in active if fact.value == "seven seconds").entity,
        next(fact for fact in active if fact.value == "seven seconds").attribute,
    ) == []


def test_nonpersistent_language_is_vetoed_after_semantic_proposal(tmp_path) -> None:
    texts = [
        "Do I prefer the build channel at stable?",
        "For this task only, I prefer the build channel at preview.",
        "The release note says I prefer the build channel at nightly.",
    ]
    proposals = [
        _fact("build channel", "build channel", "stable", texts[0]),
        _fact(
            "build channel",
            "build channel",
            "preview",
            texts[1],
            temporal_status="temporary",
        ),
        _fact(
            "build channel",
            "build channel",
            "nightly",
            texts[2],
            certainty="reported",
        ),
    ]
    provider = _RoutedProvider(
        [{"results": [
            {"index": index, "facts": [proposal]}
            for index, proposal in enumerate(proposals)
        ]}]
    )
    pipeline, core, _ = _pipeline(tmp_path, provider)
    events = [_event(text, str(index)) for index, text in enumerate(texts)]

    pipeline.process_session(events, "nonpersistent")

    assert core.active_facts() == []
    assert pipeline.runtime_stats["nonpersistent_facts"] == 3


def test_multiple_mentions_are_applied_in_message_order(tmp_path) -> None:
    text = (
        "I prefer the orbital retry window at seven seconds; "
        "change the orbital retry window to nine seconds."
    )
    provider = _RoutedProvider(
        [_result(
            _fact(
                "orbital retry window",
                "orbital retry window",
                "seven seconds",
                "I prefer the orbital retry window at seven seconds",
            ),
            _fact(
                "orbital retry window",
                "orbital retry window",
                "nine seconds",
                "change the orbital retry window to nine seconds",
                intent="update",
            ),
        )]
    )
    pipeline, core, archival = _pipeline(tmp_path, provider)

    pipeline.process_session([_event(text, "ordered")], "ordered")

    active = core.active_facts()
    assert len(active) == 1
    assert active[0].value == "nine seconds"
    assert active[0].version == 2
    assert len(archival.get_fact_history(active[0].entity, active[0].attribute)) == 1


def test_semantic_transport_failure_uses_configured_rule_fallback(tmp_path) -> None:
    provider = _RoutedProvider([LLMError("synthetic outage")])
    pipeline, core, _ = _pipeline(tmp_path, provider)
    text = "change the orbital retry window to seven"

    pipeline.process_session([_event(text, "fallback")], "fallback")

    active = [fact for fact in core.active_facts() if "preference" in fact.tags]
    assert len(active) == 1
    assert active[0].attribute == "preference.open.orbital_retry_window"
    assert active[0].value == "seven"


@pytest.mark.parametrize("relation", ["ambiguous", "independent"])
def test_nonmatching_relation_must_not_return_a_target(tmp_path, relation) -> None:
    first = "I prefer the launch cadence at weekly."
    second = "I prefer the launch rhythm at monthly."
    provider = _RoutedProvider(
        [
            _result(_fact("launch cadence", "launch cadence", "weekly", first)),
            _result(_fact("launch rhythm", "launch rhythm", "monthly", second)),
        ],
        [relation],
    )
    pipeline, core, _ = _pipeline(tmp_path, provider)
    pipeline.process_session([_event(first, "first")], "first")
    pipeline.process_session([_event(second, "second")], "second")
    assert len(core.active_facts()) == 2


@pytest.mark.parametrize(
    "invalid_relation",
    [
        {"relation": "same_slot", "candidate_id": "missing-fact"},
        {"relation": "independent", "candidate_id": "illegal-target"},
    ],
)
def test_invalid_relation_response_degrades_to_ambiguous_without_overwrite(
    tmp_path, invalid_relation
) -> None:
    first = "I prefer the launch cadence at weekly."
    second = "I prefer the launch rhythm at monthly."
    provider = _RoutedProvider(
        [
            _result(_fact("launch cadence", "launch cadence", "weekly", first)),
            _result(_fact("launch rhythm", "launch rhythm", "monthly", second)),
        ],
        [invalid_relation],
    )
    pipeline, core, archival = _pipeline(tmp_path, provider)

    pipeline.process_session([_event(first, "first")], "first")
    pipeline.process_session([_event(second, "second")], "second")

    assert {fact.value for fact in core.active_facts()} == {"weekly", "monthly"}
    assert any(
        fact.custom_metadata.get("state_status") == "needs_recheck"
        for fact in core.active_facts()
    )
    assert archival.get_fact_history(
        next(fact for fact in core.active_facts() if fact.value == "weekly").entity,
        next(fact for fact in core.active_facts() if fact.value == "weekly").attribute,
    ) == []
    assert pipeline.runtime_stats["relation_errors"] == 1


def test_forged_revert_cannot_roll_back_a_plain_assertion(tmp_path) -> None:
    text = "I prefer my drink temperature cold."
    provider = _RoutedProvider(
        [_result(
            _fact(
                "drink temperature",
                "drink temperature",
                "cold",
                text,
                intent="revert",
                temporal_status="reverted",
            )
        )]
    )
    extractor = SemanticMemoryExtractor(provider)

    with pytest.raises(LLMError, match="revert"):
        extractor.extract([_event(text, "forged-revert")], "forged-revert")


def test_fact_local_veto_does_not_drop_stable_clause_in_same_message(tmp_path) -> None:
    text = (
        "For this task only, I prefer the preview build channel. "
        "I prefer my review font size at large."
    )
    provider = _RoutedProvider(
        [_result(
            _fact(
                "build channel",
                "build channel",
                "preview",
                "For this task only, I prefer the preview build channel",
                temporal_status="temporary",
            ),
            _fact(
                "review font size",
                "review font size",
                "large",
                "I prefer my review font size at large",
            ),
        )]
    )
    pipeline, core, _ = _pipeline(tmp_path, provider)

    pipeline.process_session([_event(text, "mixed")], "mixed")

    assert [(fact.attribute, fact.value) for fact in core.active_facts()] == [
        ("preference.open.review_font_size", "large")
    ]


def test_two_natural_scopes_in_one_message_remain_independent(tmp_path) -> None:
    alpha = "For project alpha, I prefer the build channel stable"
    beta = "For project beta, I prefer the build channel preview"
    text = f"{alpha}. {beta}."
    provider = _RoutedProvider(
        [_result(
            _fact(
                "build channel",
                "build channel",
                "stable",
                alpha,
                scope="project alpha",
            ),
            _fact(
                "build channel",
                "build channel",
                "preview",
                beta,
                scope="project beta",
            ),
        )]
    )
    pipeline, core, _ = _pipeline(tmp_path, provider)

    pipeline.process_session([_event(text, "two-scopes")], "two-scopes")

    assert {(fact.scope, fact.value) for fact in core.active_facts()} == {
        ("project alpha", "stable"),
        ("project beta", "preview"),
    }


def test_candidate_truncation_prefers_recent_facts_when_overlap_ties() -> None:
    provider = _RoutedProvider([], ["independent"])
    assembler = StateAssembler(provider, context_limit=2)
    incoming = _semantic_card("operator", "unrelated incoming", "new")
    candidates = [
        _semantic_card("operator", "alpha property", "a", updated_at="2026-01-01"),
        _semantic_card("operator", "beta property", "b", updated_at="2026-02-01"),
        _semantic_card("operator", "gamma property", "c", updated_at="2026-03-01"),
    ]

    assembler.resolve_slot(incoming, candidates)

    sent = provider.relation_payloads[0]["candidates"]
    assert [item["attribute"] for item in sent] == [
        "gamma property",
        "beta property",
    ]


def _semantic_card(
    entity: str,
    attribute: str,
    value: str,
    *,
    scope: str = "",
    updated_at: str = "2026-01-01",
):
    from os_agent.memory.mem0_schemas import FactCard

    return FactCard(
        entity=entity,
        attribute=attribute,
        value=value,
        evidence=f"{attribute} {value}",
        source_session_id="synthetic",
        source_event_id=f"{attribute}-{value}",
        tags=["preference", "source:semantic_mention"],
        scope=scope,
        updated_at=updated_at,
        custom_metadata={
            "extraction": "semantic_mention",
            "attribute_surface": attribute,
            "attribute_aliases": [attribute],
        },
    )


def test_surface_alias_does_not_bypass_property_relation_gate(tmp_path) -> None:
    first = "I prefer the document format at compact."
    second = "I prefer the date format at compact."
    third = "Change the date format to expanded."
    provider = _RoutedProvider(
        [
            _result(_fact("document format", "document format", "compact", first)),
            _result(_fact("date format", "date format", "compact", second)),
            _result(
                _fact(
                    "date format",
                    "date format",
                    "expanded",
                    third,
                    intent="update",
                )
            ),
        ],
        ["same_slot", "independent"],
    )
    pipeline, core, _ = _pipeline(tmp_path, provider)

    pipeline.process_session([_event(first, "first")], "first")
    pipeline.process_session([_event(second, "second")], "second")
    pipeline.process_session([_event(third, "third")], "third")

    assert len(provider.relation_payloads) == 2
    assert {fact.value for fact in core.active_facts()} == {"compact", "expanded"}


def test_same_slot_update_keeps_target_scope_and_attribute_tag(tmp_path) -> None:
    first = "I prefer the document layout compact."
    second = "Change the page arrangement to spacious."
    provider = _RoutedProvider(
        [
            _result(
                _fact(
                    "document layout",
                    "document layout",
                    "compact",
                    first,
                    scope="workspace-alpha",
                )
            ),
            _result(
                _fact(
                    "page arrangement",
                    "page arrangement",
                    "spacious",
                    second,
                    intent="update",
                    scope="workspace_alpha",
                )
            ),
        ],
        ["same_slot"],
    )
    pipeline, core, _ = _pipeline(tmp_path, provider)

    pipeline.process_session(
        [_event(first, "first", scope="workspace-alpha")], "first"
    )
    pipeline.process_session(
        [_event(second, "second", scope="workspace_alpha")], "second"
    )

    active = core.active_facts()
    assert len(active) == 1
    assert active[0].entity == "operator::workspace-alpha"
    assert active[0].scope == "workspace-alpha"
    assert f"attribute:{active[0].attribute}" in active[0].tags
    assert "attribute:preference.open.page_arrangement" not in active[0].tags


def test_llm_config_uses_real_defaults_with_only_an_api_key(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-key")
    monkeypatch.setenv("OS_AGENT_LLM_CONFIG", str(tmp_path / "missing.json"))
    for name in ("OPENAI_BASE_URL", "OS_AGENT_BASE_URL", "OPENAI_MODEL", "OS_AGENT_MODEL"):
        monkeypatch.delenv(name, raising=False)

    config = LLMConfig.from_env()

    assert config.base_url == "https://api.openai.com/v1"
    assert config.model == "gpt-4o-mini"


def test_constructor_falls_back_to_rules_when_semantic_credentials_are_unavailable(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("OS_AGENT_MEMORY_EXTRACTOR", "semantic_state")
    monkeypatch.setenv("OS_AGENT_LLM_CONFIG", str(tmp_path / "missing.json"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OS_AGENT_API_KEY", raising=False)
    config = MemoryConfig(tmp_path / "config.json")
    core = CoreMemoryStore(tmp_path / "memory", config=config)
    archival = ArchivalMemoryStore(tmp_path / "memory", config=config)

    pipeline = ExtractUpdatePipeline(core, archival, config=config)

    assert pipeline.semantic_extractor is None
