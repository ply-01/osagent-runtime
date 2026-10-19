from __future__ import annotations


from os_agent.memory.embedding_cache import EmbeddingCache
from os_agent.memory.fact_vector_index import FactSemanticIndex
from os_agent.memory.interface import RecallRequest, RememberRequest, UpdateRequest
from os_agent.memory.mem0_archival import ArchivalMemoryStore
from os_agent.memory.mem0_core import CoreMemoryStore
from os_agent.memory.mem0_pipeline import ExtractUpdatePipeline
from os_agent.memory.mem0_schemas import FactCard
from os_agent.memory.memory_config import MemoryConfig
from os_agent.memory.memory_frame import MemoryFrameParser
from os_agent.memory.module import AgentMemoryModule
from os_agent.memory.query_rewriter import QueryRewriter
from os_agent.memory.retrieval_text import expand_query_terms, tokenize_search_text
from os_agent.memory.semantic_alignment import find_aligned_fact


class _Provider:
    enabled = True
    required = False
    cache_namespace = "test-provider"

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class _VectorStore:
    def __init__(self) -> None:
        self.keys: list[str] = []
        self.allowed_keys: object = object()

    def replace(self, documents: list[tuple[str, list[float]]]) -> int:
        self.keys = [key for key, _ in documents]
        return len(documents)

    def search(
        self,
        query: list[float],
        limit: int,
        *,
        allowed_keys: object = None,
    ) -> list[tuple[str, float]]:
        self.allowed_keys = allowed_keys
        return [(self.keys[0], 0.9)] if self.keys else []


def _fact(entity: str, attribute: str, value: str, *, scope: str = "") -> FactCard:
    return FactCard(
        entity=entity,
        attribute=attribute,
        value=value,
        evidence=f"{attribute}={value}",
        source_session_id="session",
        source_event_id="event",
        scope=scope,
        intent="update",
    )


def test_lexical_expansion_contains_no_manual_synonym_vocabulary() -> None:
    query = "OrbitalRetryWindow alpha_beta"
    assert expand_query_terms(query) == tokenize_search_text(query)
    assert {"orbital", "retry", "window", "alpha", "beta"} <= set(
        tokenize_search_text(query)
    )


def test_open_domain_mutations_share_a_stable_slot() -> None:
    parser = MemoryFrameParser()
    first = parser.parse("change the orbital retry window to seven")
    second = parser.parse("restore the orbital retry window to five")

    assert first is not None and second is not None
    assert first.attribute == second.attribute == "preference.open.orbital_retry_window"
    assert first.value == "seven"
    assert second.value == "five"
    aligned = find_aligned_fact(
        _fact(first.storage_entity, first.attribute, first.value, scope=first.scope),
        [_fact(second.storage_entity, second.attribute, second.value, scope=second.scope)],
    )
    assert aligned is not None


def test_unverified_direct_memory_write_does_not_replace_confirmed_fact(
    tmp_path,
) -> None:
    module = AgentMemoryModule(tmp_path / "memory")
    confirmed = module.remember(
        RememberRequest(
            content="I prefer seven-second orbital retries",
            entity="operator",
            attribute="preference.open.orbital_retry_window",
            value="seven",
        )
    )

    tentative = module.update(
        UpdateRequest(
            content="maybe I prefer nine-second orbital retries",
            entity="operator",
            attribute="preference.open.orbital_retry_window",
            value="nine",
        )
    )

    assert confirmed.records
    assert tentative.ok is True
    assert tentative.records == []
    active = module.core.active_facts()
    assert len(active) == 1
    assert active[0].value == "seven"
    assert active[0].version == 1


def test_open_domain_conflict_chain_keeps_one_active_version(tmp_path) -> None:
    config = MemoryConfig(tmp_path / "config.json")
    core = CoreMemoryStore(tmp_path / "memory", config=config)
    archival = ArchivalMemoryStore(tmp_path / "memory", config=config)
    pipeline = ExtractUpdatePipeline(core, archival, config=config)
    statements = [
        "change the orbital retry window to seven",
        "change the orbital retry window to nine",
        "maybe change the orbital retry window to eleven",
        "restore the orbital retry window to five",
    ]

    for index, statement in enumerate(statements):
        session_id = f"session-{index}"
        pipeline.process_session(
            [
                {
                    "event_type": "message_start",
                    "event_id": f"event-{index}",
                    "session_id": session_id,
                    "payload": {"role": "user", "content": statement},
                }
            ],
            session_id,
        )

    active = [fact for fact in core.active_facts() if "preference" in fact.tags]
    assert [(fact.attribute, fact.value) for fact in active] == [
        ("preference.open.orbital_retry_window", "five")
    ]


def test_open_domain_chinese_mutations_share_a_stable_slot() -> None:
    parser = MemoryFrameParser()
    first = parser.parse(
        "\u628a\u8f68\u9053\u91cd\u8bd5\u7a97\u53e3\u6539\u6210\u4e03"
    )
    second = parser.parse(
        "\u628a\u8f68\u9053\u91cd\u8bd5\u7a97\u53e3\u6539\u56de\u4e94"
    )

    assert first is not None and second is not None
    assert first.attribute == second.attribute == "preference.open.\u8f68\u9053\u91cd\u8bd5\u7a97\u53e3"
    assert first.value == "\u4e03"
    assert second.value == "\u4e94"


def test_scoped_and_embedded_entity_forms_align() -> None:
    left = _fact("operator::workspace-alpha", "retention_window", "7", scope="workspace-alpha")
    right = _fact("operator", "retention-window", "14", scope="workspace_alpha")

    assert find_aligned_fact(left, [right]) is not None


def test_equal_values_never_make_distinct_unclassified_attributes_align() -> None:
    left = _fact("operator", "preference.unclassified.review_font", "seven")
    right = _fact("operator", "preference.unclassified.review_contrast", "seven")

    assert find_aligned_fact(left, [right]) is None


def test_query_constraints_keep_all_explicitly_mentioned_entities() -> None:
    parser = MemoryFrameParser()

    constraints = parser.query_constraints(
        "compare VegaService latency with OrionService latency",
        known_entities=["OrionService", "VegaService", "UnrelatedService"],
        known_attributes=["latency"],
    )

    assert constraints.entity is None
    assert constraints.entities == ("VegaService", "OrionService")


def test_multi_entity_recall_reserves_evidence_for_each_entity(tmp_path) -> None:
    module = AgentMemoryModule(tmp_path / "memory")
    module.remember(
        RememberRequest(
            content="VegaService latency is 18ms",
            entity="VegaService",
            attribute="latency",
            value="18ms",
        )
    )
    module.remember(
        RememberRequest(
            content="OrionService latency is 27ms",
            entity="OrionService",
            attribute="latency",
            value="27ms",
        )
    )
    for index in range(12):
        module.remember(
            RememberRequest(
                content=f"OtherService{index} latency is {index}ms",
                entity=f"OtherService{index}",
                attribute="latency",
                value=f"{index}ms",
            )
        )

    result = module.recall(
        RecallRequest(
            query="compare VegaService latency with OrionService latency",
            scope="ltm",
            limit=2,
        )
    )

    assert {record.entity for record in result.records} == {
        "VegaService",
        "OrionService",
    }


def test_core_search_reuses_fact_tokenization(tmp_path) -> None:
    store = CoreMemoryStore(tmp_path)
    store.set("object-a", "orbital_window", "seven", evidence="stable evidence")
    store.set("object-b", "orbital_window", "nine", evidence="other evidence")
    original = store._tokenize
    calls = 0

    def counting_tokenize(text: str) -> list[str]:
        nonlocal calls
        calls += 1
        return original(text)

    store._tokenize = counting_tokenize  # type: ignore[method-assign]
    store.search("object-a orbital window", top_k=2)
    first_pass = calls
    store.search("object-a orbital window", top_k=2)

    assert first_pass == 2
    assert calls == first_pass


def test_batched_core_search_matches_independent_routes(tmp_path) -> None:
    store = CoreMemoryStore(tmp_path)
    facts = [
        _fact("VegaService", "latency", "18ms", scope="workspace-alpha"),
        _fact("OrionService", "latency", "27ms", scope="workspace-alpha"),
        _fact("OtherService", "owner", "platform", scope="workspace-beta"),
    ]
    store._facts = {fact.key: fact for fact in facts}
    store._invalidate_lexical_index()
    queries = ["VegaService latency", "OrionService latency"]

    independent = [
        store.search(
            query,
            top_k=3,
            attribute_filter="latency",
            scope_filter="workspace-alpha",
        )
        for query in queries
    ]
    batched = store.search_many(
        queries,
        top_k=3,
        attribute_filter="latency",
        scope_filter="workspace-alpha",
    )

    assert [[fact.fact_id for fact in row] for row in batched] == [
        [fact.fact_id for fact in row] for row in independent
    ]


def test_full_semantic_snapshot_skips_allowed_key_expression() -> None:
    vector_store = _VectorStore()
    index = FactSemanticIndex(
        provider=_Provider(),
        cache=EmbeddingCache(),
        vector_store=vector_store,
    )
    facts = [_fact("object-a", "orbital_window", "seven")]

    index.refresh(facts)
    hits = index.search("orbital window", facts, 1)

    assert hits[0].fact is facts[0]
    assert vector_store.allowed_keys is None


def test_semantic_routes_share_one_matrix_search() -> None:
    index = FactSemanticIndex(provider=_Provider(), cache=EmbeddingCache())
    facts = [
        _fact("object-a", "orbital_window", "seven"),
        _fact("object-b", "orbital_window", "nine"),
    ]
    index.refresh(facts)

    results = index.search_many(
        ["orbital window", "retry setting"],
        facts,
        2,
        query_embeddings=[[1.0, 0.0], [1.0, 0.0]],
    )

    assert len(results) == 2
    assert all(len(route) == 2 for route in results)


def test_normalized_query_keeps_original_context_as_a_batched_route() -> None:
    plan = QueryRewriter().plan(
        "please tell me orbital status",
        variants=["orbital health"],
    )

    assert plan.original == "please tell me orbital status"
    assert plan.retrieval_queries == (
        "please tell me orbital status",
        "orbital status",
        "orbital health",
    )


def test_mixed_script_keyword_projection_is_derived_only_from_query() -> None:
    plan = QueryRewriter().plan(
        "\u8bf7\u67e5\u770b OrbitalRetryWindow \u72b6\u6001\uff1f"
    )

    projection_terms = set(plan.variants[-1].split())
    assert {"orbital", "retry", "window"} <= projection_terms


def test_core_inverted_index_avoids_full_fact_rescore(tmp_path) -> None:
    store = CoreMemoryStore(tmp_path)
    store.set("object-a", "orbital_window", "seven", evidence="stable evidence")
    store.set("object-b", "thermal_window", "nine", evidence="other evidence")
    store.search("orbital", top_k=2)

    original = store._search_terms_for_fact
    calls: list[str] = []

    def tracking_terms(fact: FactCard) -> frozenset[str]:
        calls.append(fact.fact_id)
        return original(fact)

    store._search_terms_for_fact = tracking_terms  # type: ignore[method-assign]
    results = store.search("orbital", top_k=2)

    assert [fact.entity for fact in results] == ["object-a"]
    assert calls == []


def test_core_inverted_index_is_invalidated_by_mutation(tmp_path) -> None:
    store = CoreMemoryStore(tmp_path)
    store.set("object-a", "orbital_window", "seven")
    assert store.search("seven", top_k=1)

    store.set("object-a", "orbital_window", "nine")

    assert store.search("seven", top_k=1) == []
    assert [fact.value for fact in store.search("nine", top_k=1)] == ["nine"]


def test_system_state_is_not_persisted_as_a_user_preference() -> None:
    parser = MemoryFrameParser()

    assert parser.parse("the orbital retry window is set to seven") is None
