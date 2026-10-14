from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from os_agent.memory.interface import RecallRequest
from os_agent.memory.mem0_schemas import FactCard, RecallResult
from os_agent.memory.module import AgentMemoryModule
from os_agent.memory.query_rewriter import QueryRewriter


def fact(fact_id: str, value: str) -> FactCard:
    return FactCard(
        fact_id=fact_id,
        entity="service",
        attribute="throughput",
        value=value,
        evidence=value,
        source_session_id=f"session-{fact_id}",
        source_event_id=f"event-{fact_id}",
    )


class RouteRecallStrategy:
    max_total = 8

    def __init__(self, routes: dict[str, list[FactCard]]) -> None:
        self.routes = routes
        self.queries: list[str] = []

    def recall(self, query: str, **_: object) -> RecallResult:
        self.queries.append(query)
        return RecallResult(
            facts=list(self.routes.get(query, [])),
            source_layer="core",
            query=query,
        )


def test_query_plan_keeps_routes_separate_and_bounded() -> None:
    plan = QueryRewriter().plan(
        "compare deployments; inspect rollback history",
        variants=["deployment comparison", "deployment comparison"],
        subqueries=["rollback history", "capacity limits"],
    )

    assert plan.original == "compare deployments; inspect rollback history"
    assert plan.variants == ("deployment comparison",)
    assert "compare deployments" in plan.subqueries
    assert "inspect rollback history" in plan.subqueries
    assert "rollback history" in plan.subqueries
    assert len(plan.retrieval_queries) <= QueryRewriter.MAX_ROUTES


def test_multi_query_recall_preserves_one_result_per_subquery() -> None:
    dominant = fact("dominant", "comparison overview")
    alternate = fact("alternate", "general system comparison")
    alpha = fact("alpha", "alpha throughput is 120")
    beta = fact("beta", "beta throughput is 90")
    routes = {
        "compare systems": [dominant, alternate],
        "system comparison": [dominant, alternate],
        "alpha throughput": [alpha],
        "beta throughput": [beta],
    }

    with TemporaryDirectory() as tmp:
        module = AgentMemoryModule(Path(tmp) / "memory")
        strategy = RouteRecallStrategy(routes)
        module.recall_strategy = strategy  # type: ignore[assignment]

        result = module.recall(
            RecallRequest(
                query="compare systems",
                scope="ltm",
                limit=2,
                query_variants=["system comparison"],
                subqueries=["alpha throughput", "beta throughput"],
            )
        )

    assert strategy.queries == [
        "compare systems",
        "system comparison",
        "alpha throughput",
        "beta throughput",
    ]
    assert {record.memory_id for record in result.records} == {"alpha", "beta"}
    assert all(record.metadata["subquery_hits"] for record in result.records)


def test_single_query_recall_remains_one_route() -> None:
    target = fact("target", "one route result")
    with TemporaryDirectory() as tmp:
        module = AgentMemoryModule(Path(tmp) / "memory")
        strategy = RouteRecallStrategy({"plain lookup": [target]})
        module.recall_strategy = strategy  # type: ignore[assignment]

        result = module.recall(
            RecallRequest(query="plain lookup", scope="ltm", limit=1)
        )

    assert strategy.queries == ["plain lookup"]
    assert [record.memory_id for record in result.records] == ["target"]


def test_query_rewriter_normalizes_chinese_language_acts_and_coordinates() -> None:
    rewriter = QueryRewriter()
    normalized = rewriter.plan("请问文档输出格式？")
    assert normalized.variants[0] == "文档输出格式"

    coordinated = rewriter.plan("比较部署，同时查看回滚历史")
    assert "比较部署" in coordinated.subqueries
    assert "查看回滚历史" in coordinated.subqueries
