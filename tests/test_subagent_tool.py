"""Test that main agent can dynamically call subagents as tools."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from os_agent.agent.llm_planner import LLMPlanner
from os_agent.agent.loop import AgentLoop
from os_agent.events.logger import EventLogger
from os_agent.memory.markdown_store import MarkdownMemoryStore
from os_agent.subagents.memory import MemorySubagentHit


class FakeMemorySubagent:
    """Minimal fake subagent for testing."""

    name = "fake-memory"

    def __init__(self):
        self.recall_sessions: list[str] = []
        self.observed: list[str] = []

    def recall(self, query: str, *, session_id: str) -> list[MemorySubagentHit]:
        """Return fake search results."""
        self.recall_sessions.append(session_id)
        return [
            MemorySubagentHit(
                memory_id="test-001",
                content=f"Fake result for query: {query}",
                kind="test",
                score=0.95,
            )
        ]

    def observe_turn(self, *, session_id: str, user_input: str, assistant_output: str) -> dict:
        self.observed.append(session_id)
        return {"status": "ok"}

    def render_context(self, hits: list[MemorySubagentHit]) -> str:
        return "\n".join(hit.content for hit in hits)

    def health(self) -> dict:
        return {"status": "healthy"}


class MutableFakeMemorySubagent(FakeMemorySubagent):
    def __init__(self):
        super().__init__()
        self.managed: list[tuple[str, dict, str]] = []

    def manage(self, action: str, arguments: dict, *, session_id: str) -> dict:
        self.managed.append((action, arguments, session_id))
        return {"ok": True, "action": action}


class CapturingProvider:
    enabled = True

    def __init__(self):
        self.messages = []

    def complete(self, messages, response_format="text"):
        self.messages = messages
        return '{"objective":"route","confidence":1,"steps":[],"done":true}'


def build_test_loop(workspace: Path, **kwargs) -> AgentLoop:
    return AgentLoop(
        workspace=workspace,
        event_logger=EventLogger(workspace / "events"),
        memory_store=MarkdownMemoryStore(workspace / "memory"),
        **kwargs,
    )


class SubagentToolTests(unittest.TestCase):
    """Test dynamic subagent tool registration."""

    def test_subagent_tool_registered_when_subagent_available(self):
        """When a memory subagent exists, it should be registered as a callable tool."""
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            fake_subagent = FakeMemorySubagent()

            loop = build_test_loop(
                workspace=workspace,
                memory_subagent=fake_subagent,
                use_llm=False,
            )

            # Before run(), tool should not be registered
            self.assertNotIn("subagent.memory_search", loop.registry.names())

            # Trigger run() to register the tool
            loop.run("test query", session_id="test-session")

            # After run(), tool should be registered
            self.assertIn("subagent.memory_search", loop.registry.names())

    def test_subagent_tool_not_registered_when_subagent_unavailable(self):
        """When no memory subagent exists, the tool should not be registered."""
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)

            loop = build_test_loop(
                workspace=workspace,
                memory_subagent=None,  # No subagent
                use_llm=False,
            )

            loop.run("test query", session_id="test-session")

            # Tool should not be registered
            self.assertNotIn("subagent.memory_search", loop.registry.names())

    def test_subagent_tool_can_be_called(self):
        """The registered subagent tool should be callable and return results."""
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            fake_subagent = FakeMemorySubagent()

            loop = build_test_loop(
                workspace=workspace,
                memory_subagent=fake_subagent,
                use_llm=False,
            )

            # Register the tool
            loop.run("setup", session_id="test-session")

            # Call the tool directly
            tool = loop.registry.get("subagent.memory_search")
            result = tool.run({"query": "find user preferences"})

            self.assertEqual(result.status, "success")
            self.assertIsInstance(result.output, dict)
            self.assertEqual(result.output["query"], "find user preferences")
            self.assertEqual(result.output["subagent"], "fake-memory")
            self.assertEqual(result.output["hits"], 1)
            self.assertEqual(len(result.output["results"]), 1)
            self.assertEqual(result.output["results"][0]["memory_id"], "test-001")

    def test_tool_mode_does_not_recall_before_planning(self):
        with TemporaryDirectory() as tmp:
            subagent = FakeMemorySubagent()
            loop = build_test_loop(
                workspace=Path(tmp),
                memory_subagent=subagent,
                memory_subagent_mode="tool",
                use_llm=False,
                use_mem0=False,
            )

            loop.run("simple query", session_id="tool-only")

            self.assertEqual(subagent.recall_sessions, [])
            self.assertEqual(subagent.observed, ["tool-only"])

    def test_subagent_tool_tracks_the_current_session(self):
        with TemporaryDirectory() as tmp:
            subagent = FakeMemorySubagent()
            loop = build_test_loop(
                workspace=Path(tmp),
                memory_subagent=subagent,
                memory_subagent_mode="tool",
                use_llm=False,
                use_mem0=False,
            )

            loop.run("first", session_id="session-one")
            loop.registry.get("subagent.memory_search").run({"query": "first"})
            loop.run("second", session_id="session-two")
            loop.registry.get("subagent.memory_search").run({"query": "second"})

            self.assertEqual(
                subagent.recall_sessions,
                ["session-one", "session-two"],
            )

    def test_subagent_tool_uses_current_request_when_query_is_missing(self):
        with TemporaryDirectory() as tmp:
            subagent = FakeMemorySubagent()
            loop = build_test_loop(
                workspace=Path(tmp),
                memory_subagent=subagent,
                memory_subagent_mode="tool",
                use_llm=False,
                use_mem0=False,
            )

            loop.run("find my earlier deployment choice", session_id="fallback-query")
            result = loop.registry.get("subagent.memory_search").run({})

            self.assertEqual(result.status, "success")
            self.assertEqual(
                subagent.recall_sessions,
                ["fallback-query"],
            )
            self.assertEqual(
                result.output["query"],
                "find my earlier deployment choice",
            )

    def test_llm_planner_receives_tool_descriptions(self):
        provider = CapturingProvider()
        planner = LLMPlanner(provider)  # type: ignore[arg-type]

        planner.create_plan(
            user_input="find cross-session memory",
            context="",
            tool_names=["memory.recall", "subagent.memory_search"],
            max_steps=1,
            tool_descriptions={
                "memory.recall": "Search local memory.",
                "subagent.memory_search": "Search complex cross-session memory.",
            },
        )

        prompt = provider.messages[-1].content
        self.assertIn("subagent.memory_search: Search complex cross-session memory.", prompt)

    def test_mutable_subagent_registers_and_executes_manage_tool(self):
        with TemporaryDirectory() as tmp:
            subagent = MutableFakeMemorySubagent()
            loop = build_test_loop(
                workspace=Path(tmp),
                memory_subagent=subagent,
                memory_subagent_mode="tool",
                use_llm=False,
                use_mem0=False,
            )

            loop.run("setup", session_id="manage-session")
            result = loop.registry.get("subagent.memory_manage").run(
                {
                    "action": "remember",
                    "content": "Use the stable release channel.",
                }
            )

            self.assertEqual(result.status, "success")
            self.assertEqual(
                subagent.managed,
                [
                    (
                        "remember",
                        {"content": "Use the stable release channel."},
                        "manage-session",
                    )
                ],
            )


if __name__ == "__main__":
    unittest.main()
