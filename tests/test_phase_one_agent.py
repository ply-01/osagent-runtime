import unittest
import json
import os
import tomllib
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from os_agent.cli import main as cli_main
from os_agent.agent.context_builder import ContextBuilder
from os_agent.agent.hooks import AgentHook, BeforeToolCallDecision
from os_agent.agent.llm_planner import LLMPlanner
from os_agent.agent.loop import AgentLoop
from os_agent.agent.plan import PlanStep, TaskPlan
from os_agent.agent.plan_validator import PlanValidator
from os_agent.events.schemas import AgentEvent, AgentRequest, MemoryCandidate, ToolCall, ToolResult
from os_agent.events.logger import EventLogger
from os_agent.events.session import SessionRecorder
from os_agent.llm.provider import ChatMessage, LLMConfig
from os_agent.memory import embedding_provider as embedding_provider_module
from os_agent.memory.embedding_provider import EmbeddingConfig
from os_agent.memory.embedding_cache import EmbeddingCache
from os_agent.memory.interface import ForgetRequest, RecallRequest, RememberRequest, UpdateRequest
from os_agent.memory.maintenance import MemoryMaintenanceService
from os_agent.memory.markdown_store import MarkdownMemoryStore
from os_agent.memory.memory_config import MemoryConfig
from os_agent.memory.mem0_archival import ArchivalMemoryStore
from os_agent.memory.mem0_core import CoreMemoryStore
from os_agent.memory.mem0_pipeline import ExtractUpdatePipeline
from os_agent.memory.mem0_schemas import FactCard
from os_agent.memory.module import AgentMemoryModule
from os_agent.memory.policy import MemoryPolicy
from os_agent.memory.vector_index import MemoryVectorIndex
from os_agent.skills.registry import SkillRegistry
from os_agent.tools.builtins import build_default_registry
from os_agent.tools.filesystem import ReadFileTool
from os_agent.tools.shell import SafeShellTool


class FakeLLMProvider:
    enabled = True

    def __init__(self) -> None:
        self.calls: list[list[ChatMessage]] = []

    def complete(self, messages: list[ChatMessage], response_format: str = "text") -> str:
        self.calls.append(messages)
        if response_format == "json":
            return (
                '{"objective":"Read project overview","confidence":0.8,'
                '"steps":[{"goal":"Read README.md","tool_name":"filesystem.read_file",'
                '"arguments":{"path":"README.md"},"reason":"Read project overview"}],'
                '"done":false}'
            )
        return "LLM 已读取 README.md，并生成了总结。"


class AdaptiveFakeLLMProvider:
    enabled = True

    def complete(self, messages: list[ChatMessage], response_format: str = "text") -> str:
        if response_format != "json":
            return "LLM 根据两步工具观察生成了总结。"
        prompt = messages[-1].content
        if "# Observations" not in prompt:
            return (
                '{"objective":"Inspect then summarize","confidence":0.7,'
                '"steps":[{"goal":"Inspect workspace","tool_name":"filesystem.list_dir",'
                '"arguments":{"path":"."},"reason":"Inspect workspace first"}],'
                '"done":false}'
            )
        if "filesystem.list_dir: success" in prompt:
            return (
                '{"objective":"Inspect then summarize","confidence":0.8,'
                '"steps":[{"goal":"Read discovered overview","tool_name":"filesystem.read_file",'
                '"arguments":{"path":"README.md"},"reason":"Read discovered overview"}],'
                '"done":false}'
            )
        return '{"objective":"Inspect then summarize","confidence":0.9,"steps":[],"done":true}'


class FakeEmbeddingProvider:
    enabled = True
    cache_namespace = "fake-semantic"

    def __init__(self) -> None:
        self.calls = 0

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        lower = text.lower()
        if "brief" in lower or "concise" in lower:
            return [1.0, 0.0, 0.0]
        if "verbose" in lower:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


class BlockingReadHook(AgentHook):
    def before_tool_call(
        self,
        request: AgentRequest,
        call: ToolCall,
        turn_index: int,
    ) -> BeforeToolCallDecision | None:
        if call.tool_name == "filesystem.read_file":
            return BeforeToolCallDecision(block=True, reason="read disabled in test")
        return None


class PatchAppStateHook(AgentHook):
    def after_tool_result(
        self,
        request: AgentRequest,
        call: ToolCall,
        result: ToolResult,
        turn_index: int,
    ) -> ToolResult | None:
        if call.tool_name != "mock_os.get_app_state" or not result.ok:
            return None
        patched_output = dict(result.output)
        patched_output["active_app"] = "patched-app"
        return ToolResult(
            tool_name=result.tool_name,
            status=result.status,
            input=result.input,
            output=patched_output,
            error=result.error,
            duration_ms=result.duration_ms,
        )


class PhaseOneAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self._previous_env = {
            name: os.environ.get(name)
            for name in [
                "OS_AGENT_LLM_CONFIG",
                "OS_AGENT_API_KEY",
                "OPENAI_API_KEY",
                "OS_AGENT_BASE_URL",
                "OPENAI_BASE_URL",
                "OS_AGENT_MODEL",
                "OPENAI_MODEL",
                "OS_AGENT_EMBEDDING_API_KEY",
                "OPENAI_EMBEDDING_MODEL",
                "OS_AGENT_EMBEDDING_BASE_URL",
                "OS_AGENT_EMBEDDING_MODEL",
                "OS_AGENT_EMBEDDING_TIMEOUT_SECONDS",
            ]
        }
        os.environ["OS_AGENT_LLM_CONFIG"] = "__missing_test_llm_config__.json"
        for name in self._previous_env:
            if name != "OS_AGENT_LLM_CONFIG":
                os.environ.pop(name, None)

    def tearDown(self) -> None:
        for name, value in self._previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_llm_config_loads_local_file_and_env_overrides(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "llm.local.json"
            config_path.write_text(
                json.dumps(
                    {
                        "api_key": "file-key",
                        "base_url": "https://api.deepseek.com",
                        "model": "deepseek-v4-flash",
                        "timeout_seconds": 12,
                        "temperature": 0.3,
                    }
                ),
                encoding="utf-8",
            )
            previous = {
                name: os.environ.get(name)
                for name in [
                    "OS_AGENT_LLM_CONFIG",
                    "OS_AGENT_API_KEY",
                    "OPENAI_API_KEY",
                    "OS_AGENT_MODEL",
                    "OPENAI_MODEL",
                ]
            }
            try:
                os.environ["OS_AGENT_LLM_CONFIG"] = str(config_path)
                os.environ["OS_AGENT_MODEL"] = "env-model"
                os.environ.pop("OS_AGENT_API_KEY", None)
                os.environ.pop("OPENAI_API_KEY", None)
                os.environ.pop("OPENAI_MODEL", None)

                config = LLMConfig.from_env()

                self.assertEqual(config.api_key, "file-key")
                self.assertEqual(config.base_url, "https://api.deepseek.com")
                self.assertEqual(config.model, "env-model")
                self.assertEqual(config.timeout_seconds, 12)
                self.assertEqual(config.temperature, 0.3)
            finally:
                for name, value in previous.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value

    def test_embedding_config_resolves_project_relative_file_like_llm_config(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            project = tmp_path / "project"
            config_dir = project / "config"
            package_dir = project / "src" / "os_agent" / "memory"
            other_cwd = tmp_path / "other"
            config_dir.mkdir(parents=True)
            package_dir.mkdir(parents=True)
            other_cwd.mkdir()
            (config_dir / "llm.local.json").write_text(
                json.dumps(
                    {
                        "embedding_api_key": "embed-key",
                        "embedding_base_url": "https://embedding.example/v1",
                        "embedding_model": "embed-model",
                        "embedding_timeout_seconds": 17,
                    }
                ),
                encoding="utf-8",
            )
            previous_env = os.environ.get("OS_AGENT_LLM_CONFIG")
            previous_file = embedding_provider_module.__file__
            previous_cwd = Path.cwd()
            try:
                os.environ["OS_AGENT_LLM_CONFIG"] = "config/llm.local.json"
                embedding_provider_module.__file__ = str(package_dir / "embedding_provider.py")
                os.chdir(other_cwd)

                config = EmbeddingConfig.from_env()

                self.assertEqual(config.api_key, "embed-key")
                self.assertEqual(config.base_url, "https://embedding.example/v1")
                self.assertEqual(config.model, "embed-model")
                self.assertEqual(config.timeout_seconds, 17)
            finally:
                embedding_provider_module.__file__ = previous_file
                os.chdir(previous_cwd)
                if previous_env is None:
                    os.environ.pop("OS_AGENT_LLM_CONFIG", None)
                else:
                    os.environ["OS_AGENT_LLM_CONFIG"] = previous_env

    def test_pdf_tool_dependency_is_declared(self) -> None:
        data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        dependencies = data["project"]["dependencies"]

        self.assertTrue(any(item.startswith("pdfplumber") for item in dependencies))

    def test_memory_config_can_set_and_reload_tuning_values(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory_config.json"
            cfg = MemoryConfig(path)

            cfg.set_value("recall_min_core_results", 2)
            cfg.set_value("recall_include_legacy_markdown_context", False)
            cfg.save()
            reloaded = MemoryConfig(path)

            self.assertEqual(reloaded.recall_min_core_results, 2)
            self.assertFalse(reloaded.recall_include_legacy_markdown_context)

    def test_memory_config_cli_sets_values(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory_config.json"
            output = StringIO()

            with redirect_stdout(output):
                cli_main(
                    [
                        "memory-config",
                        "set",
                        "--config-file",
                        str(path),
                        "core_search_top_k",
                        "3",
                    ]
                )

            cfg = MemoryConfig(path)
            self.assertEqual(cfg.core_search_top_k, 3)
            self.assertIn("core_search_top_k -> 3", output.getvalue())

    def test_agent_writes_events_and_memory_candidates(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace = tmp_path / "workspace"
            workspace.mkdir()
            (workspace / "README.md").write_text(
                "# Demo\n\nThis is an OS Agent project.",
                encoding="utf-8",
            )

            event_logger = EventLogger(tmp_path / "data" / "events")
            memory_store = MarkdownMemoryStore(tmp_path / "memory")
            agent = AgentLoop(
                workspace=workspace,
                registry=build_default_registry(workspace),
                event_logger=event_logger,
                memory_store=memory_store,
            )

            response = agent.run(
                "帮我查看根目录文件，并总结 README.md",
                session_id="test-session",
            )

            self.assertGreaterEqual(response.steps, 2)
            self.assertIn("filesystem.list_dir", response.content)
            self.assertIn("filesystem.read_file", response.content)
            event_files = list((tmp_path / "data" / "events").glob("*.jsonl"))
            self.assertTrue(event_files)
            event_text = event_files[0].read_text(encoding="utf-8")
            self.assertIn("tool_result", event_text)
            self.assertIn("plan_validated", event_text)
            events = [
                json.loads(line)
                for line in event_files[0].read_text(encoding="utf-8").splitlines()
            ]
            event_types = [event["event_type"] for event in events]
            self.assertEqual(event_types[0], "agent_start")
            self.assertEqual(event_types[1], "message_start")
            self.assertEqual(event_types[2], "message_end")
            self.assertEqual(events[1]["payload"]["role"], "user")
            self.assertEqual(
                events[1]["payload"]["message_id"],
                events[2]["payload"]["message_id"],
            )
            assistant_messages = [
                event
                for event in events
                if event["event_type"] == "message_end"
                and event["payload"]["role"] == "assistant"
            ]
            self.assertTrue(assistant_messages)
            self.assertIn("turn_start", event_types)
            self.assertIn("turn_end", event_types)
            tool_call_events = [
                event for event in events if event["event_type"] == "tool_call"
            ]
            tool_result_events = [
                event for event in events if event["event_type"] == "tool_result"
            ]
            self.assertTrue(tool_call_events)
            self.assertTrue(tool_result_events)
            self.assertEqual(
                tool_call_events[0]["payload"]["tool_call_id"],
                tool_result_events[0]["payload"]["tool_call_id"],
            )
            self.assertEqual(tool_call_events[0]["payload"]["turn_index"], 1)
            stable_text = (tmp_path / "memory" / "project-knowledge.md").read_text(
                encoding="utf-8",
            )
            self.assertIn("Knowledge from", stable_text)
            self.assertIn("auto_written_at", stable_text)
            self.assertFalse(list((tmp_path / "memory" / "inbox").glob("*.md")))

    def test_plan_validator_normalizes_and_rejects_bad_steps(self) -> None:
        plan = TaskPlan(
            objective="",
            confidence=2.0,
            steps=[
                PlanStep(
                    goal="List workspace",
                    tool_call=ToolCall("filesystem.list_dir", {}, "inspect"),
                ),
                PlanStep(
                    goal="Duplicate list",
                    tool_call=ToolCall("filesystem.list_dir", {}, "inspect again"),
                ),
                PlanStep(
                    goal="Bad tool",
                    tool_call=ToolCall("unknown.tool", {}, "bad"),
                ),
                PlanStep(
                    goal="Missing path",
                    tool_call=ToolCall("filesystem.read_file", {}, "bad"),
                ),
                PlanStep(goal="Think without tool"),
            ],
        )

        result = PlanValidator().validate(
            plan=plan,
            available_tools=["filesystem.list_dir", "filesystem.read_file"],
            max_steps=5,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.plan.objective, "Respond to user request.")
        self.assertEqual(result.plan.confidence, 1.0)
        self.assertEqual(len(result.plan.steps), 2)
        self.assertEqual(
            result.plan.steps[0].tool_call.arguments,
            {"path": "."},
        )
        self.assertEqual(result.plan.steps[1].status, "done")
        issue_codes = {issue.code for issue in result.issues}
        self.assertIn("missing_objective", issue_codes)
        self.assertIn("confidence_clamped", issue_codes)
        self.assertIn("defaulted_list_dir_path", issue_codes)
        self.assertIn("duplicate_tool_call", issue_codes)
        self.assertIn("unknown_tool", issue_codes)
        self.assertIn("missing_read_file_path", issue_codes)
        self.assertIn("non_executable_pending_step", issue_codes)

    def test_agent_persists_replayable_session_trace(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace = tmp_path / "workspace"
            workspace.mkdir()
            (workspace / "README.md").write_text("# Demo", encoding="utf-8")

            agent = AgentLoop(
                workspace=workspace,
                registry=build_default_registry(workspace),
                event_logger=EventLogger(tmp_path / "data" / "events"),
                memory_store=MarkdownMemoryStore(tmp_path / "memory"),
                use_llm=False,
            )

            response = agent.run("总结 README.md", session_id="trace-session")

            session_file = tmp_path / "data" / "sessions" / "trace-session.jsonl"
            self.assertTrue(session_file.exists())
            events = [
                json.loads(line)
                for line in session_file.read_text(encoding="utf-8").splitlines()
            ]
            event_types = [event["event_type"] for event in events]
            self.assertEqual(event_types[0], "agent_start")
            self.assertIn("message_start", event_types)
            self.assertIn("tool_call", event_types)
            self.assertIn("tool_result", event_types)
            self.assertEqual(event_types[-1], "agent_end")
            self.assertEqual({event["session_id"] for event in events}, {"trace-session"})
            self.assertEqual({event["task_id"] for event in events}, {response.task_id})
            self.assertLess(event_types.index("tool_call"), event_types.index("tool_result"))

    def test_session_cli_lists_and_shows_trace(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            recorder = SessionRecorder(root)
            recorder.append(
                AgentEvent(
                    event_type="agent_start",
                    session_id="cli-session",
                    task_id="task-1",
                    payload={},
                )
            )
            recorder.append(
                AgentEvent(
                    event_type="agent_end",
                    session_id="cli-session",
                    task_id="task-1",
                    payload={"response": "done"},
                )
            )

            listed = StringIO()
            with redirect_stdout(listed):
                cli_main(["session", "list", "--root", str(root)])

            self.assertIn("cli-session", listed.getvalue())
            self.assertIn("events=2", listed.getvalue())

            shown = StringIO()
            with redirect_stdout(shown):
                cli_main(["session", "show", "cli-session", "--root", str(root)])

            self.assertIn("agent_start", shown.getvalue())
            self.assertIn("agent_end", shown.getvalue())

    def test_unknown_prompt_still_produces_observable_tool_result(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace = tmp_path / "workspace"
            workspace.mkdir()

            agent = AgentLoop(
                workspace=workspace,
                event_logger=EventLogger(tmp_path / "events"),
                memory_store=MarkdownMemoryStore(tmp_path / "memory"),
            )

            response = agent.run("你好", session_id="fallback-session")

            self.assertEqual(response.steps, 1)
            self.assertIn("mock_os.get_app_state", response.content)
            self.assertFalse(response.memory_candidates)

    def test_skill_registry_loads_project_skills(self) -> None:
        registry = SkillRegistry("skills")
        registry.load()

        self.assertIn("file-inspection", registry.names())
        self.assertIn("markdown-memory", registry.names())
        self.assertIn("filesystem.read_file", registry.tools_for_skill("file-inspection"))

    def test_agent_context_event_includes_skills(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace = tmp_path / "workspace"
            workspace.mkdir()
            (workspace / "README.md").write_text("# Demo", encoding="utf-8")

            event_logger = EventLogger(tmp_path / "events")
            agent = AgentLoop(
                workspace=workspace,
                event_logger=event_logger,
                memory_store=MarkdownMemoryStore(tmp_path / "memory"),
                skill_registry=SkillRegistry("skills"),
            )
            agent.run("总结 README.md", session_id="skill-session")

            event_file = next((tmp_path / "events").glob("*.jsonl"))
            events = [
                json.loads(line)
                for line in event_file.read_text(encoding="utf-8").splitlines()
            ]
            context_events = [
                event for event in events if event["event_type"] == "context_built"
            ]
            self.assertTrue(context_events)
            self.assertIn("file-inspection", context_events[0]["payload"]["skills"])

    def test_memory_policy_sends_high_risk_items_to_pending(self) -> None:
        """Conflict on preference triggers pending, first occurrence goes stable."""
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = MarkdownMemoryStore(tmp_path / "memory")
            first = MemoryCandidate(
                title="User prefers concise answers",
                memory_type="preference",
                content="The user prefers concise answers.",
                source_event_id="evt-1",
                confidence=0.90,
                tags=["profile"],
            )
            # first preference → stable
            decision = MemoryPolicy().decide(first)
            self.assertEqual(decision.action, "stable")
            store.append_stable(first, decision)
            profile_text = (tmp_path / "memory" / "profile.md").read_text(encoding="utf-8")
            self.assertIn("User prefers concise answers", profile_text)

            # conflicting preference → pending
            second = MemoryCandidate(
                title="User prefers verbose answers",
                memory_type="preference",
                content="The user prefers verbose long answers.",
                source_event_id="evt-2",
                confidence=0.90,
                tags=["profile"],
            )
            decision2 = MemoryPolicy().decide(second, has_conflict=store.has_conflict(second))
            self.assertEqual(decision2.action, "pending")
            self.assertEqual(decision2.reason, "possible_conflict")

    def test_agent_can_use_llm_planner_and_responder(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace = tmp_path / "workspace"
            workspace.mkdir()
            (workspace / "README.md").write_text("# Demo", encoding="utf-8")
            fake_provider = FakeLLMProvider()

            event_logger = EventLogger(tmp_path / "events")
            agent = AgentLoop(
                workspace=workspace,
                registry=build_default_registry(workspace),
                event_logger=event_logger,
                memory_store=MarkdownMemoryStore(tmp_path / "memory"),
                llm_planner=LLMPlanner(fake_provider),  # type: ignore[arg-type]
            )
            response = agent.run("请总结项目", session_id="llm-session")

            self.assertEqual(response.steps, 1)
            self.assertIn("LLM 已读取 README.md", response.content)
            self.assertGreaterEqual(len(fake_provider.calls), 2)
            event_file = next((tmp_path / "events").glob("*.jsonl"))
            event_text = event_file.read_text(encoding="utf-8")
            self.assertIn("llm_plan_succeeded", event_text)
            self.assertIn("llm_plan_revised", event_text)
            self.assertIn('"planner": "llm"', event_text)
            self.assertIn('"objective": "Read project overview"', event_text)
            system_prompts = [call[0].content for call in fake_provider.calls]
            self.assertTrue(all("profile.md" in prompt for prompt in system_prompts))
            self.assertTrue(all("memory/" in prompt for prompt in system_prompts))

    def test_llm_planner_can_continue_after_observation(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace = tmp_path / "workspace"
            workspace.mkdir()
            (workspace / "README.md").write_text("# Demo", encoding="utf-8")

            event_logger = EventLogger(tmp_path / "events")
            agent = AgentLoop(
                workspace=workspace,
                registry=build_default_registry(workspace),
                event_logger=event_logger,
                memory_store=MarkdownMemoryStore(tmp_path / "memory"),
                llm_planner=LLMPlanner(AdaptiveFakeLLMProvider()),  # type: ignore[arg-type]
                max_steps=3,
            )
            response = agent.run("先看目录再总结项目", session_id="react-session")

            self.assertEqual(response.steps, 2)
            event_file = next((tmp_path / "events").glob("*.jsonl"))
            events = [
                json.loads(line)
                for line in event_file.read_text(encoding="utf-8").splitlines()
            ]
            plan_events = [
                event for event in events if event["event_type"] == "plan_created"
            ]
            self.assertGreaterEqual(len(plan_events), 2)
            self.assertEqual(plan_events[0]["payload"]["step_index"], 1)
            self.assertEqual(plan_events[1]["payload"]["step_index"], 2)
            self.assertEqual(
                plan_events[0]["payload"]["plan"]["steps"][0]["goal"],
                "Inspect workspace",
            )

    def test_simple_planner_recovers_from_failed_file_read(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace = tmp_path / "workspace"
            workspace.mkdir()
            (workspace / "README.md").write_text("# Demo", encoding="utf-8")

            event_logger = EventLogger(tmp_path / "events")
            agent = AgentLoop(
                workspace=workspace,
                registry=build_default_registry(workspace),
                event_logger=event_logger,
                memory_store=MarkdownMemoryStore(tmp_path / "memory"),
                use_llm=False,
                max_steps=3,
            )
            response = agent.run("请总结 MISSING.md", session_id="recover-session")

            self.assertEqual(response.steps, 2)
            self.assertIn("filesystem.read_file: error", response.content)
            self.assertIn("filesystem.list_dir: success", response.content)
            event_file = next((tmp_path / "events").glob("*.jsonl"))
            events = [
                json.loads(line)
                for line in event_file.read_text(encoding="utf-8").splitlines()
            ]
            plan_events = [
                event for event in events if event["event_type"] == "plan_created"
            ]
            self.assertGreaterEqual(len(plan_events), 2)
            self.assertEqual(
                plan_events[1]["payload"]["plan"]["steps"][0]["goal"],
                "Recover by listing workspace files.",
            )

    def test_before_tool_call_hook_can_block_tool_execution(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace = tmp_path / "workspace"
            workspace.mkdir()
            (workspace / "README.md").write_text("# Demo", encoding="utf-8")
            event_logger = EventLogger(tmp_path / "events")
            agent = AgentLoop(
                workspace=workspace,
                event_logger=event_logger,
                memory_store=MarkdownMemoryStore(tmp_path / "memory"),
                hooks=[BlockingReadHook()],
                use_llm=False,
                max_steps=1,
            )

            response = agent.run("总结 README.md", session_id="hook-block-session")

            self.assertEqual(response.steps, 1)
            self.assertIn("blocked by hook", response.content)
            event_file = next((tmp_path / "events").glob("*.jsonl"))
            events = [
                json.loads(line)
                for line in event_file.read_text(encoding="utf-8").splitlines()
            ]
            event_types = [event["event_type"] for event in events]
            self.assertIn("tool_call_blocked", event_types)
            blocked = next(event for event in events if event["event_type"] == "tool_call_blocked")
            result = next(event for event in events if event["event_type"] == "tool_result")
            self.assertEqual(
                blocked["payload"]["tool_call_id"],
                result["payload"]["tool_call_id"],
            )

    def test_after_tool_result_hook_can_patch_result(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace = tmp_path / "workspace"
            workspace.mkdir()
            event_logger = EventLogger(tmp_path / "events")
            agent = AgentLoop(
                workspace=workspace,
                event_logger=event_logger,
                memory_store=MarkdownMemoryStore(tmp_path / "memory"),
                hooks=[PatchAppStateHook()],
                use_llm=False,
            )

            agent.run("你好", session_id="hook-patch-session")

            event_file = next((tmp_path / "events").glob("*.jsonl"))
            events = [
                json.loads(line)
                for line in event_file.read_text(encoding="utf-8").splitlines()
            ]
            result = next(event for event in events if event["event_type"] == "tool_result")
            self.assertEqual(result["payload"]["output"]["active_app"], "patched-app")

    def test_filesystem_read_rejects_workspace_escape(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace = tmp_path / "workspace"
            workspace.mkdir()
            outside = tmp_path / "secret.txt"
            outside.write_text("secret", encoding="utf-8")

            result = ReadFileTool(workspace).run({"path": str(outside)})

            self.assertEqual(result.status, "error")
            self.assertIn("escapes workspace", result.error or "")

    def test_safe_shell_rejects_shell_metacharacters(self) -> None:
        with TemporaryDirectory() as tmp:
            result = SafeShellTool(tmp).run({"command": "echo ok; whoami"})

            self.assertEqual(result.status, "error")
            self.assertIn("metacharacters", result.error or "")

    def test_safe_shell_rejects_memory_files(self) -> None:
        with TemporaryDirectory() as tmp:
            tool = SafeShellTool(tmp)

            for command in (
                "Get-ChildItem memory",
                "type memory/profile.md",
                "type profile.md",
                "type .tier_index",
            ):
                with self.subTest(command=command):
                    result = tool.run({"command": command})
                    self.assertEqual(result.status, "error")
                    self.assertIn("managed by the agent", result.error or "")

    def test_safe_shell_rejects_paths_outside_workspace(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            nested = workspace / "nested"
            nested.mkdir(parents=True)
            outside = root / "outside.txt"
            outside.write_text("secret", encoding="utf-8")
            tool = SafeShellTool(str(nested), workspace=str(workspace))

            for command in (f'type "{outside}"', "Get-ChildItem ../.."):
                with self.subTest(command=command):
                    result = tool.run({"command": command})
                    self.assertEqual(result.status, "error")
                    self.assertIn("escapes workspace", result.error or "")

    def test_safe_shell_allows_paths_inside_workspace(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            target = workspace / "notes.txt"
            target.write_text("ok", encoding="utf-8")
            source = workspace / "source.txt"
            source.write_text("memory is an ordinary search term", encoding="utf-8")

            result = SafeShellTool(str(workspace)).run({"command": "type notes.txt"})
            search = SafeShellTool(str(workspace)).run(
                {"command": "rg memory source.txt"}
            )

            self.assertEqual(result.status, "success")
            self.assertIn("ok", result.output["stdout"])
            self.assertEqual(search.status, "success")
            self.assertIn("ordinary search term", search.output["stdout"])

    def test_memory_retrieval_finds_relevant_markdown_sections(self) -> None:
        with TemporaryDirectory() as tmp:
            memory_store = MarkdownMemoryStore(Path(tmp) / "memory")
            project_knowledge = Path(tmp) / "memory" / "project-knowledge.md"
            project_knowledge.write_text(
                """# Project Knowledge

## Memory Indexing

The OS Agent should retrieve relevant Markdown memory before planning.
""",
                encoding="utf-8",
            )

            matches = memory_store.retrieve("How should memory indexing work?")

            self.assertTrue(matches)
            self.assertEqual(matches[0].title, "Memory Indexing")

    def test_memory_vector_and_hybrid_retrieval_find_sections(self) -> None:
        with TemporaryDirectory() as tmp:
            memory_store = MarkdownMemoryStore(Path(tmp) / "memory")
            project_knowledge = Path(tmp) / "memory" / "project-knowledge.md"
            project_knowledge.write_text(
                """# Project Knowledge

## Vector Memory

Embedding vectors support memory retrieval and ranking.

## Workflow Notes

The planner validates tool calls before execution.
""",
                encoding="utf-8",
            )

            vector_matches = memory_store.retrieve_vector("embedding vector retrieval")
            hybrid_matches = memory_store.retrieve_hybrid("embedding vector retrieval")

            self.assertTrue(vector_matches)
            self.assertEqual(vector_matches[0].title, "Vector Memory")
            self.assertEqual(vector_matches[0].retrieval_method, "local_vector")
            self.assertTrue(hybrid_matches)
            self.assertEqual(hybrid_matches[0].title, "Vector Memory")
            self.assertIn("vector", hybrid_matches[0].retrieval_method)

    def test_memory_vector_retrieval_can_use_semantic_embedding_provider(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            memory_store = MarkdownMemoryStore(tmp_path / "memory")
            provider = FakeEmbeddingProvider()
            memory_store.vector_index = MemoryVectorIndex(
                embedding_provider=provider,  # type: ignore[arg-type]
                embedding_cache=EmbeddingCache(tmp_path / "cache" / "embeddings.json"),
            )
            project_knowledge = tmp_path / "memory" / "project-knowledge.md"
            project_knowledge.write_text(
                """# Project Knowledge

## User Response Preference

The user prefers concise answers.

## Long Form Notes

The user asked for verbose research notes.
""",
                encoding="utf-8",
            )

            matches = memory_store.retrieve_vector("brief response style")

            self.assertTrue(matches)
            self.assertEqual(matches[0].title, "User Response Preference")
            self.assertEqual(matches[0].retrieval_method, "semantic_embedding")
            self.assertGreater(provider.calls, 0)

    def test_memory_store_pages_stable_sections(self) -> None:
        with TemporaryDirectory() as tmp:
            memory_store = MarkdownMemoryStore(Path(tmp) / "memory")
            project_knowledge = Path(tmp) / "memory" / "project-knowledge.md"
            project_knowledge.write_text(
                """# Project Knowledge

## Alpha

First fact.

## Beta

Second fact.

## Gamma

Third fact.
""",
                encoding="utf-8",
            )

            first_page = memory_store.page_memory(page=1, page_size=2)
            second_page = memory_store.page_memory(page=2, page_size=2)
            filtered = memory_store.page_memory(
                page=1,
                page_size=10,
                source="project_knowledge",
            )

            self.assertEqual(first_page.total_items, 3)
            self.assertEqual(first_page.total_pages, 2)
            self.assertTrue(first_page.has_next)
            self.assertFalse(first_page.has_previous)
            self.assertEqual([item.title for item in first_page.items], ["Alpha", "Beta"])
            self.assertEqual([item.title for item in second_page.items], ["Gamma"])
            self.assertEqual(filtered.source, "project-knowledge.md")
            self.assertEqual([item.index for item in filtered.items], [1, 2, 3])

    def test_memory_store_promotes_frequently_retrieved_sections_to_hot_layer(self) -> None:
        with TemporaryDirectory() as tmp:
            memory_store = MarkdownMemoryStore(Path(tmp) / "memory")
            project_knowledge = Path(tmp) / "memory" / "project-knowledge.md"
            project_knowledge.write_text(
                """# Project Knowledge

## Alpha

Alpha fact.

## Beta

Beta fact.
""",
                encoding="utf-8",
            )

            for _ in range(5):
                memory_store.retrieve("Beta")

            hot_page = memory_store.page_memory(layer="hot")
            cold_page = memory_store.page_memory(layer="cold")
            summary = memory_store.memory_layer_summary()

            self.assertEqual([item.title for item in hot_page.items], ["Beta"])
            self.assertEqual(hot_page.items[0].access_count, 5)
            self.assertEqual([item.title for item in cold_page.items], ["Alpha"])
            self.assertEqual(summary["hot"], 1)
            self.assertEqual(summary["cold"], 1)

    def test_memory_store_priority_paging_uses_query_relevance(self) -> None:
        with TemporaryDirectory() as tmp:
            memory_store = MarkdownMemoryStore(Path(tmp) / "memory")
            project_knowledge = Path(tmp) / "memory" / "project-knowledge.md"
            project_knowledge.write_text(
                """# Project Knowledge

## Alpha

confidence: 0.55

Alpha contains planner details for the current task.

## Beta

confidence: 0.55

Beta contains unrelated historical notes.
""",
                encoding="utf-8",
            )

            for _ in range(3):
                memory_store.retrieve("Beta")

            page = memory_store.page_memory(
                page=1,
                page_size=1,
                query="planner Alpha",
                strategy="priority",
            )

            self.assertEqual(page.strategy, "priority")
            self.assertEqual([item.title for item in page.items], ["Alpha"])
            self.assertGreater(page.items[0].relevance_score, 0)
            self.assertGreater(page.items[0].priority_score, 0)

    def test_memory_store_priority_page_keeps_layer_quota(self) -> None:
        with TemporaryDirectory() as tmp:
            memory_store = MarkdownMemoryStore(Path(tmp) / "memory")
            project_knowledge = Path(tmp) / "memory" / "project-knowledge.md"
            project_knowledge.write_text(
                """# Project Knowledge

## Hot

Hot memory.

## Warm

Warm memory.

## Cold

Cold memory.
""",
                encoding="utf-8",
            )
            for _ in range(3):
                memory_store.retrieve("Hot")
            memory_store.retrieve("Warm")

            page = memory_store.page_memory(page=1, page_size=3, strategy="priority")

            self.assertEqual({item.layer for item in page.items}, {"hot", "warm", "cold"})

    def test_context_builder_uses_paged_markdown_memory(self) -> None:
        with TemporaryDirectory() as tmp:
            memory_store = MarkdownMemoryStore(Path(tmp) / "memory")
            project_knowledge = Path(tmp) / "memory" / "project-knowledge.md"
            project_knowledge.write_text(
                """# Project Knowledge

## Alpha

First fact.

## Beta

Second fact.
""",
                encoding="utf-8",
            )

            context = ContextBuilder(memory_store).build(
                ["filesystem.read_file"],
                query="Beta",
                memory_page=1,
                memory_page_size=1,
            )

            self.assertIn("# Markdown Memory Page", context)
            self.assertIn("vector", context)
            self.assertIn('page="1"', context)
            self.assertIn('strategy="priority"', context)
            self.assertIn("Beta", context)
            self.assertNotIn("First fact.", context)

    def test_memory_cli_pages_stable_memory(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "memory"
            MarkdownMemoryStore(root)
            (root / "project-knowledge.md").write_text(
                """# Project Knowledge

## Alpha

First fact.
""",
                encoding="utf-8",
            )

            output = StringIO()
            with redirect_stdout(output):
                cli_main(
                    [
                        "memory",
                        "page",
                        "--root",
                        str(root),
                        "--source",
                        "project-knowledge.md",
                    ]
                )

            self.assertIn("Memory page 1/1", output.getvalue())
            self.assertIn("[project-knowledge.md] Alpha", output.getvalue())

    def test_memory_cli_filters_memory_layers(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "memory"
            memory_store = MarkdownMemoryStore(root)
            (root / "project-knowledge.md").write_text(
                """# Project Knowledge

## Alpha

Alpha fact.

## Beta

Beta fact.
""",
                encoding="utf-8",
            )
            for _ in range(5):
                memory_store.retrieve("Beta")

            output = StringIO()
            with redirect_stdout(output):
                cli_main(
                    [
                        "memory",
                        "page",
                        "--root",
                        str(root),
                        "--layer",
                        "hot",
                    ]
                )

            self.assertIn("layer=hot", output.getvalue())
            self.assertIn("Beta", output.getvalue())
            self.assertNotIn("Alpha", output.getvalue())

    def test_memory_cli_searches_stable_memory(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "memory"
            MarkdownMemoryStore(root)
            (root / "project-knowledge.md").write_text(
                """# Project Knowledge

## Vector Memory

Embedding vectors support memory retrieval.
""",
                encoding="utf-8",
            )

            output = StringIO()
            with redirect_stdout(output):
                cli_main(
                    [
                        "memory",
                        "search",
                        "embedding retrieval",
                        "--root",
                        str(root),
                        "--mode",
                        "hybrid",
                    ]
                )

            self.assertIn("Vector Memory", output.getvalue())
            self.assertIn("score=", output.getvalue())

    def test_memory_module_supports_ltm_remember_update_forget(self) -> None:
        with TemporaryDirectory() as tmp:
            module = AgentMemoryModule(Path(tmp) / "memory")

            remembered = module.remember(
                RememberRequest(
                    scope="ltm",
                    content="User prefers concise Chinese answers.",
                    entity="user",
                    attribute="reply_style",
                    value="concise",
                    tags=["preference"],
                )
            )
            self.assertTrue(remembered.ok)
            self.assertEqual(remembered.records[0].entity, "user")

            updated = module.update(
                UpdateRequest(
                    scope="ltm",
                    entity="user",
                    attribute="reply_style",
                    value="concise with examples",
                    content="User prefers concise answers with examples.",
                    tags=["preference"],
                )
            )
            self.assertTrue(updated.ok)
            self.assertEqual(updated.records[0].value, "concise with examples")
            history = module.archival.get_fact_history("user", "reply_style")
            self.assertEqual(history[0].superseded_by, updated.records[0].memory_id)

            recalled = module.recall(RecallRequest(query="concise examples", scope="ltm"))
            self.assertTrue(recalled.ok)
            self.assertEqual(recalled.records[0].attribute, "reply_style")

            forgotten = module.forget(
                ForgetRequest(scope="ltm", entity="user", attribute="reply_style")
            )
            self.assertTrue(forgotten.ok)
            self.assertEqual(forgotten.records[0].metadata["status"], "deleted")

    def test_memory_module_supports_stm_remember_update_forget(self) -> None:
        with TemporaryDirectory() as tmp:
            module = AgentMemoryModule(Path(tmp) / "memory")

            remembered = module.remember(
                RememberRequest(
                    scope="stm",
                    memory_id="stm-note",
                    content="Temporary planner note about README.",
                    tags=["scratchpad"],
                )
            )
            self.assertTrue(remembered.ok)
            self.assertEqual(remembered.records[0].memory_id, "stm-note")

            updated = module.update(
                UpdateRequest(
                    scope="stm",
                    memory_id="stm-note",
                    content="Temporary planner note about README and tests.",
                )
            )
            self.assertTrue(updated.ok)

            recalled = module.recall(RecallRequest(query="README tests", scope="stm"))
            self.assertTrue(recalled.ok)
            self.assertEqual(recalled.records[0].memory_id, "stm-note")

            forgotten = module.forget(ForgetRequest(scope="stm", memory_id="stm-note"))
            self.assertTrue(forgotten.ok)
            self.assertFalse(module.recall(RecallRequest(query="README", scope="stm")).ok)

    def test_memory_tools_are_registered_and_callable(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            memory_root = Path(tmp) / "memory"
            registry = build_default_registry(workspace, memory_root=memory_root)

            self.assertIn("memory.remember", registry.names())
            remember = registry.execute(
                ToolCall(
                    "memory.remember",
                    {
                        "scope": "ltm",
                        "content": "Project uses pytest-compatible unittest.",
                        "entity": "project",
                        "attribute": "test_runner",
                        "value": "python -m unittest -q",
                    },
                )
            )
            self.assertTrue(remember.ok)
            self.assertEqual(remember.output["action"], "remember")

            update = registry.execute(
                ToolCall(
                    "memory.update",
                    {
                        "scope": "ltm",
                        "entity": "project",
                        "attribute": "test_runner",
                        "value": "python -m unittest -q",
                        "content": "Tests run with unittest.",
                    },
                )
            )
            self.assertTrue(update.ok)

            recall = registry.execute(
                ToolCall(
                    "memory.recall",
                    {"scope": "ltm", "query": "unittest", "limit": 3},
                )
            )
            self.assertTrue(recall.ok)
            self.assertTrue(recall.output["records"])

            forget = registry.execute(
                ToolCall(
                    "memory.forget",
                    {
                        "scope": "ltm",
                        "entity": "project",
                        "attribute": "test_runner",
                    },
                )
            )
            self.assertTrue(forget.ok)

    def test_mem0_pipeline_backfill_uses_legacy_memory_files_and_links_versions(self) -> None:
        with TemporaryDirectory() as tmp:
            memory_root = Path(tmp) / "memory"
            legacy = MarkdownMemoryStore(memory_root)
            core = CoreMemoryStore(memory_root)
            archival = ArchivalMemoryStore(memory_root)
            pipeline = ExtractUpdatePipeline(core, archival, legacy)

            first = FactCard(
                entity="project",
                attribute="Shell执行",
                value="echo hello: hello",
                evidence="执行命令 echo hello",
                source_session_id="s",
                source_event_id="e1",
                confidence=0.55,
                tags=["tool_trace", "shell"],
            )
            second = FactCard(
                entity="project",
                attribute="Shell执行",
                value="echo hello: hello again",
                evidence="执行命令 echo hello",
                source_session_id="s",
                source_event_id="e2",
                confidence=0.55,
                tags=["tool_trace", "shell"],
            )

            pipeline.process_session([], session_id="s")
            created = pipeline._merge_into_core(first)
            updated = pipeline._merge_into_core(second)

            self.assertNotEqual(created.fact_id, updated.fact_id)
            history = archival.get_fact_history("project", "Shell执行")
            self.assertEqual(history[0].superseded_by, updated.fact_id)
            self.assertTrue((memory_root / "tool-history.md").exists())
            self.assertIn("Shell执行", (memory_root / "tool-history.md").read_text(encoding="utf-8"))
            self.assertFalse((memory_root / "tool_trace").exists())

    def test_core_markdown_mirror_replaces_versions_and_removes_deleted_facts(self) -> None:
        with TemporaryDirectory() as tmp:
            memory_root = Path(tmp) / "memory"
            markdown = MarkdownMemoryStore(memory_root)
            core = CoreMemoryStore(memory_root)
            archival = ArchivalMemoryStore(memory_root)
            pipeline = ExtractUpdatePipeline(core, archival, markdown)

            first = FactCard(
                entity="project",
                attribute="test_runner",
                value="pytest",
                evidence="项目使用 pytest",
                source_session_id="s",
                source_event_id="e1",
                tags=["knowledge"],
            )
            second = FactCard(
                entity="project",
                attribute="test_runner",
                value="unittest",
                evidence="项目改用 unittest",
                source_session_id="s",
                source_event_id="e2",
                tags=["preference"],
            )

            created = pipeline._merge_into_core(first)
            mirrored = (memory_root / "project-knowledge.md").read_text(encoding="utf-8")
            self.assertIn("Value: pytest", mirrored)
            self.assertIn(f"core_fact_id: {created.fact_id}", mirrored)

            updated = pipeline._merge_into_core(second)
            knowledge_mirror = (memory_root / "project-knowledge.md").read_text(encoding="utf-8")
            profile_mirror = (memory_root / "profile.md").read_text(encoding="utf-8")
            self.assertNotIn("Value: pytest", knowledge_mirror)
            self.assertIn("Value: unittest", profile_mirror)
            self.assertIn(f"version: {updated.version}", profile_mirror)
            self.assertEqual(profile_mirror.count("core_key: project|test_runner"), 1)

            removed = core.remove("project", "test_runner")
            self.assertIsNotNone(removed)
            markdown.remove_core_fact(removed)  # type: ignore[arg-type]
            mirrored = (memory_root / "project-knowledge.md").read_text(encoding="utf-8")
            self.assertNotIn("core_key: project|test_runner", mirrored)

    def test_memory_module_keeps_markdown_mirror_in_sync_for_mutations(self) -> None:
        with TemporaryDirectory() as tmp:
            memory_root = Path(tmp) / "memory"
            markdown = MarkdownMemoryStore(memory_root)
            module = AgentMemoryModule(memory_root, markdown_store=markdown)

            module.remember(
                RememberRequest(
                    scope="ltm",
                    content="项目使用 pytest",
                    entity="project",
                    attribute="test_runner",
                    value="pytest",
                )
            )
            module.update(
                UpdateRequest(
                    scope="ltm",
                    content="项目改用 unittest",
                    entity="project",
                    attribute="test_runner",
                    value="unittest",
                )
            )
            mirrored = (memory_root / "project-knowledge.md").read_text(encoding="utf-8")
            self.assertNotIn("Value: pytest", mirrored)
            self.assertIn("Value: unittest", mirrored)

            module.forget(
                ForgetRequest(
                    scope="ltm",
                    entity="project",
                    attribute="test_runner",
                )
            )
            mirrored = (memory_root / "project-knowledge.md").read_text(encoding="utf-8")
            self.assertNotIn("core_key: project|test_runner", mirrored)

    def test_core_search_uses_tunable_default_limit(self) -> None:
        with TemporaryDirectory() as tmp:
            cfg = MemoryConfig(Path(tmp) / "memory_config.json")
            cfg.set_value("core_search_top_k", 1)
            store = CoreMemoryStore(Path(tmp) / "memory", config=cfg)
            store.set("project", "first", "python unittest")
            store.set("project", "second", "python memory")

            results = store.search("python", top_k=0)

            self.assertEqual(len(results), 1)
            self.assertGreater(results[0].search_score, 0)

    def test_pipeline_tuning_skips_memory_files_and_uses_per_file_attributes(self) -> None:
        with TemporaryDirectory() as tmp:
            cfg = MemoryConfig(Path(tmp) / "memory_config.json")
            cfg.set_value("pipeline_skip_memory_file_facts", True)
            cfg.set_value("pipeline_per_file_attributes", True)
            memory_root = Path(tmp) / "memory"
            pipeline = ExtractUpdatePipeline(
                CoreMemoryStore(memory_root, config=cfg),
                ArchivalMemoryStore(memory_root, config=cfg),
                config=cfg,
            )
            events = [
                {
                    "event_type": "tool_result",
                    "event_id": "e1",
                    "payload": {
                        "tool_name": "filesystem.read_file",
                        "status": "success",
                        "output": {"path": str(Path(tmp) / "README.md"), "content": "# Demo"},
                    },
                },
                {
                    "event_type": "tool_result",
                    "event_id": "e2",
                    "payload": {
                        "tool_name": "filesystem.read_file",
                        "status": "success",
                        "output": {
                            "path": str(Path(tmp) / "memory" / "profile.md"),
                            "content": "# Profile",
                        },
                    },
                },
            ]

            facts = pipeline._extract_facts(events, session_id="s")

            self.assertEqual([fact.attribute for fact in facts], ["文件内容:README.md"])

    def test_top_level_help_lists_mem0_commands(self) -> None:
        output = StringIO()
        with self.assertRaises(SystemExit) as raised:
            with redirect_stdout(output):
                cli_main(["--help"])

        self.assertEqual(raised.exception.code, 0)
        text = output.getvalue()
        self.assertIn("core      Manage structured Core memory.", text)
        self.assertIn("pipeline  Run the Extract-Update memory pipeline.", text)

    def test_memory_hook_skips_duplicate_candidates(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace = tmp_path / "workspace"
            workspace.mkdir()
            (workspace / "README.md").write_text("# Demo", encoding="utf-8")
            memory_store = MarkdownMemoryStore(tmp_path / "memory")
            agent = AgentLoop(
                workspace=workspace,
                registry=build_default_registry(workspace),
                event_logger=EventLogger(tmp_path / "events"),
                memory_store=memory_store,
                use_mem0=False,
            )

            first = agent.run("总结 README.md", session_id="dup-session")
            second = agent.run("总结 README.md", session_id="dup-session")

            self.assertTrue(first.memory_candidates)
            self.assertFalse(second.memory_candidates)

    def test_memory_maintenance_removes_duplicates_and_archives_overflow(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            MarkdownMemoryStore(tmp_path / "memory")
            project_knowledge = tmp_path / "memory" / "project-knowledge.md"
            project_knowledge.write_text(
                """# Project Knowledge

## Alpha

Same fact.

## Alpha

Same fact.

## Beta

Another fact.
""",
                encoding="utf-8",
            )

            report = MemoryMaintenanceService(tmp_path / "memory").maintain(
                max_sections_per_file=1
            )
            stable = project_knowledge.read_text(encoding="utf-8")

            self.assertEqual(report.removed_duplicates, 1)
            self.assertEqual(report.archived_sections, 1)
            self.assertIn("## Beta", stable)
            self.assertNotIn("## Alpha", stable)


if __name__ == "__main__":
    unittest.main()
