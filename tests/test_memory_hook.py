"""Tests for the expanded MemoryHook covering all 6 extraction rules."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from os_agent.events.schemas import AgentEvent
from os_agent.memory.hook import MemoryHook
from os_agent.memory.markdown_store import MarkdownMemoryStore
from os_agent.tools.builtins import build_default_registry


def _tool_result_event(
    tool_name: str,
    status: str,
    output: dict | None = None,
    error: str | None = None,
    input_args: dict | None = None,
    session_id: str = "test-session",
) -> AgentEvent:
    """Build a minimal tool_result AgentEvent for hook testing."""
    return AgentEvent(
        event_type="tool_result",
        session_id=session_id,
        task_id="task-test",
        payload={
            "tool_name": tool_name,
            "status": status,
            "output": output or {},
            "error": error,
            "input": input_args or {},
            "tool_call_id": "call-test",
            "turn_index": 1,
        },
    )


def _user_message_event(
    content: str,
    session_id: str = "test-session",
) -> AgentEvent:
    """Build a minimal user message_start AgentEvent for hook testing."""
    return AgentEvent(
        event_type="message_start",
        session_id=session_id,
        task_id="task-test",
        payload={"role": "user", "content": content},
    )


class MemoryHookUnitTests(unittest.TestCase):
    """Test each extraction rule of MemoryHook in isolation."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self._tmp_path = Path(self._tmp.name)
        self.store = MarkdownMemoryStore(self._tmp_path / "memory")
        self.hook = MemoryHook(self.store)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # ------------------------------------------------------------------
    # Rule 1: read_file success → knowledge
    # ------------------------------------------------------------------
    def test_read_file_success_produces_knowledge(self) -> None:
        event = _tool_result_event(
            tool_name="filesystem.read_file",
            status="success",
            output={
                "path": "README.md",
                "content": "# Project Title\n\nThis is a demo.\n\nLine three.",
                "truncated": False,
                "chars": 42,
            },
        )
        candidates = self.hook.observe(event)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].memory_type, "knowledge")
        self.assertEqual(candidates[0].confidence, 0.55)
        self.assertIn("README.md", candidates[0].title)
        self.assertIn("# Project Title", candidates[0].content)
        self.assertEqual(candidates[0].tags, ["tool_trace", "file"])

    def test_read_file_success_writes_to_project_knowledge(self) -> None:
        event = _tool_result_event(
            tool_name="filesystem.read_file",
            status="success",
            output={"path": "docs.md", "content": "# Docs\n\nContent here."},
        )
        self.hook.observe(event)

        knowledge_file = self._tmp_path / "memory" / "project-knowledge.md"
        text = knowledge_file.read_text(encoding="utf-8")
        self.assertIn("Knowledge from docs.md", text)
        self.assertIn("source_event_id:", text)
        self.assertIn("confidence: 0.55", text)
        self.assertIn("policy_reason: low_risk_stable_write", text)

    def test_read_file_empty_content_produces_no_candidate(self) -> None:
        event = _tool_result_event(
            tool_name="filesystem.read_file",
            status="success",
            output={"path": "empty.md", "content": ""},
        )
        candidates = self.hook.observe(event)
        self.assertEqual(candidates, [])

    # ------------------------------------------------------------------
    # Rule 2: list_dir success → knowledge
    # ------------------------------------------------------------------
    def test_list_dir_success_produces_knowledge(self) -> None:
        event = _tool_result_event(
            tool_name="filesystem.list_dir",
            status="success",
            output={
                "path": ".",
                "entries": [
                    {"name": "README.md", "path": "README.md", "type": "file"},
                    {"name": "src", "path": "src", "type": "directory"},
                    {"name": "tests", "path": "tests", "type": "directory"},
                ],
            },
        )
        candidates = self.hook.observe(event)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].memory_type, "knowledge")
        self.assertEqual(candidates[0].confidence, 0.50)
        self.assertIn("Directory listing", candidates[0].title)
        self.assertIn("README.md", candidates[0].content)
        self.assertIn("src", candidates[0].content)
        self.assertEqual(candidates[0].tags, ["tool_trace", "directory"])

    def test_list_dir_empty_entries_produces_no_candidate(self) -> None:
        event = _tool_result_event(
            tool_name="filesystem.list_dir",
            status="success",
            output={"path": ".", "entries": []},
        )
        candidates = self.hook.observe(event)
        self.assertEqual(candidates, [])

    def test_list_dir_truncates_to_20_entries(self) -> None:
        entries = [
            {"name": f"file_{i:02d}.txt", "path": f"file_{i:02d}.txt", "type": "file"}
            for i in range(25)
        ]
        event = _tool_result_event(
            tool_name="filesystem.list_dir",
            status="success",
            output={"path": ".", "entries": entries},
        )
        candidates = self.hook.observe(event)
        self.assertEqual(len(candidates), 1)
        # file_00 .. file_19 should appear (20 names), file_20..24 should not
        self.assertIn("file_00.txt", candidates[0].content)
        self.assertIn("file_19.txt", candidates[0].content)
        self.assertNotIn("file_20.txt", candidates[0].content)

    # ------------------------------------------------------------------
    # Rule 3: shell.run_safe success → tool_history
    # ------------------------------------------------------------------
    def test_shell_success_produces_tool_history(self) -> None:
        event = _tool_result_event(
            tool_name="shell.run_safe",
            status="success",
            output={
                "returncode": 0,
                "stdout": "hello world",
                "stderr": "",
            },
            input_args={"command": "echo hello"},
        )
        candidates = self.hook.observe(event)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].memory_type, "tool_history")
        self.assertEqual(candidates[0].confidence, 0.55)
        self.assertIn("echo hello", candidates[0].content)
        self.assertIn("rc=0", candidates[0].content)
        self.assertEqual(candidates[0].tags, ["tool_trace", "shell"])

    def test_shell_success_writes_to_tool_history_file(self) -> None:
        event = _tool_result_event(
            tool_name="shell.run_safe",
            status="success",
            output={"returncode": 0, "stdout": "ok", "stderr": ""},
            input_args={"command": "ls"},
        )
        self.hook.observe(event)

        history_file = self._tmp_path / "memory" / "tool-history.md"
        text = history_file.read_text(encoding="utf-8")
        self.assertIn("Shell:", text)
        self.assertIn("ls", text)
        self.assertIn("policy_reason: low_risk_stable_write", text)

    def test_shell_success_includes_stderr(self) -> None:
        event = _tool_result_event(
            tool_name="shell.run_safe",
            status="success",
            output={
                "returncode": 1,
                "stdout": "",
                "stderr": "command not found",
            },
            input_args={"command": "nonexistent"},
        )
        candidates = self.hook.observe(event)
        self.assertIn("command not found", candidates[0].content)

    def test_shell_with_long_command_truncates_title(self) -> None:
        long_cmd = "echo " + "x" * 100
        event = _tool_result_event(
            tool_name="shell.run_safe",
            status="success",
            output={"returncode": 0, "stdout": "x" * 100, "stderr": ""},
            input_args={"command": long_cmd},
        )
        candidates = self.hook.observe(event)
        self.assertLessEqual(len(candidates[0].title), 60 + len("Shell: "))

    # ------------------------------------------------------------------
    # Rule 4: any tool failure → tool_history
    # ------------------------------------------------------------------
    def test_tool_failure_produces_tool_history(self) -> None:
        event = _tool_result_event(
            tool_name="filesystem.read_file",
            status="error",
            error="FileNotFoundError: no such file",
        )
        candidates = self.hook.observe(event)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].memory_type, "tool_history")
        self.assertEqual(candidates[0].confidence, 0.60)
        self.assertIn("Tool failure", candidates[0].title)
        self.assertIn("failed", candidates[0].content)
        self.assertIn("FileNotFoundError", candidates[0].content)
        self.assertEqual(candidates[0].tags, ["tool_trace", "failure"])

    def test_tool_failure_higher_confidence_than_success(self) -> None:
        """Failures (0.60) are more worth remembering than routine successes (0.50-0.55)."""
        fail_event = _tool_result_event(
            tool_name="filesystem.list_dir",
            status="error",
            error="permission denied",
        )
        fail_candidates = self.hook.observe(fail_event)

        success_event = _tool_result_event(
            tool_name="filesystem.list_dir",
            status="success",
            output={"path": ".", "entries": [{"name": "x", "path": "x", "type": "file"}]},
        )
        success_candidates = self.hook.observe(success_event)

        failure_conf = fail_candidates[0].confidence
        success_conf = success_candidates[0].confidence
        self.assertGreater(failure_conf, success_conf)

    def test_tool_failure_without_error_string_still_works(self) -> None:
        event = _tool_result_event(
            tool_name="mock_os.get_app_state",
            status="error",
            error=None,
        )
        candidates = self.hook.observe(event)
        self.assertEqual(len(candidates), 1)
        self.assertIn("unknown error", candidates[0].content)

    # ------------------------------------------------------------------
    # Rule 5: user message → preference detection
    # ------------------------------------------------------------------
    def test_user_preference_keywords_detected(self) -> None:
        cases = [
            "我习惯用中文回复",
            "我喜欢先看目录再读文件",
            "我一般用 PowerShell",
            "我通常先检查 workspace",
            "I prefer concise answers",
            "I always want to list files first",
            "我总是想先看看目录结构",
        ]
        for text in cases:
            with self.subTest(input=text):
                event = _user_message_event(text)
                candidates = self.hook.observe(event)
                self.assertEqual(len(candidates), 1, f"Failed for: {text}")
                self.assertEqual(candidates[0].memory_type, "preference")

    def test_preference_first_occurrence_goes_to_stable(self) -> None:
        """First preference with no conflict goes to stable profile.md."""
        event = _user_message_event("我习惯用中文回复")
        self.hook.observe(event)

        # should be in profile.md, not in pending
        profile_text = (self._tmp_path / "memory" / "profile.md").read_text(encoding="utf-8")
        self.assertIn("User preference: 我习惯", profile_text)
        self.assertIn("low_risk_stable_write", profile_text)
        pending_files = list(
            (self._tmp_path / "memory" / "pending-confirmation").glob("*.md")
        )
        self.assertEqual(pending_files, [])

    def test_preference_conflict_goes_to_pending(self) -> None:
        """A contradictory preference triggers conflict → pending."""
        # first preference goes to stable
        self.hook.observe(_user_message_event("我习惯用中文回复"))
        # contradictory preference → conflict → pending
        self.hook.observe(_user_message_event("我习惯用英文回复"))

        pending_files = list(
            (self._tmp_path / "memory" / "pending-confirmation").glob("*.md")
        )
        self.assertTrue(pending_files)
        pending_text = pending_files[0].read_text(encoding="utf-8")
        self.assertIn("possible_conflict", pending_text)
        self.assertIn("status: pending_confirmation", pending_text)

    # ------------------------------------------------------------------
    # Rule 6: user message → safety detection
    # ------------------------------------------------------------------
    def test_user_safety_keywords_detected(self) -> None:
        cases = [
            "不要删除文件",
            "千万别改注册表",
            "禁止执行危险命令",
            "don't run rm -rf",
            "never execute as admin",
            "避免修改系统配置",
            "当心覆盖已有文件",
        ]
        for text in cases:
            with self.subTest(input=text):
                event = _user_message_event(text)
                candidates = self.hook.observe(event)
                self.assertEqual(len(candidates), 1, f"Failed for: {text}")
                self.assertEqual(candidates[0].memory_type, "safety")

    def test_safety_first_occurrence_goes_to_stable(self) -> None:
        """First safety constraint with no conflict goes to stable safety.md."""
        event = _user_message_event("不要执行 rm -rf")
        self.hook.observe(event)

        safety_text = (self._tmp_path / "memory" / "safety.md").read_text(encoding="utf-8")
        self.assertIn("Safety constraint: 不要", safety_text)
        self.assertIn("low_risk_stable_write", safety_text)

    def test_preference_and_safety_both_detected_in_same_message(self) -> None:
        """A message containing both preference and safety keywords produces TWO candidates."""
        event = _user_message_event("我喜欢用中文，不要删我文件")
        candidates = self.hook.observe(event)
        self.assertEqual(len(candidates), 2)
        types = {c.memory_type for c in candidates}
        self.assertEqual(types, {"preference", "safety"})

    # ------------------------------------------------------------------
    # Non-trigger cases (no false positives)
    # ------------------------------------------------------------------
    def test_normal_input_does_not_trigger_preference(self) -> None:
        cases = [
            "帮我总结 README.md",
            "列出目录下的所有文件",
            "list files in current directory",
            "运行测试",
            "",
        ]
        for text in cases:
            with self.subTest(input=text):
                event = _user_message_event(text)
                candidates = self.hook.observe(event)
                self.assertEqual(candidates, [], f"Unexpected candidates for: {text}")

    def test_normal_input_does_not_trigger_safety(self) -> None:
        """Words like '要' alone should NOT trigger safety false positive."""
        event = _user_message_event("我要查看文件列表")
        candidates = self.hook.observe(event)
        # "我要" does not match _SAFETY_CN which looks for "不要" (two specific chars)
        safety_candidates = [c for c in candidates if c.memory_type == "safety"]
        self.assertEqual(safety_candidates, [])

    # ------------------------------------------------------------------
    # Event type filtering
    # ------------------------------------------------------------------
    def test_observe_ignores_non_relevant_events(self) -> None:
        events = [
            AgentEvent("agent_start", "s1", "t1", {}),
            AgentEvent("turn_start", "s1", "t1", {"turn_index": 1}),
            AgentEvent("plan_created", "s1", "t1", {}),
            AgentEvent("message_start", "s1", "t1", {"role": "assistant"}),
            AgentEvent("message_end", "s1", "t1", {"role": "user"}),
        ]
        for event in events:
            with self.subTest(event_type=event.event_type):
                candidates = self.hook.observe(event)
                self.assertEqual(candidates, [])

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------
    def test_duplicate_candidate_is_skipped(self) -> None:
        event = _tool_result_event(
            tool_name="filesystem.read_file",
            status="success",
            output={"path": "dup.md", "content": "# Once"},
        )
        first = self.hook.observe(event)
        self.assertEqual(len(first), 1)

        # Same event again → skipped
        second = self.hook.observe(event)
        self.assertEqual(second, [])

    # ------------------------------------------------------------------
    # Full pipeline: event → candidate → policy → file
    # ------------------------------------------------------------------
    def test_all_stable_types_land_in_correct_files(self) -> None:
        """Verify knowledge goes to project-knowledge.md, tool_history to tool-history.md."""
        # knowledge
        self.hook.observe(
            _tool_result_event(
                "filesystem.read_file",
                "success",
                output={"path": "f.md", "content": "# F"},
            )
        )
        # tool_history (failure)
        self.hook.observe(
            _tool_result_event(
                "filesystem.list_dir",
                "error",
                error="denied",
            )
        )

        knowledge_text = (
            (self._tmp_path / "memory" / "project-knowledge.md")
            .read_text(encoding="utf-8")
        )
        history_text = (
            (self._tmp_path / "memory" / "tool-history.md")
            .read_text(encoding="utf-8")
        )

        self.assertIn("Knowledge from f.md", knowledge_text)
        self.assertIn("Tool failure", history_text)
        self.assertNotIn("Knowledge from f.md", history_text)
        self.assertNotIn("Tool failure", knowledge_text)

    def test_summarize_text_truncates_long_input(self) -> None:
        long_content = "\n".join(f"Line {i}" for i in range(100))
        event = _tool_result_event(
            tool_name="filesystem.read_file",
            status="success",
            output={"path": "long.md", "content": long_content},
        )
        candidates = self.hook.observe(event)
        # Summary should be max 500 chars (3 lines joined)
        self.assertLessEqual(
            len(candidates[0].content.split("Summary: ")[1]),
            500,
        )


class MemoryHookIntegrationTests(unittest.TestCase):
    """End-to-end tests using the real ToolRegistry."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self._tmp_path = Path(self._tmp.name)
        self.workspace = self._tmp_path / "workspace"
        self.workspace.mkdir()
        self.store = MarkdownMemoryStore(self._tmp_path / "memory")
        self.hook = MemoryHook(self.store)
        self.registry = build_default_registry(self.workspace)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_read_file_hook_with_real_tool(self) -> None:
        """Hook extracts knowledge from a real ReadFileTool result."""
        (self.workspace / "hello.txt").write_text("Hello\nWorld\n!", encoding="utf-8")
        result = self.registry.execute(
            type("ToolCall", (), {"tool_name": "filesystem.read_file", "arguments": {"path": "hello.txt"}})()
        )

        event = AgentEvent(
            event_type="tool_result",
            session_id="s",
            task_id="t",
            payload={
                "tool_name": "filesystem.read_file",
                "status": result.status,
                "output": result.output,
                "error": result.error,
                "tool_call_id": "c1",
                "turn_index": 1,
            },
        )
        candidates = self.hook.observe(event)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].memory_type, "knowledge")
        self.assertIn("hello.txt", candidates[0].title)
        self.assertIn("Hello World !", candidates[0].content)

    def test_shell_hook_with_real_tool(self) -> None:
        """Hook extracts tool_history from a real SafeShellTool result."""
        result = self.registry.execute(
            type("ToolCall", (), {"tool_name": "shell.run_safe", "arguments": {"command": "echo hello"}})()
        )
        event = AgentEvent(
            event_type="tool_result",
            session_id="s",
            task_id="t",
            payload={
                "tool_name": "shell.run_safe",
                "status": result.status,
                "output": result.output,
                "input": {"command": "echo hello"},
                "error": result.error,
                "tool_call_id": "c2",
                "turn_index": 1,
            },
        )
        candidates = self.hook.observe(event)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].memory_type, "tool_history")
        self.assertIn("echo hello", candidates[0].content)

    def test_blocked_tool_produces_failure_memory(self) -> None:
        """A shell command outside the allowlist should produce a failure memory."""
        result = self.registry.execute(
            type("ToolCall", (), {"tool_name": "shell.run_safe", "arguments": {"command": "rm -rf /"}})()
        )
        event = AgentEvent(
            event_type="tool_result",
            session_id="s",
            task_id="t",
            payload={
                "tool_name": "shell.run_safe",
                "status": result.status,
                "output": result.output,
                "input": {"command": "rm -rf /"},
                "error": result.error,
                "tool_call_id": "c3",
                "turn_index": 1,
            },
        )
        candidates = self.hook.observe(event)
        self.assertTrue(any(c.memory_type == "tool_history" and "failure" in c.tags for c in candidates))


if __name__ == "__main__":
    unittest.main()
