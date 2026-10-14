import asyncio
import base64
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mcp import Client
from mcp.types import ImageContent

from os_agent.mcp_server import (
    OSAgentMCPConfig,
    OSAgentMCPService,
    create_mcp_server,
)


class OSAgentMCPServiceTests(unittest.TestCase):
    def test_config_fixes_roots_and_validates_limits(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            config = OSAgentMCPConfig.from_values(
                workspace=workspace,
                memory_root=Path(tmp) / "memory",
                max_steps=3,
            )

            self.assertEqual(config.workspace, workspace.resolve())
            self.assertEqual(config.memory_root, (Path(tmp) / "memory").resolve())
            self.assertEqual(config.max_steps, 3)
            with self.assertRaisesRegex(ValueError, "max_steps"):
                OSAgentMCPConfig.from_values(workspace=workspace, max_steps=0)

    def test_capabilities_do_not_expose_embedding_credentials(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            config = OSAgentMCPConfig.from_values(
                workspace=workspace,
                memory_root=Path(tmp) / "memory",
            )
            with patch.dict(
                os.environ,
                {
                    "OS_AGENT_EMBEDDING_BACKEND": "openai",
                    "OS_AGENT_EMBEDDING_MODEL": "example-multilingual-model",
                    "OS_AGENT_EMBEDDING_API_KEY": "must-not-leak",
                },
                clear=True,
            ):
                capabilities = OSAgentMCPService(config).capabilities()

            self.assertEqual(
                capabilities["embedding"]["model"],
                "example-multilingual-model",
            )
            self.assertNotIn("api_key", capabilities["embedding"])
            self.assertNotIn("must-not-leak", repr(capabilities))
            self.assertFalse(capabilities["explicit_memory_mutations"])
            self.assertFalse(capabilities["workspace_file_write"])
            self.assertEqual(capabilities["file_access"]["mode"], "read_only")
            self.assertTrue(capabilities["agent_run_persists_runtime_state"])
            self.assertEqual(len(capabilities["tool_names"]), 9)
            self.assertEqual(len(capabilities["memory_tool_names"]), 4)
            self.assertTrue(capabilities["plugins"]["marketplace"])
            self.assertFalse(capabilities["plugins"]["installation_enabled"])
            self.assertTrue(capabilities["mcp"]["enabled"])
            self.assertTrue(capabilities["computer_use"]["qianxin_browser"])
            self.assertTrue(capabilities["computer_use"]["desktop"])
            self.assertFalse(capabilities["computer_use"]["enabled"])
            self.assertEqual(capabilities["computer_use"]["mode"], "allowlist")

    def test_mutations_are_denied_by_default(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            service = OSAgentMCPService(
                OSAgentMCPConfig.from_values(
                    workspace=workspace,
                    memory_root=Path(tmp) / "memory",
                )
            )

            with self.assertRaises(PermissionError):
                service.remember(content="Keep this general project decision.")

    def test_enabled_mutation_can_be_recalled_with_query_expansion(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            service = OSAgentMCPService(
                OSAgentMCPConfig.from_values(
                    workspace=workspace,
                    memory_root=Path(tmp) / "memory",
                    allow_memory_mutations=True,
                )
            )

            stored = service.remember(
                content="The project release channel is stable.",
                entity="project",
                attribute="release_channel",
                value="stable",
            )
            recalled = service.recall(
                query="Which release channel is used?",
                entity="project",
                query_variants=["project release track"],
                subqueries=["release channel setting"],
            )

            self.assertTrue(stored["ok"])
            self.assertTrue(recalled["ok"])
            self.assertEqual(recalled["records"][0]["value"], "stable")

    def test_workspace_write_is_denied_by_default(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            service = OSAgentMCPService(
                OSAgentMCPConfig.from_values(
                    workspace=workspace,
                    memory_root=Path(tmp) / "memory",
                )
            )

            with self.assertRaises(PermissionError):
                service.write_file(path="note.txt", content="blocked")

    def test_workspace_write_modes_and_boundaries(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            service = OSAgentMCPService(
                OSAgentMCPConfig.from_values(
                    workspace=workspace,
                    memory_root=Path(tmp) / "memory",
                    allow_workspace_mutations=True,
                )
            )

            created = service.write_file(
                path="notes/result.txt",
                content="first",
                create_parents=True,
            )
            service.write_file(path="notes/result.txt", content="+second", mode="append")
            self.assertEqual(
                (workspace / "notes" / "result.txt").read_text(encoding="utf-8"),
                "first+second",
            )
            overwritten = service.write_file(
                path="notes/result.txt",
                content="final",
                mode="overwrite",
            )

            self.assertTrue(created["ok"])
            self.assertEqual(created["path"], "notes/result.txt")
            self.assertEqual(overwritten["bytes_written"], 5)
            self.assertEqual(
                (workspace / "notes" / "result.txt").read_text(encoding="utf-8"),
                "final",
            )
            with self.assertRaises(PermissionError):
                service.write_file(path="../outside.txt", content="escape")
            with self.assertRaises(FileExistsError):
                service.write_file(path="notes/result.txt", content="replace")

    def test_file_access_can_switch_between_read_only_and_workspace(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            service = OSAgentMCPService(
                OSAgentMCPConfig.from_values(
                    workspace=workspace,
                    memory_root=root / "memory",
                    allow_workspace_mutations=True,
                )
            )

            self.assertEqual(service.file_access_mode, "workspace")
            service.configure_file_access(mode="read_only")
            with self.assertRaisesRegex(PermissionError, "read-only"):
                service.write_file(path="blocked.txt", content="blocked")
            service.configure_file_access(mode="workspace")
            service.write_file(path="allowed.txt", content="allowed")
            self.assertTrue((workspace / "allowed.txt").exists())

    def test_full_file_access_requires_host_confirmation(self) -> None:
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            service = OSAgentMCPService(
                OSAgentMCPConfig.from_values(
                    workspace=workspace,
                    memory_root=root / "memory",
                    allow_workspace_mutations=True,
                )
            )
            with self.assertRaises(PermissionError):
                service.configure_file_access(mode="full_access")

    def test_full_file_access_allows_an_absolute_path_when_confirmed(self) -> None:
        with TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "OS_AGENT_FILE_FULL_ACCESS_CONFIRM": (
                    "I_UNDERSTAND_FULL_FILESYSTEM_ACCESS"
                )
            },
            clear=True,
        ):
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            service = OSAgentMCPService(
                OSAgentMCPConfig.from_values(
                    workspace=workspace,
                    memory_root=root / "memory",
                    allow_workspace_mutations=True,
                )
            )
            service.configure_file_access(mode="full_access")
            target = root / "outside" / "result.txt"
            result = service.write_file(
                path=str(target),
                content="outside workspace",
                create_parents=True,
            )

            self.assertEqual(result["access_mode"], "full_access")
            self.assertEqual(target.read_text(encoding="utf-8"), "outside workspace")


class OSAgentMCPProtocolTests(unittest.TestCase):
    def test_official_client_lists_and_calls_tools(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            server = create_mcp_server(
                OSAgentMCPConfig.from_values(
                    workspace=workspace,
                    memory_root=Path(tmp) / "memory",
                    use_llm=False,
                )
            )

            async def scenario() -> None:
                async with Client(server) as client:
                    listed = await client.list_tools()
                    tools = {tool.name: tool for tool in listed.tools}
                    self.assertEqual(
                        set(tools),
                        {
                            "os_agent_capabilities",
                            "os_agent_run",
                            "os_agent_memory_recall",
                            "os_agent_memory_remember",
                            "os_agent_memory_update",
                            "os_agent_memory_forget",
                            "os_agent_file_write",
                            "os_agent_plugin_marketplace",
                            "os_agent_computer_use",
                        },
                    )
                    recall_schema = tools["os_agent_memory_recall"].input_schema
                    self.assertIn("query_variants", recall_schema["properties"])
                    self.assertIn("subqueries", recall_schema["properties"])

                    result = await client.call_tool("os_agent_capabilities", {})
                    self.assertFalse(result.is_error)
                    self.assertEqual(result.structured_content["service"], "os-agent")

            asyncio.run(scenario())

    def test_computer_use_screenshot_returns_an_image_block(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            screenshot = workspace / "desktop.png"
            screenshot.write_bytes(
                base64.b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
                    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
                )
            )
            service = OSAgentMCPService(
                OSAgentMCPConfig.from_values(
                    workspace=workspace,
                    memory_root=root / "memory",
                    use_llm=False,
                )
            )
            with patch.object(
                service,
                "computer_use",
                return_value={
                    "action": "screenshot",
                    "screenshot": str(screenshot),
                    "width": 1,
                    "height": 1,
                    "backend": "test",
                },
            ):
                server = create_mcp_server(service=service)

                async def scenario() -> None:
                    async with Client(server) as client:
                        result = await client.call_tool(
                            "os_agent_computer_use",
                            {"action": "screenshot", "target": "desktop"},
                        )
                        self.assertFalse(result.is_error)
                        self.assertTrue(result.structured_content["image_content"])
                        images = [
                            item for item in result.content if isinstance(item, ImageContent)
                        ]
                        self.assertEqual(len(images), 1)
                        self.assertEqual(images[0].mime_type, "image/png")
                        self.assertTrue(images[0].data)

                asyncio.run(scenario())

    def test_official_client_writes_workspace_file_when_enabled(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            server = create_mcp_server(
                OSAgentMCPConfig.from_values(
                    workspace=workspace,
                    memory_root=Path(tmp) / "memory",
                    use_llm=False,
                    allow_workspace_mutations=True,
                )
            )

            async def scenario() -> None:
                async with Client(server) as client:
                    result = await client.call_tool(
                        "os_agent_file_write",
                        {
                            "path": "mcp/output.txt",
                            "content": "written through MCP",
                            "create_parents": True,
                        },
                    )
                    self.assertFalse(result.is_error)
                    self.assertEqual(result.structured_content["path"], "mcp/output.txt")

            asyncio.run(scenario())
            self.assertEqual(
                (workspace / "mcp" / "output.txt").read_text(encoding="utf-8"),
                "written through MCP",
            )

    def test_official_client_runs_agent_without_llm(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            server = create_mcp_server(
                OSAgentMCPConfig.from_values(
                    workspace=workspace,
                    memory_root=Path(tmp) / "memory",
                    use_llm=False,
                )
            )

            async def scenario() -> None:
                async with Client(server) as client:
                    result = await client.call_tool(
                        "os_agent_run",
                        {
                            "prompt": "Report the current OS state.",
                            "session_id": "mcp-smoke",
                        },
                    )
                    self.assertFalse(result.is_error)
                    self.assertEqual(
                        result.structured_content["session_id"],
                        "mcp-smoke",
                    )
                    self.assertIn("content", result.structured_content)

            asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
