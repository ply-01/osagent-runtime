import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from starlette.testclient import TestClient

from os_agent.mcp_server import OSAgentMCPConfig, OSAgentMCPService
from os_agent.web_service import create_web_app, main


class OSAgentWebServiceTests(unittest.TestCase):
    def _client(
        self,
        root: Path,
        *,
        allow_mutations: bool = True,
        allow_workspace_mutations: bool = False,
    ) -> TestClient:
        workspace = root / "workspace"
        workspace.mkdir()
        config = OSAgentMCPConfig.from_values(
            workspace=workspace,
            memory_root=root / "memory",
            use_llm=False,
            allow_memory_mutations=allow_mutations,
            allow_workspace_mutations=allow_workspace_mutations,
        )
        return TestClient(
            create_web_app(config, api_key="test-service-key"),
        )

    def test_health_is_public_but_memory_routes_require_bearer_auth(self) -> None:
        with TemporaryDirectory() as tmp:
            with self._client(Path(tmp)) as client:
                self.assertEqual(client.get("/health").status_code, 200)
                unauthorized = client.get("/v1/capabilities")
                authorized = client.get(
                    "/v1/capabilities",
                    headers={"Authorization": "Bearer test-service-key"},
                )

            self.assertEqual(unauthorized.status_code, 401)
            self.assertEqual(authorized.status_code, 200)
            self.assertEqual(authorized.json()["data"]["service"], "os-agent")

    def test_rest_exposes_memory_write_update_delete_and_recall(self) -> None:
        headers = {"Authorization": "Bearer test-service-key"}
        with TemporaryDirectory() as tmp:
            with self._client(Path(tmp)) as client:
                remembered = client.post(
                    "/v1/memory/manage",
                    headers=headers,
                    json={
                        "action": "remember",
                        "session_id": "web-write",
                        "arguments": {
                            "content": "The preferred editor theme is dark.",
                            "entity": "user",
                            "attribute": "editor_theme",
                            "value": "dark",
                        },
                    },
                )
                recalled = client.post(
                    "/v1/memory/recall",
                    headers=headers,
                    json={"query": "preferred editor theme", "limit": 5},
                )
                updated = client.post(
                    "/v1/memory/manage",
                    headers=headers,
                    json={
                        "action": "update",
                        "session_id": "web-update",
                        "arguments": {
                            "entity": "user",
                            "attribute": "editor_theme",
                            "value": "light",
                        },
                    },
                )
                forgotten = client.post(
                    "/v1/memory/manage",
                    headers=headers,
                    json={
                        "action": "forget",
                        "session_id": "web-forget",
                        "arguments": {
                            "entity": "user",
                            "attribute": "editor_theme",
                        },
                    },
                )

            self.assertEqual(remembered.status_code, 200)
            self.assertEqual(recalled.status_code, 200)
            self.assertTrue(recalled.json()["data"]["records"])
            self.assertEqual(updated.status_code, 200)
            self.assertEqual(forgotten.status_code, 200)

    def test_empty_api_key_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            config = OSAgentMCPConfig.from_values(
                workspace=workspace,
                memory_root=Path(tmp) / "memory",
                use_llm=False,
            )
            with self.assertRaisesRegex(ValueError, "api_key"):
                create_web_app(config, api_key="")

    def test_plugin_marketplace_page_is_public_but_api_is_authenticated(self) -> None:
        with TemporaryDirectory() as tmp:
            with self._client(Path(tmp)) as client:
                page = client.get("/plugins")
                unauthorized = client.get("/v1/plugins")
                authorized = client.get(
                    "/v1/plugins",
                    headers={"Authorization": "Bearer test-service-key"},
                )
            self.assertEqual(page.status_code, 200)
            self.assertIn("OSagent Plugin Marketplace", page.text)
            self.assertEqual(unauthorized.status_code, 401)
            self.assertEqual(authorized.status_code, 200)
            self.assertEqual(authorized.json()["data"]["plugins"], [])

    def test_computer_use_policy_can_be_toggled_and_persists(self) -> None:
        headers = {"Authorization": "Bearer test-service-key"}
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self._client(root) as client:
                initial = client.get("/v1/computer-use", headers=headers)
                enabled = client.put(
                    "/v1/computer-use",
                    headers=headers,
                    json={"enabled": True, "mode": "allowlist"},
                )
                after = client.get("/v1/computer-use", headers=headers)

            self.assertEqual(initial.status_code, 200)
            self.assertFalse(initial.json()["data"]["enabled"])
            self.assertEqual(enabled.status_code, 200)
            self.assertTrue(after.json()["data"]["enabled"])
            self.assertTrue(
                (root / "data" / "settings" / "computer-use.json").exists()
            )

    def test_computer_use_policy_rejects_non_boolean_enabled_value(self) -> None:
        headers = {"Authorization": "Bearer test-service-key"}
        with TemporaryDirectory() as tmp:
            with self._client(Path(tmp)) as client:
                response = client.put(
                    "/v1/computer-use",
                    headers=headers,
                    json={"enabled": "true"},
                )

            self.assertEqual(response.status_code, 400)
            self.assertIn("boolean", response.json()["error"])

    def test_file_access_policy_has_three_host_guarded_modes(self) -> None:
        headers = {"Authorization": "Bearer test-service-key"}
        with TemporaryDirectory() as tmp:
            with self._client(
                Path(tmp),
                allow_workspace_mutations=True,
            ) as client:
                initial = client.get("/v1/file-access", headers=headers)
                read_only = client.put(
                    "/v1/file-access",
                    headers=headers,
                    json={"mode": "read_only"},
                )
                full_access = client.put(
                    "/v1/file-access",
                    headers=headers,
                    json={"mode": "full_access"},
                )

            self.assertEqual(initial.json()["data"]["mode"], "workspace")
            self.assertEqual(len(initial.json()["data"]["modes"]), 3)
            self.assertEqual(read_only.status_code, 200)
            self.assertEqual(full_access.status_code, 403)

    def test_knowledge_upload_indexes_text_file(self) -> None:
        headers = {
            "Authorization": "Bearer test-service-key",
            "Content-Type": "text/plain; charset=utf-8",
        }
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self._client(root) as client:
                uploaded = client.post(
                    "/v1/knowledge/upload?filename=notes.txt",
                    headers=headers,
                    content="Portable deployment uses the native OSagent memory service.".encode(),
                )
                recalled = client.post(
                    "/v1/memory/recall",
                    headers={"Authorization": "Bearer test-service-key"},
                    json={"query": "Portable deployment native memory service", "limit": 5},
                )
            self.assertEqual(uploaded.status_code, 200)
            self.assertEqual(uploaded.json()["data"]["filename"], "notes.txt")
            self.assertTrue((root / "knowledge-files" / "notes.txt").exists())
            self.assertEqual(recalled.status_code, 200)
            self.assertTrue(recalled.json()["data"]["records"])

    def test_mobile_command_endpoint_is_authenticated_and_pollable(self) -> None:
        headers = {"Authorization": "Bearer test-service-key"}
        with TemporaryDirectory() as tmp, patch.object(
            OSAgentMCPService,
            "run_agent",
            return_value={"answer": "mobile command completed"},
        ) as run_agent:
            with self._client(Path(tmp)) as client:
                unauthorized = client.post(
                    "/v1/mobile/commands",
                    json={"prompt": "create a status report"},
                )
                submitted = client.post(
                    "/v1/mobile/commands",
                    headers=headers,
                    json={"prompt": "create a status report", "session_id": "phone-1"},
                )
                self.assertEqual(unauthorized.status_code, 401)
                self.assertEqual(submitted.status_code, 202)
                command_id = submitted.json()["data"]["command_id"]
                status = None
                for _ in range(30):
                    status = client.get(
                        f"/v1/mobile/commands/{command_id}", headers=headers
                    )
                    if status.json()["data"]["status"] in {"completed", "failed"}:
                        break
                    time.sleep(0.01)

            self.assertIsNotNone(status)
            self.assertEqual(status.status_code, 200)
            self.assertEqual(status.json()["data"]["status"], "completed")
            self.assertEqual(
                status.json()["data"]["result"]["answer"],
                "mobile command completed",
            )
            run_agent.assert_called_once_with("create a status report", session_id="phone-1")

    def test_main_reads_native_service_settings_from_environment(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            memory_root = Path(tmp) / "memory"
            environment = {
                "OS_AGENT_MEMORY_API_KEY": "native-service-key",
                "OS_AGENT_MEMORY_HOST": "127.0.0.1",
                "OS_AGENT_MEMORY_PORT": "9876",
                "OS_AGENT_MCP_WORKSPACE": str(workspace),
                "OS_AGENT_MCP_MEMORY_ROOT": str(memory_root),
                "OS_AGENT_MCP_NO_LLM": "true",
                "OS_AGENT_MCP_ALLOW_MEMORY_MUTATIONS": "true",
                "OS_AGENT_MCP_ALLOW_WORKSPACE_MUTATIONS": "true",
            }
            app = object()
            with (
                patch.dict(os.environ, environment, clear=False),
                patch("os_agent.web_service.create_web_app", return_value=app) as create,
                patch("os_agent.web_service.uvicorn.run") as run,
            ):
                main([])

            config = create.call_args.args[0]
            self.assertFalse(config.use_llm)
            self.assertTrue(config.allow_memory_mutations)
            self.assertTrue(config.allow_workspace_mutations)
            self.assertEqual(config.workspace, workspace.resolve())
            self.assertEqual(config.memory_root, memory_root.resolve())
            create.assert_called_once_with(
                config,
                api_key="native-service-key",
                host="127.0.0.1",
            )
            run.assert_called_once_with(app, host="127.0.0.1", port=9876)


if __name__ == "__main__":
    unittest.main()
