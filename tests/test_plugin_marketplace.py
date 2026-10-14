from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from os_agent.plugins.marketplace import PluginMarketplace


class PluginMarketplaceTests(TestCase):
    def test_search_reads_catalog_and_filters(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "catalog.json").write_text(
                json.dumps({"plugins": [{"name": "calendar", "version": "1.0", "description": "Calendar tools"}]}),
                encoding="utf-8",
            )
            market = PluginMarketplace(root)
            self.assertEqual(len(market.search("calendar")), 1)
            self.assertEqual(market.search("missing"), [])

    def test_search_includes_imported_catalog_metadata(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "catalog.json").write_text(
                json.dumps(
                    {
                        "plugins": [
                            {
                                "name": "github",
                                "display_name": "GitHub",
                                "version": "0.1.0",
                                "description": "Repository workflows",
                                "category": "Developer Tools",
                                "author": "OpenAI",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            market = PluginMarketplace(root)

            self.assertEqual(market.search("developer")[0]["display_name"], "GitHub")
            self.assertEqual(len(market.search("openai")), 1)

    def test_install_requires_flag_and_verifies_checksum(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "plugin.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("plugin.json", "{}")
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            (root / "catalog.json").write_text(
                json.dumps({"plugins": [{"name": "calendar", "version": "1.0", "download_url": "https://example.test/plugin.zip", "sha256": digest}]}),
                encoding="utf-8",
            )
            market = PluginMarketplace(root)
            with self.assertRaises(PermissionError):
                market.install("calendar")
            with patch.dict(os.environ, {"OS_AGENT_PLUGIN_INSTALL_ENABLED": "true"}), patch(
                "urllib.request.urlopen", return_value=archive.open("rb")
            ) as opener:
                # The mocked file is sufficient to exercise checksum and safe extraction.
                result = market.install("calendar")
            self.assertTrue(result["installed"])
            self.assertTrue(result["enabled"])
            self.assertTrue((root / "calendar-1.0" / "marketplace-manifest.json").exists())
            listed = market.list()
            self.assertEqual(listed[0]["status"], "enabled")
            disabled = market.set_enabled("calendar", False)
            self.assertEqual(disabled["status"], "disabled")
            self.assertFalse(market.list()[0]["enabled"])
            enabled = market.set_enabled("calendar", True)
            self.assertEqual(enabled["status"], "enabled")
            removed = market.uninstall("calendar")
            self.assertFalse(removed["installed"])
            self.assertFalse((root / "calendar-1.0").exists())
            opener.assert_called_once()

    def test_install_rejects_oversized_archive(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            digest = "0" * 64
            (root / "catalog.json").write_text(
                json.dumps({"plugins": [{"name": "large", "version": "1.0", "download_url": "https://example.test/plugin.zip", "sha256": digest}]}),
                encoding="utf-8",
            )

            class Response:
                headers = {"Content-Length": str(101 * 1024 * 1024)}

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return False

            market = PluginMarketplace(root)
            with patch.dict(os.environ, {"OS_AGENT_PLUGIN_INSTALL_ENABLED": "true"}), patch(
                "urllib.request.urlopen", return_value=Response()
            ):
                with self.assertRaisesRegex(ValueError, "100 MiB"):
                    market.install("large")

    def test_install_copies_a_local_catalog_source_without_executing_it(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "sources" / "official" / "plugins" / "github"
            (source / ".codex-plugin").mkdir(parents=True)
            (source / ".codex-plugin" / "plugin.json").write_text(
                json.dumps({"name": "github", "version": "0.1.0"}),
                encoding="utf-8",
            )
            (source / "payload.txt").write_text("data only", encoding="utf-8")
            (root / "catalog.json").write_text(
                json.dumps(
                    {
                        "plugins": [
                            {
                                "name": "github",
                                "version": "0.1.0",
                                "source_path": "official/plugins/github",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"OS_AGENT_PLUGIN_INSTALL_ENABLED": "true"}):
                result = PluginMarketplace(root).install("github")

            installed = Path(str(result["path"]))
            self.assertEqual((installed / "payload.txt").read_text(encoding="utf-8"), "data only")
            self.assertTrue((installed / "marketplace-manifest.json").is_file())

    def test_install_rejects_a_local_source_outside_the_source_root(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = root / "outside"
            (outside / ".codex-plugin").mkdir(parents=True)
            (outside / ".codex-plugin" / "plugin.json").write_text("{}", encoding="utf-8")
            (root / "catalog.json").write_text(
                json.dumps(
                    {
                        "plugins": [
                            {
                                "name": "escape",
                                "version": "1.0",
                                "source_path": "../outside",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"OS_AGENT_PLUGIN_INSTALL_ENABLED": "true"}):
                with self.assertRaisesRegex(ValueError, "escapes"):
                    PluginMarketplace(root).install("escape")

    def test_enabled_plugin_exposes_declarative_invocation(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "sources" / "official" / "plugins" / "notes"
            (source / ".codex-plugin").mkdir(parents=True)
            (source / ".codex-plugin" / "plugin.json").write_text(
                json.dumps({"name": "notes", "version": "1.0.0"}),
                encoding="utf-8",
            )
            (source / "SKILL.md").write_text(
                "# Notes\nCreate and search notes using the configured tool.",
                encoding="utf-8",
            )
            (root / "catalog.json").write_text(
                json.dumps(
                    {
                        "plugins": [
                            {
                                "name": "notes",
                                "version": "1.0.0",
                                "display_name": "Notes",
                                "source_path": "official/plugins/notes",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"OS_AGENT_PLUGIN_INSTALL_ENABLED": "true"}):
                market = PluginMarketplace(root)
                market.install("notes")
            listed = market.list()
            self.assertTrue(listed[0]["invocation_available"])
            result = market.invoke("notes", request="保存今天的会议记录")
            self.assertTrue(result["invoked"])
            self.assertEqual(result["execution"], "declarative_only")
            self.assertIn("Create and search notes", result["instructions"][0]["content"])

            market.set_enabled("notes", False)
            with self.assertRaises(PermissionError):
                market.invoke("notes")

    def test_mcp_plugin_exposes_a_sanitized_oauth_connector(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "sources" / "official" / "plugins" / "linear"
            (source / ".codex-plugin").mkdir(parents=True)
            (source / ".codex-plugin" / "plugin.json").write_text(
                json.dumps(
                    {
                        "name": "linear",
                        "version": "5.0.1",
                        "mcpServers": "./.mcp.json",
                    }
                ),
                encoding="utf-8",
            )
            (source / ".mcp.json").write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "linear": {
                                "type": "http",
                                "url": "https://mcp.linear.app/mcp",
                                "oauth_resource": "https://mcp.linear.app/mcp",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "catalog.json").write_text(
                json.dumps(
                    {
                        "plugins": [
                            {
                                "name": "linear",
                                "version": "5.0.1",
                                "source_path": "official/plugins/linear",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"OS_AGENT_PLUGIN_INSTALL_ENABLED": "true"}):
                market = PluginMarketplace(root)
                market.install("linear")

            listed = market.list()[0]
            self.assertEqual(listed["runtime_status"], "connection_required")
            self.assertFalse(listed["invocation_available"])
            self.assertEqual(
                listed["connectors"],
                [
                    {
                        "id": "osagent-plugin-linear-linear",
                        "name": "linear",
                        "url": "https://mcp.linear.app/mcp",
                        "transport": "http",
                        "auth_type": "oauth_2.1",
                        "oauth_resource": "https://mcp.linear.app/mcp",
                        "oauth_scope": "",
                        "connectable": True,
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "connect it in the host UI"):
                market.invoke("linear")

    def test_mcp_plugin_rejects_unsafe_runtime_declarations(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / "unsafe-1.0"
            (destination / ".codex-plugin").mkdir(parents=True)
            (destination / ".codex-plugin" / "plugin.json").write_text(
                json.dumps({"name": "unsafe", "version": "1.0"}),
                encoding="utf-8",
            )
            (destination / ".mcp.json").write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "command": {"command": "python", "args": ["plugin.py"]},
                            "plain-http": {"url": "http://example.test/mcp"},
                            "secret": {
                                "url": "https://example.test/mcp",
                                "headers": {"Authorization": "Bearer secret"},
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "catalog.json").write_text(
                json.dumps({"plugins": [{"name": "unsafe", "version": "1.0"}]}),
                encoding="utf-8",
            )
            (root / "marketplace-state.json").write_text(
                json.dumps({"version": 1, "plugins": {"unsafe@1.0": {"enabled": True}}}),
                encoding="utf-8",
            )

            listed = PluginMarketplace(root).list()[0]
            self.assertEqual(listed["runtime_status"], "unsupported")
            self.assertEqual(listed["connectors"], [])
            self.assertEqual(len(listed["runtime"]["unsupported"]), 3)
