from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from os_agent.plugins.openai_catalog import import_openai_catalog


def _write_plugin(checkout: Path, name: str) -> None:
    plugin = checkout / "plugins" / name
    (plugin / ".codex-plugin").mkdir(parents=True)
    (plugin / ".codex-plugin" / "plugin.json").write_text(
        json.dumps(
            {
                "name": name,
                "version": "0.1.0",
                "description": "Long description",
                "author": {"name": "OpenAI"},
                "repository": "https://github.com/openai/plugins",
                "interface": {
                    "displayName": "GitHub",
                    "shortDescription": "Repository workflows",
                    "category": "Developer Tools",
                    "capabilities": ["Interactive", "Write"],
                },
            }
        ),
        encoding="utf-8",
    )


def test_import_openai_catalog_copies_valid_bundles_and_preserves_custom_entries() -> None:
    with TemporaryDirectory() as tmp:
        base = Path(tmp)
        checkout = base / "checkout"
        plugin_root = base / "runtime" / "plugins"
        _write_plugin(checkout, "github")
        marketplace = checkout / ".agents" / "plugins"
        marketplace.mkdir(parents=True)
        (marketplace / "marketplace.json").write_text(
            json.dumps(
                {
                    "name": "openai-curated",
                    "plugins": [
                        {
                            "name": "github",
                            "source": {"source": "local", "path": "./plugins/github"},
                            "category": "Developer Tools",
                        },
                        {
                            "name": "missing",
                            "source": {"source": "local", "path": "./plugins/missing"},
                            "category": "Productivity",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        plugin_root.mkdir(parents=True)
        (plugin_root / "catalog.json").write_text(
            json.dumps(
                {
                    "plugins": [
                        {"name": "custom", "version": "1.0", "description": "keep me"},
                        {
                            "name": "old-official",
                            "version": "0.1",
                            "source_marketplace": "openai-curated",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

        with patch(
            "os_agent.plugins.openai_catalog._git_revision",
            return_value="a" * 40,
        ):
            result = import_openai_catalog(
                checkout,
                plugin_root,
                revision="a" * 40,
            )

        assert result["imported"] == 1
        assert result["preserved_custom"] == 1
        assert result["skipped"][0]["name"] == "missing"
        output = json.loads((plugin_root / "catalog.json").read_text(encoding="utf-8"))
        assert [item["name"] for item in output["plugins"]] == ["custom", "github"]
        github = output["plugins"][1]
        assert github["display_name"] == "GitHub"
        assert github["category"] == "Developer Tools"
        assert github["compatibility"] == "codex-bundle"
        assert (plugin_root / "sources" / github["source_path"] / ".codex-plugin" / "plugin.json").is_file()


def test_import_openai_catalog_rejects_a_revision_that_is_not_checkout_head() -> None:
    with TemporaryDirectory() as tmp:
        checkout = Path(tmp) / "checkout"
        marketplace = checkout / ".agents" / "plugins"
        marketplace.mkdir(parents=True)
        (marketplace / "marketplace.json").write_text(
            json.dumps({"plugins": []}),
            encoding="utf-8",
        )

        with patch(
            "os_agent.plugins.openai_catalog._git_revision",
            return_value="a" * 40,
        ):
            try:
                import_openai_catalog(
                    checkout,
                    Path(tmp) / "runtime" / "plugins",
                    revision="b" * 40,
                )
            except ValueError as error:
                assert "does not match" in str(error)
            else:
                raise AssertionError("mismatched checkout revision must be rejected")
