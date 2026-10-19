from __future__ import annotations

import importlib.util
import json

import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = REPO_ROOT / "deploy" / "open-webui" / "compose.yaml"




def test_open_webui_native_memory_features_are_disabled() -> None:
    compose = COMPOSE_FILE.read_text(encoding="utf-8")

    for setting in (
        'ENABLE_MEMORIES: "False"',
        'ENABLE_MEMORY_SYSTEM_CONTEXT: "False"',
        'ENABLE_MEMORY_BACKGROUND_REVIEW: "False"',
        'ENABLE_NOTES: "False"',
    ):
        assert setting in compose


def test_plugin_marketplace_page_has_no_knowledge_upload_controls() -> None:
    if not (REPO_ROOT / "vendor/open-webui").exists():
        pytest.skip("optional Open WebUI source/overlay is not included in the core snapshot")
    page = (
        REPO_ROOT
        / "vendor"
        / "open-webui"
        / "src"
        / "routes"
        / "(app)"
        / "workspace"
        / "plugins"
        / "+page.svelte"
    ).read_text(encoding="utf-8")
    standalone = (
        REPO_ROOT / "src" / "os_agent" / "plugins" / "marketplace_ui.py"
    ).read_text(encoding="utf-8")

    for content in (page, standalone):
        assert "/v1/knowledge/upload" not in content
        assert "Knowledge file upload" not in content
        assert "dropzone" not in content


def test_plugin_marketplace_integrates_remote_mcp_with_chat() -> None:
    if not (REPO_ROOT / "vendor/open-webui").exists():
        pytest.skip("optional Open WebUI source/overlay is not included in the core snapshot")
    page = (
        REPO_ROOT
        / "vendor"
        / "open-webui"
        / "src"
        / "routes"
        / "(app)"
        / "workspace"
        / "plugins"
        / "+page.svelte"
    ).read_text(encoding="utf-8")

    assert "getToolServerConnections" in page
    assert "verifyToolServerConnection" in page
    assert "registerOAuthClient" in page
    assert "setToolServerConnections" in page
    assert "initiateOAuthRedirect" in page
    assert "integrationsReady" in page
    assert "integrationRetryTimer" in page
    assert "在对话中使用" in page
    assert "server:mcp:" in page


def test_chat_composer_keeps_osagent_tools_out_of_the_input_area() -> None:
    if not (REPO_ROOT / "vendor/open-webui").exists():
        pytest.skip("optional Open WebUI source/overlay is not included in the core snapshot")
    message_input = (
        REPO_ROOT
        / "vendor"
        / "open-webui"
        / "src"
        / "lib"
        / "components"
        / "chat"
        / "MessageInput.svelte"
    ).read_text(encoding="utf-8")

    assert "OSagentControls" not in message_input
    assert "osagent-plugin-invoke" not in message_input


def test_open_webui_uses_osagent_memory_service() -> None:
    compose = COMPOSE_FILE.read_text(encoding="utf-8")

    assert "OS_AGENT_MEMORY_URL: http://osagent-memory:8765" in compose
    assert "../../integrations/open_webui:/opt/osagent-integrations:ro" in compose
    assert "osagent-memory-data:/data/memory" in compose






















def test_standalone_marketplace_has_no_knowledge_upload_controls() -> None:
    content = (REPO_ROOT / "src/os_agent/plugins/marketplace_ui.py").read_text(encoding="utf-8")
    assert "/v1/knowledge/upload" not in content
    assert "Knowledge file upload" not in content
    assert "dropzone" not in content
