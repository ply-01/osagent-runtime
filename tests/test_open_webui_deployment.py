from __future__ import annotations

import importlib.util
import json

import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = REPO_ROOT / "deploy" / "open-webui" / "compose.yaml"
GALAXY_KYLIN_DIR = REPO_ROOT / "deploy" / "galaxy-kylin-v11"


def _load_verify_mcp_module():
    path = GALAXY_KYLIN_DIR / "verify_mcp.py"
    spec = importlib.util.spec_from_file_location("verify_mcp", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_galaxy_kylin_webui_connects_to_native_loopback_service() -> None:
    compose = (GALAXY_KYLIN_DIR / "compose.yaml").read_text(encoding="utf-8")

    assert "network_mode: host" in compose
    assert "HOST: ${WEBUI_BIND_ADDRESS:-127.0.0.1}" in compose
    assert "OS_AGENT_MEMORY_URL: http://127.0.0.1:" in compose
    assert "osagent-memory:" not in compose
    assert 'ENABLE_MEMORIES: "False"' in compose
    assert 'ENABLE_NOTES: "False"' in compose
    assert 'OFFLINE_MODE: "True"' in compose
    assert 'OS_AGENT_MCP_AUTO_REGISTER: "True"' in compose
    assert "WEBUI_URL: ${WEBUI_PUBLIC_URL:-http://127.0.0.1:${WEBUI_PORT:-10200}}" in compose
    assert "OS_AGENT_PUBLIC_WEBUI_URL: ${WEBUI_PUBLIC_URL:-http://127.0.0.1:${WEBUI_PORT:-10200}}" in compose


def test_open_webui_restores_the_managed_osagent_mcp_at_startup() -> None:
    if not (REPO_ROOT / "vendor/open-webui").exists():
        pytest.skip("optional Open WebUI source/overlay is not included in the core snapshot")
    main = (
        REPO_ROOT / "vendor" / "open-webui" / "backend" / "open_webui" / "main.py"
    ).read_text(encoding="utf-8")

    assert "OS_AGENT_MCP_AUTO_REGISTER" in main
    assert "OS_AGENT_PUBLIC_WEBUI_URL" in main
    assert "await Config.upsert({'webui.url': osagent_public_webui_url})" in main
    assert "'id': 'osagent'" in main
    assert "await Config.upsert({'tool_server.connections': connections})" in main

    native_dockerfile = (GALAXY_KYLIN_DIR / "Dockerfile.native").read_text(
        encoding="utf-8"
    )
    build_script = (GALAXY_KYLIN_DIR / "build-openwebui.sh").read_text(
        encoding="utf-8"
    )
    assert "COPY backend/open_webui/main.py /app/backend/open_webui/main.py" in native_dockerfile
    assert 'OVERLAY_CONTEXT="$(mktemp -d' in build_script
    assert 'cp -a "$ROOT_DIR/vendor/open-webui/build" "$OVERLAY_CONTEXT/build"' in build_script
    assert '"$OVERLAY_CONTEXT"' in build_script


def test_galaxy_kylin_service_uses_private_environment_file() -> None:
    service = (GALAXY_KYLIN_DIR / "osagent-memory.service").read_text(
        encoding="utf-8"
    )

    assert "EnvironmentFile=%h/.config/osagent/osagent-memory.env" in service
    assert "WorkingDirectory=%h/.local/share/osagent" in service
    assert "ExecStart=%h/.local/share/osagent/venv/bin/os-agent-web" in service
    assert "PrivateTmp=false" in service
    assert "ProtectSystem=full" in service


def test_galaxy_kylin_installer_records_the_browser_visible_webui_url() -> None:
    installer = (GALAXY_KYLIN_DIR / "install.sh").read_text(encoding="utf-8")

    assert "--webui-public-url" in installer
    assert 'set_env "$webui_env" WEBUI_PUBLIC_URL "$webui_public_url"' in installer


def test_verify_mcp_decodes_json_and_event_stream_responses() -> None:
    module = _load_verify_mcp_module()
    payload = {"jsonrpc": "2.0", "id": 2, "result": {"tools": []}}
    encoded = json.dumps(payload).encode("utf-8")

    assert module._decode_response(encoded, "application/json") == payload
    assert module._decode_response(
        b"event: message\ndata: " + encoded + b"\n\n",
        "text/event-stream",
    ) == payload


def test_galaxy_kylin_installer_keeps_secrets_out_of_arguments() -> None:
    installer = (GALAXY_KYLIN_DIR / "install.sh").read_text(encoding="utf-8")

    assert "secrets.token_hex(32)" in installer
    assert "chmod 600" in installer
    assert "OS_AGENT_EMBEDDING_CONFIG" in installer
    assert "--api-key" not in installer


def test_galaxy_kylin_installer_targets_v11_and_webui_port() -> None:
    installer = (GALAXY_KYLIN_DIR / "install.sh").read_text(encoding="utf-8")
    start_script = (GALAXY_KYLIN_DIR / "start-webui.sh").read_text(
        encoding="utf-8"
    )

    assert "--webui-port" in installer
    assert "--webui-bind-address" in installer
    assert "version_id,,} != v11" in installer
    assert "webui_port=10200" in installer
    assert 'set_env "$webui_env" WEBUI_PORT "$webui_port"' in installer
    assert "${webui_port:-3000}" in start_script


def test_galaxy_kylin_installer_builds_against_matching_vector_sdk() -> None:
    installer = (GALAXY_KYLIN_DIR / "install.sh").read_text(encoding="utf-8")

    assert "libkysdk-vector-engine-client-dev" in installer
    assert "vector_runtime_version" in installer
    assert 'vector_runtime_version != "$vector_development_version"' in installer
    assert 'native/kylin_vector_bridge/build.sh" "$bridge_path"' in installer


def test_galaxy_kylin_preflight_runs_strict_native_sdk_smoke() -> None:
    preflight = (GALAXY_KYLIN_DIR / "preflight.sh").read_text(encoding="utf-8")

    assert "competition preflight" in preflight
    assert "competition vector-smoke" in preflight
    assert '. "$host_env"' in preflight
    assert '[[ -n $docker_bin ]] && "$docker_bin" compose version' in preflight


def test_galaxy_kylin_deployment_has_no_openkylin_or_kare_files() -> None:
    files = list(GALAXY_KYLIN_DIR.iterdir())
    assert not any(path.name in {"docker-kare.service", "ssh-kare.service"} for path in files)
    for path in files:
        if path.is_file() and path.suffix in {".sh", ".yaml", ".yml", ".md", ".service", ".example"}:
            assert "openkylin" not in path.read_text(encoding="utf-8").lower()
            assert "kare" not in path.read_text(encoding="utf-8").lower()


def test_standalone_marketplace_has_no_knowledge_upload_controls() -> None:
    content = (REPO_ROOT / "src/os_agent/plugins/marketplace_ui.py").read_text(encoding="utf-8")
    assert "/v1/knowledge/upload" not in content
    assert "Knowledge file upload" not in content
    assert "dropzone" not in content
