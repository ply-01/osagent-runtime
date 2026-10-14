#!/usr/bin/env python3
"""Register the OSagent memory filter in an existing Open WebUI instance."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


FUNCTION_ID = "osagent_memory_filter"
MCP_SERVER_ID = "osagent"


def _request(
    base_url: str,
    token: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> Any:
    data = None
    headers = {"Authorization": f"Bearer {token}"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Open WebUI returned HTTP {exc.code}: {detail}") from exc
    return json.loads(body) if body else None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:3000")
    parser.add_argument(
        "--function-file",
        type=Path,
        default=Path(__file__).parents[2]
        / "integrations"
        / "open_webui"
        / "osagent_memory_filter.py",
    )
    parser.add_argument("--memory-url", default="http://127.0.0.1:8765")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    token = os.environ.get("OPEN_WEBUI_TOKEN", "").strip().strip('"').strip("'")
    memory_key = (
        os.environ.get("OS_AGENT_MEMORY_API_KEY", "")
        .strip()
        .strip('"')
        .strip("'")
    )
    provider_url = os.environ.get("OPENAI_API_BASE_URL", "").strip().rstrip("/")
    provider_key = (
        os.environ.get("OPENAI_API_KEY", "").strip().strip('"').strip("'")
    )
    provider_model = os.environ.get("OPENAI_MODEL", "").strip()
    if not token or not memory_key:
        print(
            "OPEN_WEBUI_TOKEN and OS_AGENT_MEMORY_API_KEY must be set",
            file=sys.stderr,
        )
        return 2

    content = args.function_file.read_text(encoding="utf-8")
    functions = _request(args.base_url, token, "GET", "/api/v1/functions/")
    existing = next(
        (item for item in functions if item.get("id") == FUNCTION_ID),
        None,
    )
    if existing is None:
        existing = _request(
            args.base_url,
            token,
            "POST",
            "/api/v1/functions/create",
            {
                "id": FUNCTION_ID,
                "name": "OSagent Memory",
                "content": content,
                "meta": {
                    "description": "Automatic OSagent memory recall and observation",
                    "manifest": {},
                },
            },
        )

    _request(
        args.base_url,
        token,
        "POST",
        f"/api/v1/functions/id/{FUNCTION_ID}/valves/update",
        {
            "memory_url": args.memory_url,
            "api_key": memory_key,
            "recall_limit": 8,
            "max_context_chars": 6000,
            "timeout_seconds": 15.0,
            "fail_closed": False,
        },
    )
    if not existing.get("is_active", False):
        existing = _request(
            args.base_url,
            token,
            "POST",
            f"/api/v1/functions/id/{FUNCTION_ID}/toggle",
        )
    if not existing.get("is_global", False):
        existing = _request(
            args.base_url,
            token,
            "POST",
            f"/api/v1/functions/id/{FUNCTION_ID}/toggle/global",
        )

    tool_server = {
        "url": f"{args.memory_url.rstrip('/')}/mcp",
        "path": "",
        "type": "mcp",
        "auth_type": "bearer",
        "headers": None,
        "key": memory_key,
        "config": {"enable": True, "access_grants": []},
        "info": {
            "id": MCP_SERVER_ID,
            "name": "OSagent Tools",
            "description": "OSagent memory and workspace tools",
        },
    }
    verification = _request(
        args.base_url,
        token,
        "POST",
        "/api/v1/configs/tool_servers/verify",
        tool_server,
    )
    if not verification.get("status"):
        raise RuntimeError("Open WebUI did not verify the OSagent MCP server")

    current_config = _request(
        args.base_url,
        token,
        "GET",
        "/api/v1/configs/tool_servers",
    )
    connections = current_config.get("TOOL_SERVER_CONNECTIONS") or []
    connections = [
        connection
        for connection in connections
        if (connection.get("info") or {}).get("id") != MCP_SERVER_ID
    ]
    connections.append(tool_server)
    _request(
        args.base_url,
        token,
        "POST",
        "/api/v1/configs/tool_servers",
        {"TOOL_SERVER_CONNECTIONS": connections},
    )

    llm_configured = bool(provider_url and provider_key and provider_model)
    if llm_configured:
        _request(
            args.base_url,
            token,
            "POST",
            "/openai/config/update",
            {
                "ENABLE_OPENAI_API": True,
                "OPENAI_API_BASE_URLS": [provider_url],
                "OPENAI_API_KEYS": [provider_key],
                "OPENAI_API_CONFIGS": {},
            },
        )
        _request(
            args.base_url,
            token,
            "POST",
            "/api/v1/configs/models",
            {
                "DEFAULT_MODELS": provider_model,
                "DEFAULT_PINNED_MODELS": provider_model,
                "MODEL_ORDER_LIST": [provider_model],
                "DEFAULT_MODEL_METADATA": {},
                "DEFAULT_MODEL_PARAMS": {},
            },
        )

    print(
        json.dumps(
            {
                "id": existing.get("id"),
                "active": existing.get("is_active"),
                "global": existing.get("is_global"),
                "valves_configured": True,
                "mcp_verified": True,
                "mcp_tool_count": len(verification.get("specs") or []),
                "llm_configured": llm_configured,
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
