#!/usr/bin/env python3
"""Verify that a deployed Computer Use screenshot includes MCP image content."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from verify_mcp import MCP_PROTOCOL_VERSION, _parse_env, _request


def _call_screenshot(url: str, api_key: str, screenshot_path: str) -> dict[str, Any]:
    initialize = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "osagent-image-verifier", "version": "1"},
        },
    }
    _, headers, initialized = _request(url, api_key, "POST", initialize)
    if not initialized or initialized.get("error"):
        raise RuntimeError(f"MCP initialization failed: {initialized}")
    session_id = headers.get("mcp-session-id", "")
    if not session_id:
        raise RuntimeError("MCP initialization did not return a session ID")

    try:
        _request(
            url,
            api_key,
            "POST",
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            session_id=session_id,
        )
        _, _, response = _request(
            url,
            api_key,
            "POST",
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "os_agent_computer_use",
                    "arguments": {
                        "action": "screenshot",
                        "target": "desktop",
                        "screenshot_path": screenshot_path,
                    },
                },
            },
            session_id=session_id,
        )
        if not response or response.get("error"):
            raise RuntimeError(f"MCP screenshot call failed: {response}")
        result = response.get("result") or {}
        if result.get("isError"):
            raise RuntimeError(f"OSagent screenshot failed: {result}")
        content = result.get("content") or []
        images = [item for item in content if item.get("type") == "image"]
        structured = result.get("structuredContent")
        return {
            "mcp_ok": bool(images),
            "structured_content": structured,
            "image_blocks": len(images),
            "image_mime_types": [item.get("mimeType", "") for item in images],
            "image_base64_lengths": [len(str(item.get("data", ""))) for item in images],
        }
    finally:
        _request(url, api_key, "DELETE", session_id=session_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765/mcp")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path.home() / ".config" / "osagent" / "osagent-memory.env",
    )
    parser.add_argument(
        "--screenshot-path",
        default="artifacts/mcp-image-verify.png",
    )
    args = parser.parse_args()
    values = _parse_env(args.env_file)
    api_key = values.get("OS_AGENT_MEMORY_API_KEY", "")
    if not api_key:
        raise SystemExit("OS_AGENT_MEMORY_API_KEY is not configured")
    result = _call_screenshot(args.url, api_key, args.screenshot_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["mcp_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
