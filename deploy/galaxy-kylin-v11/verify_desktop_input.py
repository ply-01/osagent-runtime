#!/usr/bin/env python3
"""Verify direct desktop mouse/keyboard actions through the OSagent MCP tool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from verify_mcp import MCP_PROTOCOL_VERSION, _parse_env, _request


def _tool_payload(response: dict[str, Any] | None) -> dict[str, Any]:
    if not response or response.get("error"):
        raise RuntimeError(f"MCP tool call failed: {response}")
    result = response.get("result") or {}
    if result.get("isError"):
        raise RuntimeError(f"OSagent tool returned an error: {result}")
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        return structured
    for item in result.get("content") or []:
        if isinstance(item, dict) and item.get("type") == "text":
            try:
                decoded = json.loads(str(item.get("text", "")))
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, dict):
                return decoded
    raise RuntimeError(f"MCP tool did not return structured JSON: {result}")


def _call(
    url: str,
    api_key: str,
    session_id: str,
    request_id: int,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    _, _, response = _request(
        url,
        api_key,
        "POST",
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": "os_agent_computer_use", "arguments": arguments},
        },
        session_id=session_id,
    )
    return _tool_payload(response)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765/mcp")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path.home() / ".config" / "osagent" / "osagent-memory.env",
    )
    parser.add_argument("--x", type=int, default=1)
    parser.add_argument("--y", type=int, default=1)
    parser.add_argument("--no-keypress", action="store_true")
    args = parser.parse_args()
    values = _parse_env(args.env_file)
    api_key = values.get("OS_AGENT_MEMORY_API_KEY", "")
    if not api_key:
        raise RuntimeError("OS_AGENT_MEMORY_API_KEY is missing")

    _, headers, initialized = _request(
        args.url,
        api_key,
        "POST",
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "osagent-desktop-verifier", "version": "1"},
            },
        },
    )
    if not initialized or initialized.get("error"):
        raise RuntimeError(f"MCP initialization failed: {initialized}")
    session_id = headers.get("mcp-session-id", "")
    if not session_id:
        raise RuntimeError("MCP initialization did not return a session ID")
    try:
        _request(
            args.url,
            api_key,
            "POST",
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            session_id=session_id,
        )
        results = [
            _call(
                args.url,
                api_key,
                session_id,
                2,
                {"action": "status", "target": "desktop"},
            ),
            _call(
                args.url,
                api_key,
                session_id,
                3,
                {"action": "move", "target": "desktop", "x": args.x, "y": args.y},
            ),
        ]
        if not args.no_keypress:
            results.append(
                _call(
                    args.url,
                    api_key,
                    session_id,
                    4,
                    {"action": "keypress", "target": "desktop", "key": "Escape"},
                )
            )
        results.append(
            _call(
                args.url,
                api_key,
                session_id,
                5,
                {
                    "action": "screenshot",
                    "target": "desktop",
                    "screenshot_path": "artifacts/desktop-direct-input.png",
                },
            )
        )
        print(json.dumps({"mcp_ok": True, "results": results}, ensure_ascii=False, indent=2))
        return 0
    finally:
        _request(args.url, api_key, "DELETE", session_id=session_id)


if __name__ == "__main__":
    raise SystemExit(main())
