#!/usr/bin/env python3
"""Verify the deployed OSagent Streamable HTTP MCP endpoint."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


MCP_PROTOCOL_VERSION = "2025-06-18"


def _parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _decode_response(body: bytes, content_type: str) -> dict[str, Any] | None:
    if not body.strip():
        return None
    text = body.decode("utf-8")
    if "text/event-stream" not in content_type.lower():
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise RuntimeError("MCP response must be a JSON object")
        return payload

    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = json.loads(line[5:].strip())
        if isinstance(payload, dict):
            return payload
    raise RuntimeError("MCP event stream did not contain a JSON data event")


def _curl_escape(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )


def _request(
    url: str,
    api_key: str,
    method: str,
    payload: dict[str, Any] | None = None,
    *,
    session_id: str = "",
) -> tuple[int, dict[str, str], dict[str, Any] | None]:
    config = [
        "silent",
        "show-error",
        "connect-timeout = 5",
        "max-time = 30",
        'dump-header = "-"',
        'output = "-"',
        f'url = "{_curl_escape(url)}"',
        f'request = "{method}"',
        f'header = "Authorization: Bearer {_curl_escape(api_key)}"',
        'header = "Accept: application/json, text/event-stream"',
    ]
    if payload is not None:
        encoded = json.dumps(payload, separators=(",", ":"))
        config.extend(
            (
                'header = "Content-Type: application/json"',
                f'data = "{_curl_escape(encoded)}"',
            )
        )
    if session_id:
        config.append(f'header = "Mcp-Session-Id: {_curl_escape(session_id)}"')

    completed = subprocess.run(
        ["curl", "--config", "-"],
        input=("\n".join(config) + "\n").encode("utf-8"),
        capture_output=True,
        text=False,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"curl failed with exit {completed.returncode}: {detail}")

    separator = b"\r\n\r\n" if b"\r\n\r\n" in completed.stdout else b"\n\n"
    header_block, body = completed.stdout.split(separator, 1)
    header_lines = header_block.decode("iso-8859-1").splitlines()
    status = int(header_lines[0].split()[1])
    headers = {
        name.strip().lower(): value.strip()
        for line in header_lines[1:]
        if ":" in line
        for name, value in (line.split(":", 1),)
    }
    if status >= 400:
        detail = body.decode("utf-8", errors="replace")
        raise RuntimeError(f"MCP returned HTTP {status}: {detail}")
    return status, headers, _decode_response(body, headers.get("content-type", ""))


def _progress(enabled: bool, stage: str) -> None:
    if enabled:
        print(f"mcp_stage={stage}", flush=True)


def _verify(url: str, api_key: str, *, verbose: bool = False) -> list[str]:
    initialize = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "osagent-deployment-verifier", "version": "1"},
        },
    }
    _progress(verbose, "initialize_start")
    _, headers, initialized = _request(url, api_key, "POST", initialize)
    _progress(verbose, "initialize_complete")
    if not initialized or initialized.get("error"):
        raise RuntimeError(f"MCP initialization failed: {initialized}")
    session_id = headers.get("mcp-session-id", "")
    if not session_id:
        raise RuntimeError("MCP initialization did not return a session ID")

    try:
        _progress(verbose, "initialized_notification_start")
        _request(
            url,
            api_key,
            "POST",
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            session_id=session_id,
        )
        _progress(verbose, "initialized_notification_complete")
        _progress(verbose, "tools_list_start")
        _, _, result = _request(
            url,
            api_key,
            "POST",
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            session_id=session_id,
        )
        _progress(verbose, "tools_list_complete")
        if not result or result.get("error"):
            raise RuntimeError(f"MCP tools/list failed: {result}")
        tools = (result.get("result") or {}).get("tools") or []
        return [str(tool["name"]) for tool in tools]
    finally:
        _progress(verbose, "session_delete_start")
        _request(url, api_key, "DELETE", session_id=session_id)
        _progress(verbose, "session_delete_complete")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765/mcp")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path.home() / ".config" / "osagent" / "osagent-memory.env",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    values = _parse_env(args.env_file)
    api_key = os.environ.get("OS_AGENT_MEMORY_API_KEY") or values.get(
        "OS_AGENT_MEMORY_API_KEY",
        "",
    )
    if not api_key:
        raise SystemExit("OS_AGENT_MEMORY_API_KEY is not configured")
    tools = _verify(args.url, api_key, verbose=args.verbose)
    print(f"mcp_ok=true tool_count={len(tools)} tools={','.join(sorted(tools))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
