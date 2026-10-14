#!/usr/bin/env python3
"""Drive the deployed Qianxin browser through the public OSagent MCP tool."""

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
        if not isinstance(item, dict) or item.get("type") != "text":
            continue
        try:
            decoded = json.loads(str(item.get("text", "")))
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            return decoded
    raise RuntimeError(f"MCP tool did not return structured JSON: {result}")


def _call_tool(
    url: str,
    api_key: str,
    session_id: str,
    request_id: int,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    action = str(arguments.get("action", "unknown"))
    print(f"computer_use_stage={action}_start", flush=True)
    _, _, response = _request(
        url,
        api_key,
        "POST",
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {
                "name": "os_agent_computer_use",
                "arguments": arguments,
            },
        },
        session_id=session_id,
    )
    try:
        result = _tool_payload(response)
    except RuntimeError as exc:
        raise RuntimeError(f"computer-use action {action!r} failed: {exc}") from exc
    print(f"computer_use_stage={action}_complete", flush=True)
    return result


def _verify(
    url: str,
    api_key: str,
    *,
    target_url: str,
    query: str,
    input_selector: str,
    submit_selector: str,
    submit_key: str,
    screenshot_path: str,
    interact: bool,
) -> list[dict[str, Any]]:
    _, headers, initialized = _request(
        url,
        api_key,
        "POST",
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "osagent-qianxin-verifier",
                    "version": "1",
                },
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
            url,
            api_key,
            "POST",
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            session_id=session_id,
        )
        calls: list[dict[str, Any]] = [
            {"action": "status"},
            {"action": "launch", "url": target_url},
        ]
        if interact:
            calls.append(
                {
                    "action": "fill",
                    "selector": input_selector,
                    "text": query,
                }
            )
            calls.append(
                {
                    "action": "press",
                    "selector": input_selector,
                    "text": submit_key,
                }
                if submit_key
                else {"action": "click", "selector": submit_selector}
            )
        calls.extend(
            (
                {"action": "screenshot", "screenshot_path": screenshot_path},
                {"action": "status"},
            )
        )
        return [
            _call_tool(url, api_key, session_id, index + 2, arguments)
            for index, arguments in enumerate(calls)
        ]
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
    parser.add_argument("--target-url", default="https://www.baidu.com")
    parser.add_argument(
        "--keep-current-page",
        action="store_true",
        help="Do not navigate when an existing controlled browser is connected.",
    )
    parser.add_argument("--query", required=True)
    parser.add_argument("--input-selector", default="#kw")
    parser.add_argument("--submit-selector", default="#su")
    parser.add_argument(
        "--submit-key",
        default="",
        help="Press this key on the input instead of clicking the submit selector.",
    )
    parser.add_argument(
        "--no-interaction",
        action="store_true",
        help="Launch and capture the page without filling or clicking.",
    )
    parser.add_argument(
        "--screenshot-path",
        default="artifacts/qianxin-mcp-browser-test.png",
    )
    args = parser.parse_args()

    values = _parse_env(args.env_file)
    api_key = values.get("OS_AGENT_MEMORY_API_KEY", "")
    if not api_key:
        raise SystemExit("OS_AGENT_MEMORY_API_KEY is not configured")
    results = _verify(
        args.url,
        api_key,
        target_url="" if args.keep_current_page else args.target_url,
        query=args.query,
        input_selector=args.input_selector,
        submit_selector=args.submit_selector,
        submit_key=args.submit_key,
        screenshot_path=args.screenshot_path,
        interact=not args.no_interaction,
    )
    print(json.dumps({"mcp_ok": True, "results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
