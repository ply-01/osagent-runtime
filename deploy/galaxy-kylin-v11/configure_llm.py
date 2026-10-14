#!/usr/bin/env python3
"""Apply one OpenAI-compatible model configuration to OSagent and Open WebUI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlparse


def _set_env(path: Path, updates: dict[str, str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    positions = {
        line.split("=", 1)[0]: index
        for index, line in enumerate(lines)
        if "=" in line and not line.lstrip().startswith("#")
    }
    for name, value in updates.items():
        if "\n" in value or "\r" in value:
            raise ValueError(f"{name} must be a single-line value")
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        line = f'{name}="{escaped}"'
        if name in positions:
            lines[positions[name]] = line
        else:
            positions[name] = len(lines)
            lines.append(line)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    path.chmod(0o600)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--host-env",
        type=Path,
        default=Path.home() / ".config" / "osagent" / "osagent-memory.env",
    )
    parser.add_argument(
        "--webui-env",
        type=Path,
        default=Path(__file__).with_name(".env"),
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    base_url = str(config.get("base_url") or "").strip().rstrip("/")
    api_key = str(config.get("api_key") or "").strip()
    model = str(config.get("model") or "").strip()
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SystemExit("base_url must be an absolute HTTP(S) URL")
    if not api_key or not model:
        raise SystemExit("api_key and model must not be empty")

    _set_env(
        args.host_env,
        {
            "OS_AGENT_MCP_NO_LLM": "false",
            "OS_AGENT_BASE_URL": base_url,
            "OS_AGENT_API_KEY": api_key,
            "OS_AGENT_MODEL": model,
        },
    )
    _set_env(
        args.webui_env,
        {
            "OPENAI_API_BASE_URL": base_url,
            "OPENAI_API_KEY": api_key,
            "OPENAI_MODEL": model,
        },
    )
    print(f"llm_configured=true provider={parsed.hostname} model={model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
