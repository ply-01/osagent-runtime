"""Verify the real main-agent/native-subagent boundary end to end."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from os_agent.agent.loop import AgentLoop
from os_agent.events.logger import EventLogger
from os_agent.llm.provider import LLMConfig
from os_agent.memory.markdown_store import MarkdownMemoryStore
from os_agent.subagents.memory import MemorySubagentError
from os_agent.subagents.native_memory import (
    OSagentMemorySubagentClient,
    OSagentMemorySubagentConfig,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "doc" / "evaluation" / "results" / "native-subagent-e2e.json"
)


def free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def start_subagent(root: Path, port: int, log_path: Path) -> subprocess.Popen[str]:
    log_handle = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "os_agent",
            "subagent",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--root",
            str(root),
            "--top-k",
            "8",
        ],
        cwd=ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_handle.close()
    return process


def stop_subagent(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def wait_for_health(
    client: OSagentMemorySubagentClient,
    process: subprocess.Popen[str],
    *,
    timeout_seconds: float = 15.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"subagent exited with code {process.returncode}")
        try:
            return client.health()
        except MemorySubagentError as exc:
            last_error = str(exc)
            time.sleep(0.1)
    raise RuntimeError(f"subagent did not become healthy: {last_error}")


def session_tool_calls(event_root: Path, session_id: str) -> list[str]:
    calls: list[str] = []
    for event_file in event_root.glob("*.jsonl"):
        for line in event_file.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("session_id") != session_id:
                continue
            if event.get("event_type") == "tool_call":
                calls.append(str(event.get("payload", {}).get("tool_name", "")))
    return calls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    marker = f"Polaris-{uuid4().hex[:8]}"
    llm_config = LLMConfig.from_env()
    started = time.perf_counter()
    result: dict[str, Any] = {
        "provenance": {
            "system": "OSagent main AgentLoop + native memory subagent",
            "evaluated_at": datetime.now(UTC).isoformat(),
            "uses_real_subagent_process": True,
            "uses_real_llm_planner": True,
            "planner_model": llm_config.model,
            "planner_endpoint_host": urlparse(llm_config.base_url).hostname,
            "test_marker": marker,
        }
    }

    with tempfile.TemporaryDirectory(prefix="osagent-native-subagent-e2e-") as tmp:
        temp_root = Path(tmp)
        subagent_root = temp_root / "subagent-memory"
        event_root = temp_root / "events"
        port = free_loopback_port()
        client = OSagentMemorySubagentClient(
            OSagentMemorySubagentConfig(
                base_url=f"http://127.0.0.1:{port}",
                top_k=8,
            )
        )
        process = start_subagent(subagent_root, port, temp_root / "subagent.log")
        try:
            initial_health = wait_for_health(client, process)
            loop = AgentLoop(
                workspace=temp_root / "workspace",
                event_logger=EventLogger(event_root),
                memory_store=MarkdownMemoryStore(temp_root / "main-memory"),
                use_llm=True,
                use_mem0=False,
                max_steps=1,
                memory_subagent=client,
                memory_subagent_mode="tool",
            )

            write_session = "native-e2e-write"
            write_response = loop.run(
                "请把跨会话交接代号 "
                f"{marker} 交给记忆 subagent 长期保存。",
                session_id=write_session,
            )
            write_planner_used_llm = bool(loop._last_plan_used_llm)
            write_calls = session_tool_calls(event_root, write_session)
            stored_hits = client.recall(marker, session_id="native-e2e-check")
            planner_write_succeeded = "subagent.memory_manage" in write_calls
            marker_stored_by_agent = any(
                marker.lower() in hit.content.lower() for hit in stored_hits
            )

            direct_seed_used = False
            if not marker_stored_by_agent:
                direct_seed_used = True
                client.manage(
                    "remember",
                    {
                        "content": f"The cross-session handoff code is {marker}.",
                        "entity": "evaluation",
                        "attribute": "handoff_code",
                        "value": marker,
                    },
                    session_id="native-e2e-seed",
                )

            stop_subagent(process)
            process = start_subagent(
                subagent_root,
                port,
                temp_root / "subagent.log",
            )
            restart_health = wait_for_health(client, process)
            persisted_hits = client.recall(
                marker,
                session_id="native-e2e-persistence-check",
            )
            persisted_after_restart = any(
                marker.lower() in hit.content.lower() for hit in persisted_hits
            )

            read_session = "native-e2e-read"
            read_response = loop.run(
                "我们之前约定的跨会话交接代号是什么？",
                session_id=read_session,
            )
            read_calls = session_tool_calls(event_root, read_session)
            planner_read_succeeded = "subagent.memory_search" in read_calls
            answer_contains_marker = marker.lower() in read_response.content.lower()

            result["checks"] = {
                "initial_health": initial_health.get("status") == "ok",
                "write_planner_used_llm": write_planner_used_llm,
                "write_tool_calls": write_calls,
                "write_routed_to_subagent": planner_write_succeeded,
                "marker_stored_by_main_agent": marker_stored_by_agent,
                "direct_seed_used": direct_seed_used,
                "restart_health": restart_health.get("status") == "ok",
                "persisted_after_restart": persisted_after_restart,
                "read_tool_calls": read_calls,
                "read_planner_used_llm": bool(loop._last_plan_used_llm),
                "read_routed_to_subagent": planner_read_succeeded,
                "answer_contains_marker": answer_contains_marker,
            }
            result["responses"] = {
                "write": write_response.content,
                "read": read_response.content,
            }
            result["passed"] = all(
                (
                    result["checks"]["initial_health"],
                    result["checks"]["write_routed_to_subagent"],
                    result["checks"]["marker_stored_by_main_agent"],
                    not result["checks"]["direct_seed_used"],
                    result["checks"]["restart_health"],
                    result["checks"]["persisted_after_restart"],
                    result["checks"]["read_routed_to_subagent"],
                    result["checks"]["answer_contains_marker"],
                )
            )
        finally:
            stop_subagent(process)

    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get("passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
