"""CLI for the OS Agent — a long-term-memory-augmented task agent."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Run subcommand
# ---------------------------------------------------------------------------

RUN_HELP = """Run the agent with a task prompt.

The agent will plan tool calls (optionally via LLM), execute them,
and return a structured response.  If no LLM is configured, a
rule-based planner is used as fallback.

Examples:
  os-agent run "List the files in this project and summarize the README"
  os-agent run "帮我查看项目结构" --workspace D:\\my-project --session-id demo-01
  os-agent run "快速测试" --no-llm
"""


def _add_run_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "run",
        help="Execute a task prompt through the agent loop.",
        description=RUN_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Task prompt for the agent (can be quoted or multi-word).",
    )
    parser.add_argument(
        "--workspace", "-w",
        default=".",
        help="Workspace root directory.  Default: current directory.",
    )
    parser.add_argument(
        "--session-id", "-s",
        default=None,
        help="Stable session id for trace continuity.  Auto-generated if omitted.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable the LLM planner / responder; always use rule-based fallback.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Maximum tool-call steps per run.  Default: 5.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print extra diagnostic information.",
    )


# ---------------------------------------------------------------------------
# Session subcommand
# ---------------------------------------------------------------------------

SESSION_HELP = """Inspect persisted agent session traces.

Examples:
  os-agent session list
  os-agent session show demo-01
  os-agent session show demo-01 --json --limit 20
"""


def _add_session_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "session",
        help="List or inspect persisted session traces.",
        description=SESSION_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    session_sub = parser.add_subparsers(dest="session_command")

    list_p = session_sub.add_parser("list", help="List all persisted sessions.")
    list_p.add_argument("--root", default="data/sessions", help="Session trace root directory.")

    show_p = session_sub.add_parser("show", help="Display a specific session trace.")
    show_p.add_argument("session_id", help="Session identifier to inspect.")
    show_p.add_argument("--root", default="data/sessions", help="Session trace root directory.")
    show_p.add_argument("--limit", type=int, default=0, help="Max events to print (0 = all).")
    show_p.add_argument("--json", action="store_true", help="Output raw event JSON instead of compact summary.")


# ---------------------------------------------------------------------------
# Memory subcommand
# ---------------------------------------------------------------------------

MEMORY_HELP = """Manage the Markdown long-term memory store.

Examples:
  os-agent memory list
  os-agent memory pending
  os-agent memory page --strategy priority --query "planner memory"
  os-agent memory search "embedding retrieval" --mode hybrid
  os-agent memory maintain --max-sections-per-file 60
"""


def _add_memory_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "memory",
        help="List, review, or maintain the long-term memory store.",
        description=MEMORY_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mem_sub = parser.add_subparsers(dest="memory_command")

    list_p = mem_sub.add_parser("list", help="List all stable + pending memories.")
    list_p.add_argument("--root", default="memory", help="Memory root directory.")

    pending_p = mem_sub.add_parser("pending", help="List memories awaiting human confirmation.")
    pending_p.add_argument("--root", default="memory", help="Memory root directory.")

    page_p = mem_sub.add_parser("page", help="Show a page of stable Markdown memory.")
    page_p.add_argument("--root", default="memory", help="Memory root directory.")
    page_p.add_argument("--source", default=None, help="Memory source file or key.")
    page_p.add_argument("--layer", default=None, help="Memory layer: hot, warm, or cold.")
    page_p.add_argument("--query", default=None, help="Task query for priority paging.")
    page_p.add_argument(
        "--strategy",
        choices=["layered", "priority"],
        default="layered",
        help="Paging strategy.",
    )
    page_p.add_argument("--page", type=int, default=1, help="Page number, starting at 1.")
    page_p.add_argument("--page-size", type=int, default=10, help="Sections per page.")
    page_p.add_argument("--json", action="store_true", help="Output raw page JSON.")

    layers_p = mem_sub.add_parser("layers", help="Show memory layer counts.")
    layers_p.add_argument("--root", default="memory", help="Memory root directory.")
    layers_p.add_argument("--source", default=None, help="Memory source file or key.")
    layers_p.add_argument("--json", action="store_true", help="Output raw layer JSON.")

    search_p = mem_sub.add_parser("search", help="Search stable Markdown memory.")
    search_p.add_argument("query", help="Memory search query.")
    search_p.add_argument("--root", default="memory", help="Memory root directory.")
    search_p.add_argument("--source", default=None, help="Memory source file or key.")
    search_p.add_argument("--limit", type=int, default=5, help="Maximum matches.")
    search_p.add_argument(
        "--mode",
        choices=["keyword", "vector", "hybrid"],
        default="hybrid",
        help="Retrieval mode.",
    )
    search_p.add_argument("--json", action="store_true", help="Output raw result JSON.")

    maintain_p = mem_sub.add_parser("maintain", help="Compact, deduplicate, and archive old memories.")
    maintain_p.add_argument("--root", default="memory", help="Memory root directory.")
    maintain_p.add_argument(
        "--max-sections-per-file",
        type=int,
        default=80,
        help="Archive older sections when a stable memory file exceeds this count.",
    )


# ---------------------------------------------------------------------------
# Core (Mem0) subcommand
# ---------------------------------------------------------------------------

CORE_HELP = """Manage the Mem0-style Core structured memory.

Core Memory stores structured fact cards (entity + attribute + value)
with versioning and confidence tracking.

Examples:
  os-agent core list
  os-agent core list --entity 用户
  os-agent core search "预算"
  os-agent core stats
"""


def _add_core_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "core",
        help="Manage structured Core memory (Mem0 style).",
        description=CORE_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    core_sub = parser.add_subparsers(dest="core_command")

    list_p = core_sub.add_parser("list", help="List active Core facts.")
    list_p.add_argument("--entity", "-e", default=None, help="Filter by entity.")
    list_p.add_argument("--root", default="memory", help="Memory root directory.")

    search_p = core_sub.add_parser("search", help="Search Core facts.")
    search_p.add_argument("query", help="Search query.")
    search_p.add_argument("--entity", "-e", default=None, help="Filter by entity.")
    search_p.add_argument("--top-k", type=int, default=10, help="Max results.")
    search_p.add_argument("--root", default="memory", help="Memory root directory.")

    stats_p = core_sub.add_parser("stats", help="Show Core memory statistics.")
    stats_p.add_argument("--root", default="memory", help="Memory root directory.")


# ---------------------------------------------------------------------------
# Pipeline (Mem0) subcommand
# ---------------------------------------------------------------------------

PIPELINE_HELP = """Run the Mem0 Extract-Update Pipeline.

The pipeline batch-processes session/event logs into structured
Core facts, superseding old facts when values change.

Examples:
  os-agent pipeline run --session session-abc123
  os-agent pipeline run --all
  os-agent pipeline status
"""


def _add_pipeline_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "pipeline",
        help="Run the Extract-Update pipeline (Mem0 style).",
        description=PIPELINE_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    pipe_sub = parser.add_subparsers(dest="pipeline_command")

    run_p = pipe_sub.add_parser("run", help="Run pipeline on session(s).")
    run_p.add_argument("--session", "-s", default=None, help="Single session ID.")
    run_p.add_argument("--all", action="store_true", help="Process all sessions.")
    run_p.add_argument("--root", default="memory", help="Memory root directory.")

    status_p = pipe_sub.add_parser("status", help="Show pipeline status.")
    status_p.add_argument("--root", default="memory", help="Memory root directory.")


# ---------------------------------------------------------------------------
# Config subcommand
# ---------------------------------------------------------------------------

CONFIG_HELP = """View or update the LLM configuration.

The config file is stored at config/llm.local.json (overridable via
OS_AGENT_LLM_CONFIG env var).  Environment variables always take
precedence over the file.

Examples:
  os-agent config show
  os-agent config set --api-key sk-xxx --base-url https://api.deepseek.com --model deepseek-v4-flash
  os-agent config set --timeout 120 --temperature 0.3
"""


def _add_config_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "config",
        help="View or update LLM configuration.",
        description=CONFIG_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    config_sub = parser.add_subparsers(dest="config_command")

    # -- config show --------------------------------------------------------
    show_p = config_sub.add_parser("show", help="Print the current LLM config (file + env).")
    show_p.add_argument(
        "--config-file",
        default="config/llm.local.json",
        help="Path to the local config file.",
    )

    # -- config set ---------------------------------------------------------
    set_p = config_sub.add_parser("set", help="Update values in the local config file.")
    set_p.add_argument(
        "--config-file",
        default="config/llm.local.json",
        help="Path to the local config file.",
    )
    set_p.add_argument("--api-key", default=None, help="API key for the LLM provider.")
    set_p.add_argument("--base-url", default=None, help="Base URL of the OpenAI-compatible API.")
    set_p.add_argument("--model", default=None, help="Model name to use.")
    set_p.add_argument("--timeout", type=int, default=None, help="Request timeout in seconds.")
    set_p.add_argument("--temperature", type=float, default=None, help="LLM temperature (0.0-2.0).")


# ---------------------------------------------------------------------------
# Memory config subcommand
# ---------------------------------------------------------------------------

MEMORY_CONFIG_HELP = """View or tune memory runtime parameters.

Examples:
  os-agent memory-config show
  os-agent memory-config show --prefix core
  os-agent memory-config set recall_min_core_results 2
  os-agent memory-config set recall_include_legacy_markdown_context false
"""


def _add_memory_config_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "memory-config",
        help="View or tune memory runtime parameters.",
        description=MEMORY_CONFIG_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    config_sub = parser.add_subparsers(dest="memory_config_command")

    show_p = config_sub.add_parser("show", help="Show memory tuning parameters.")
    show_p.add_argument("--prefix", default=None, help="Only show keys with this prefix.")
    show_p.add_argument("--config-file", default=None, help="Path to memory_config.json.")
    show_p.add_argument("--json", action="store_true", help="Output flat JSON.")

    set_p = config_sub.add_parser("set", help="Set one memory tuning parameter.")
    set_p.add_argument("key", help="Flat key, e.g. recall_min_core_results.")
    set_p.add_argument("value", help="New value. Booleans/numbers are parsed automatically.")
    set_p.add_argument("--config-file", default=None, help="Path to memory_config.json.")


# ---------------------------------------------------------------------------
# Native memory subagent subcommand
# ---------------------------------------------------------------------------

SUBAGENT_HELP = """Run or inspect the independently persisted memory subagent.

Examples:
  os-agent subagent serve
  os-agent subagent serve --root data/subagents/native-memory --port 8765
  os-agent subagent health --url http://127.0.0.1:8765
"""


def _add_subagent_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "subagent",
        help="Run or inspect the native memory subagent.",
        description=SUBAGENT_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subagent_sub = parser.add_subparsers(dest="subagent_command")

    serve_p = subagent_sub.add_parser(
        "serve",
        help="Start the loopback native memory subagent service.",
    )
    serve_p.add_argument("--host", default="127.0.0.1", help="Loopback bind host.")
    serve_p.add_argument("--port", type=int, default=8765, help="HTTP port.")
    serve_p.add_argument(
        "--root",
        default="data/subagents/native-memory",
        help="Independent persistent memory root.",
    )
    serve_p.add_argument("--top-k", type=int, default=8, help="Recall result limit.")

    health_p = subagent_sub.add_parser(
        "health",
        help="Check a running native memory subagent.",
    )
    health_p.add_argument(
        "--url",
        default="http://127.0.0.1:8765",
        help="Native memory subagent base URL.",
    )




# ---------------------------------------------------------------------------
# Top-level parser
# ---------------------------------------------------------------------------

TOP_HELP = """OS Agent — a long-term-memory-augmented task agent.

Subcommands:
  run       Execute a task prompt through the agent loop.
  session   List or inspect persisted session traces.
  memory    Manage the Markdown long-term memory store.
  core      Manage structured Core memory.
  pipeline  Run the Extract-Update memory pipeline.
  config    View or update LLM configuration.
  memory-config
            View or tune memory runtime parameters.
  subagent  Run or inspect the native memory subagent.
  track1    Replay a portable OpenAtom industry scenario.

Run 'os-agent <subcommand> --help' for details on each subcommand.

Quick start:
  os-agent track1 --scenario examples/track1/maintenance-handover.json
  os-agent run "List the project files and summarize the README" --no-llm
"""


def _build_top_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="os-agent",
        description=TOP_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="store_true", help="Show version and exit.")
    subparsers = parser.add_subparsers(dest="command", title="subcommands")
    _add_run_parser(subparsers)
    _add_session_parser(subparsers)
    _add_memory_parser(subparsers)
    _add_core_parser(subparsers)
    _add_pipeline_parser(subparsers)
    _add_config_parser(subparsers)
    _add_memory_config_parser(subparsers)
    _add_subagent_parser(subparsers)
    track = subparsers.add_parser("track1", help="Run the OpenAtom industry-solution scenario.")
    track.add_argument("--scenario", type=Path, required=True, help="Public JSON scenario path.")
    track.add_argument("--output", type=Path, default=Path("artifacts/track1"))
    return parser


# ===================================================================
# main entry point
# ===================================================================

def main(argv: list[str] | None = None) -> None:
    _configure_stdio()

    raw = list(sys.argv[1:] if argv is None else argv)

    # --version short-circuit
    if "--version" in raw or "-V" in raw:
        from os_agent import __version__  # type: ignore[attr-defined]
        print(f"os-agent {__version__}")
        return

    parser = _build_top_parser()
    args = parser.parse_args(raw)

    # Dispatch
    dispatch = {
        "run": _handle_run,
        "session": _handle_session,
        "memory": _handle_memory,
        "core": _handle_core,
        "pipeline": _handle_pipeline,
        "config": _handle_config,
        "memory-config": _handle_memory_config,
        "subagent": _handle_subagent,
        "track1": _handle_track1,
    }

    command = args.command
    if command is None:
        parser.print_help()
        return

    handler = dispatch.get(command)
    if handler is None:
        parser.print_help()
        return

    handler(args)


# ===================================================================
# Handlers
# ===================================================================

def _handle_run(args: argparse.Namespace) -> None:
    from os_agent.agent.loop import AgentLoop

    prompt = " ".join(args.prompt).strip() if args.prompt else ""
    if not prompt:
        prompt = "帮我查看项目根目录有哪些文件，并总结这个项目是做什么的。"

    if args.verbose:
        print(f"[os-agent] workspace = {Path(args.workspace).resolve()}")
        print(f"[os-agent] session  = {args.session_id or '(auto)'}")
        print(f"[os-agent] llm      = {'off' if args.no_llm else 'auto'}")
        print(f"[os-agent] prompt   = {prompt[:120]}{'…' if len(prompt) > 120 else ''}")
        print()

    agent = AgentLoop(
        workspace=Path(args.workspace),
        use_llm=not args.no_llm,
        max_steps=args.max_steps,
    )
    response = agent.run(prompt, session_id=args.session_id)

    print(response.content)
    print()
    print(f"session_id: {response.session_id}")
    print(f"task_id:    {response.task_id}")
    if response.retrieved_memories:
        print(f"使用记忆:   {', '.join(response.retrieved_memories)}")
    if response.memory_candidates:
        print(f"新增记忆:   {', '.join(response.memory_candidates)}")


def _handle_subagent(args: argparse.Namespace) -> None:
    from os_agent.subagents.native_memory import (
        LocalMemorySubagent,
        LocalMemorySubagentConfig,
        OSagentMemorySubagentClient,
        OSagentMemorySubagentConfig,
    )

    if args.subagent_command == "serve":
        from os_agent.subagents.server import serve

        if args.port < 0 or args.port > 65535:
            raise ValueError("port must be between 0 and 65535")
        if args.top_k <= 0:
            raise ValueError("top-k must be positive")
        subagent = LocalMemorySubagent(
            LocalMemorySubagentConfig(
                root=Path(args.root),
                top_k=args.top_k,
            )
        )
        serve(args.host, args.port, subagent)
        return

    if args.subagent_command == "health":
        client = OSagentMemorySubagentClient(
            OSagentMemorySubagentConfig(base_url=args.url)
        )
        print(json.dumps(client.health(), ensure_ascii=False, indent=2))
        return

    print("Choose a subagent subcommand: serve or health", file=sys.stderr)
    sys.exit(1)


def _handle_track1(args: argparse.Namespace) -> None:
    from os_agent.track1 import run_scenario

    try:
        report = run_scenario(args.scenario, args.output)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise SystemExit(f"Scenario failed: {exc}") from exc
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)




# -------------------------------------------------------------------
# Session handler
# -------------------------------------------------------------------

def _handle_session(args: argparse.Namespace) -> None:
    from os_agent.events.session import SessionRecorder

    recorder = SessionRecorder(getattr(args, "root", "data/sessions"))

    if args.session_command == "list":
        summaries = recorder.list()
        if not summaries:
            print("No persisted sessions found.")
            return
        for s in summaries:
            print(f"{s.session_id}\tevents={s.events}\tlast={s.last_event_type}\t{s.last_timestamp}")
        return

    if args.session_command == "show":
        events = recorder.read(args.session_id)
        if not events:
            print(f"No persisted session found for: {args.session_id}")
            return
        limit = getattr(args, "limit", 0)
        if limit and limit > 0:
            events = events[:limit]
        if getattr(args, "json", False):
            for event in events:
                print(json.dumps(event, ensure_ascii=False))
            return
        for event in events:
            payload = event.get("payload") or {}
            detail = _session_event_detail(str(event.get("event_type") or ""), payload)
            print(f"{event.get('timestamp', '')}\t{event.get('event_type', '')}\t{detail}")
        return

    print("Choose a session subcommand: list or show", file=sys.stderr)
    sys.exit(1)


def _session_event_detail(event_type: str, payload: dict) -> str:
    if event_type in {"message_start", "message_end"}:
        return f"{payload.get('role', '')}:{payload.get('message_id', '')}"
    if event_type in {"tool_call", "tool_result", "tool_call_blocked"}:
        return f"{payload.get('tool_name', '')}:{payload.get('tool_call_id', '')}"
    if event_type == "plan_created":
        return f"planner={payload.get('planner', '')} turn={payload.get('turn_index', '')}"
    if event_type in {"turn_start", "turn_end"}:
        return f"turn={payload.get('turn_index', '')}"
    return ""


# -------------------------------------------------------------------
# Memory handler
# -------------------------------------------------------------------

def _handle_memory(args: argparse.Namespace) -> None:
    from os_agent.memory.maintenance import MemoryMaintenanceService
    from os_agent.memory.markdown_store import MarkdownMemoryStore

    service = MemoryMaintenanceService(getattr(args, "root", "memory"))

    if args.memory_command == "list":
        _memory_list(args)
        return
    if args.memory_command == "pending":
        pending = service.list_pending()
        if not pending:
            print("No pending memory confirmations found.")
            return
        for index, path in enumerate(pending, start=1):
            print(f"{index}. {path}")
        return
    if args.memory_command == "page":
        store = MarkdownMemoryStore(getattr(args, "root", "memory"))
        try:
            page = store.page_memory(
                page=getattr(args, "page", 1),
                page_size=getattr(args, "page_size", 10),
                source=getattr(args, "source", None),
                layer=getattr(args, "layer", None),
                query=getattr(args, "query", None),
                strategy=getattr(args, "strategy", "layered"),
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if getattr(args, "json", False):
            print(json.dumps(page.to_dict(), ensure_ascii=False))
            return
        source = f" source={page.source}" if page.source else ""
        layer = f" layer={page.layer}" if page.layer else ""
        print(
            f"Memory page {page.page}/{page.total_pages} "
            f"items={len(page.items)}/{page.total_items}{source}{layer} "
            f"strategy={page.strategy}"
        )
        for item in page.items:
            preview = " ".join(line.strip() for line in item.content.splitlines() if line.strip())
            print(
                f"{item.index}. [{item.source}] {item.title} "
                f"layer={item.layer} access_count={item.access_count} "
                f"priority={item.priority_score:.3f} relevance={item.relevance_score:.3f}"
            )
            print(f"   {preview[:240]}")
        return
    if args.memory_command == "layers":
        store = MarkdownMemoryStore(getattr(args, "root", "memory"))
        try:
            summary = store.memory_layer_summary(source=getattr(args, "source", None))
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if getattr(args, "json", False):
            print(json.dumps(summary, ensure_ascii=False))
            return
        print(f"hot: {summary.get('hot', 0)}")
        print(f"warm: {summary.get('warm', 0)}")
        print(f"cold: {summary.get('cold', 0)}")
        print(f"total: {summary.get('total', 0)}")
        return
    if args.memory_command == "search":
        store = MarkdownMemoryStore(getattr(args, "root", "memory"))
        try:
            if args.mode == "keyword":
                matches = store.retrieve(
                    args.query,
                    limit=getattr(args, "limit", 5),
                    source=getattr(args, "source", None),
                )
            elif args.mode == "vector":
                matches = store.retrieve_vector(
                    args.query,
                    limit=getattr(args, "limit", 5),
                    source=getattr(args, "source", None),
                )
            else:
                matches = store.retrieve_hybrid(
                    args.query,
                    limit=getattr(args, "limit", 5),
                    source=getattr(args, "source", None),
                )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if getattr(args, "json", False):
            print(json.dumps([match.to_dict() for match in matches], ensure_ascii=False))
            return
        if not matches:
            print("No matching memories found.")
            return
        for index, match in enumerate(matches, start=1):
            print(
                f"{index}. [{match.source}] {match.title} "
                f"({match.retrieval_method}, score={match.score})"
            )
            print(f"   {match.excerpt}")
        return
    if args.memory_command == "maintain":
        report = service.maintain(
            max_sections_per_file=getattr(args, "max_sections_per_file", 80),
        )
        print(f"Compacted files:     {report.compacted_files}")
        print(f"Removed duplicates:  {report.removed_duplicates}")
        print(f"Archived sections:   {report.archived_sections}")
        print(f"Pending items:       {report.pending_items}")
        return

    print("Choose a memory subcommand: list, pending, page, layers, search, or maintain", file=sys.stderr)
    sys.exit(1)


def _memory_list(args: argparse.Namespace) -> None:
    import re

    from os_agent.memory.markdown_store import MarkdownMemoryStore

    store = MarkdownMemoryStore(getattr(args, "root", "memory"))
    file_labels = {
        "profile.md": "偏好", "project-knowledge.md": "知识", "workflows.md": "流程",
        "tool-history.md": "工具历史", "safety.md": "安全",
    }
    total = 0
    for field, filename in MarkdownMemoryStore.CORE_FILES.items():
        path = Path(store.root) / filename
        text = path.read_text(encoding="utf-8", errors="replace")
        sections = store._iter_sections(text, field)
        label = file_labels.get(filename, filename)
        print(f"\n=== {label} ({filename}) ===")
        if not any(line.strip().startswith("## ") for line in text.splitlines()):
            print("  (空)")
            continue
        for title, body in sections:
            total += 1
            conf = re.search(r"confidence:\s*([\d.]+)", body)
            c = conf.group(1) if conf else "?"
            ti = store.tier_info(filename, title)
            tier_label = ti["tier"]
            ws = ti["weighted_score"]
            ac = ti["access_count"]
            access_str = f" 权重{ws} 共{ac}次" if ac else ""
            content_preview = ""
            for line in body.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                if re.match(r"^\w[\w_]*:\s", stripped):
                    continue
                content_preview = stripped[:100]
                break
            if not content_preview:
                content_preview = " ".join(
                    l.strip() for l in body.splitlines()
                    if l.strip() and not re.match(r"^\w[\w_]*:\s", l.strip())
                )[:100]
            print(f"  {total}. [{tier_label}] {title}  (conf={c}{access_str})")
            if content_preview:
                print(f"     {content_preview}")

    # pending
    pending = list(store.pending.glob("*.md"))
    if pending:
        print(f"\n=== 待确认 ({len(pending)} 条) ===")
        for pf in sorted(pending):
            text = pf.read_text(encoding="utf-8", errors="replace")
            for title, body in store._iter_sections(text, "pending"):
                total += 1
                type_ = re.search(r"type:\s*(\S+)", body)
                t = type_.group(1) if type_ else "?"
                source = f"pending-confirmation/{pf.name}"
                ti = store.tier_info(source, title)
                tier_label = ti["tier"]
                ws = ti["weighted_score"]
                ac = ti["access_count"]
                access_str = f" 权重{ws} 共{ac}次" if ac else ""
                content_preview = ""
                for line in body.splitlines():
                    stripped = line.strip()
                    if not stripped or re.match(r"^\w[\w_]*:\s", stripped):
                        continue
                    content_preview = stripped[:100]
                    break
                print(f"  {total}. [{tier_label}] [{t}] {title}{access_str}")
                if content_preview:
                    print(f"     {content_preview}")
    print(f"\n共 {total} 条记忆")


# -------------------------------------------------------------------
# Config handler
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Core (Mem0) handler
# -------------------------------------------------------------------

def _handle_core(args: argparse.Namespace) -> None:
    from os_agent.memory.mem0_core import CoreMemoryStore

    store = CoreMemoryStore(getattr(args, "root", "memory"))

    if args.core_command == "list":
        facts = store.list_by_entity(args.entity) if args.entity else []
        if not args.entity:
            # List all active facts
            snapshot = store.snapshot()
            facts_data = snapshot["facts"]
            print(f"\nCore Memory — {snapshot['total_active']} active facts")
            print("-" * 70)
            for item in facts_data:
                val = str(item["value"])
                print(f"  [{item['entity']}] {item['attribute']} = {val[:80]}")
                print(f"    v{item['version']}  conf={item['confidence']:.2f}  tags={item['tags']}")
            return
        if not facts:
            print(f"No facts found for entity '{args.entity}'.")
            return
        print(f"\nCore facts for '{args.entity}':")
        for f in facts:
            print(f"  {f.attribute} = {f.value}  (v{f.version}, conf={f.confidence:.2f})")
        return

    if args.core_command == "search":
        facts = store.search(args.query, top_k=args.top_k or 10, entity_filter=args.entity)
        if not facts:
            print(f"No Core facts match '{args.query}'.")
            return
        print(f"\nCore search: '{args.query}' → {len(facts)} results")
        for f in facts:
            print(f"  [{f.entity}] {f.attribute} = {str(f.value)[:100]}")
            print(f"    v{f.version}  conf={f.confidence:.2f}  evidence: {f.evidence[:80]}")
        return

    if args.core_command == "stats":
        snapshot = store.snapshot()
        entities: dict[str, int] = {}
        total_conf = 0.0
        for f in snapshot["facts"]:
            e = f["entity"]
            entities[e] = entities.get(e, 0) + 1
            total_conf += f["confidence"]
        print("\nCore Memory Stats")
        print(f"  Active facts: {snapshot['total_active']}")
        if snapshot["total_active"]:
            print(f"  Avg confidence: {total_conf / snapshot['total_active']:.2f}")
        print(f"  Entities: {len(entities)}")
        for e, c in sorted(entities.items(), key=lambda x: -x[1]):
            print(f"    {e}: {c} facts")
        return

    print("Choose a core subcommand: list, search, or stats", file=sys.stderr)
    sys.exit(1)


# -------------------------------------------------------------------
# Pipeline (Mem0) handler
# -------------------------------------------------------------------

def _handle_pipeline(args: argparse.Namespace) -> None:
    from os_agent.memory.mem0_archival import ArchivalMemoryStore
    from os_agent.memory.mem0_core import CoreMemoryStore
    from os_agent.memory.mem0_pipeline import ExtractUpdatePipeline

    if args.pipeline_command == "run":
        core = CoreMemoryStore(getattr(args, "root", "memory"))
        archival = ArchivalMemoryStore(getattr(args, "root", "memory"))
        pipeline = ExtractUpdatePipeline(core, archival)

        if args.all:
            session_dir = Path("data/sessions")
            if not session_dir.exists():
                print("No sessions directory found.")
                return
            total = 0
            for jsonl_file in sorted(session_dir.glob("*.jsonl")):
                facts = pipeline.process_jsonl(jsonl_file)
                total += len(facts)
                print(f"  {jsonl_file.name}: {len(facts)} facts")
            print(f"\nPipeline complete — {total} facts from all sessions.")
            return

        if args.session:
            path = Path(f"data/sessions/{args.session}.jsonl")
            if not path.exists():
                print(f"Session not found: {args.session}")
                return
            facts = pipeline.process_jsonl(path)
            print(f"\nPipeline complete — {len(facts)} facts from {args.session}")
            for f in facts:
                print(f"  [{f.entity}] {f.attribute} = {str(f.value)[:100]}")
            return

        print("Use --session <id> or --all", file=sys.stderr)
        sys.exit(1)

    if args.pipeline_command == "status":
        core = CoreMemoryStore(getattr(args, "root", "memory"))
        snapshot = core.snapshot()
        print("Pipeline Status")
        print(f"  Core facts: {snapshot['total_active']}")
        # Check session count
        session_dir = Path("data/sessions")
        session_count = len(list(session_dir.glob("*.jsonl"))) if session_dir.exists() else 0
        print(f"  Sessions ready for processing: {session_count}")
        return

    print("Choose a pipeline subcommand: run or status", file=sys.stderr)
    sys.exit(1)


# -------------------------------------------------------------------
# Config handler
# -------------------------------------------------------------------

def _handle_config(args: argparse.Namespace) -> None:
    config_file = getattr(args, "config_file", "config/llm.local.json")

    if args.config_command == "show":
        _config_show(config_file)
        return
    if args.config_command == "set":
        _config_set(config_file, args)
        return

    print("Choose a config subcommand: show or set", file=sys.stderr)
    sys.exit(1)


def _resolve_config_path(config_file: str) -> Path:
    """Resolve config file path, trying relative then package-adjacent locations."""
    path = Path(config_file)
    if path.exists():
        return path.resolve()

    # If relative and not found, try resolving from the package directory
    if not path.is_absolute():
        package_dir = Path(__file__).resolve().parent  # src/os_agent/
        # Try one level up from the package (project root)
        for candidate in [
            Path.cwd() / path,
            package_dir.parent.parent / path,  # project root: src/os_agent -> project
        ]:
            resolved = candidate.resolve()
            if resolved.exists():
                return resolved

    return path.resolve()


def _config_show(config_file: str) -> None:
    path = _resolve_config_path(config_file)
    print(f"Config file: {path}")

    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  (unreadable: {exc})")
            data = {}
    else:
        print("  ⚠ 文件不存在")
        print("  请在项目目录下运行，或使用 --config-file 指定路径。")
        print("  示例: os-agent config show --config-file C:\\path\\to\\config\\llm.local.json")
        data = {}

    keys = ["api_key", "base_url", "model", "timeout_seconds", "temperature"]
    print()
    print(f"{'Key':<20} {'File':<40} {'Env override':<40}")
    print("-" * 100)

    env_map = {
        "api_key":         ("OPENAI_API_KEY", "OS_AGENT_API_KEY"),
        "base_url":        ("OPENAI_BASE_URL", "OS_AGENT_BASE_URL"),
        "model":           ("OPENAI_MODEL", "OS_AGENT_MODEL"),
        "timeout_seconds": ("OS_AGENT_TIMEOUT_SECONDS",),
        "temperature":     ("OS_AGENT_TEMPERATURE",),
    }

    for key in keys:
        file_val = data.get(key, "")
        if key == "api_key" and file_val:
            masked = file_val[:6] + "…" + file_val[-4:] if len(file_val) > 10 else "***"
            file_display = masked
        else:
            file_display = str(file_val) if file_val else "(not set)"

        env_display = ""
        for env_name in env_map.get(key, ()):
            env_val = os.getenv(env_name)
            if env_val is not None:
                if key == "api_key":
                    env_display = f"{env_name}={env_val[:6]}…{env_val[-4:]}" if len(env_val) > 10 else f"{env_name}=***"
                else:
                    env_display = f"{env_name}={env_val}"
                break

        if not env_display and key in ("api_key", "base_url"):
            env_display = "(not set)"
        elif not env_display:
            env_display = ""

        print(f"{key:<20} {file_display:<40} {env_display:<40}")


def _config_set(config_file: str, args: argparse.Namespace) -> None:
    path = _resolve_config_path(config_file) if Path(config_file).exists() else Path(config_file).resolve()

    # If relative and doesn't exist, prefer project root for new files
    if not Path(config_file).is_absolute() and not path.exists():
        package_dir = Path(__file__).resolve().parent
        project_root = package_dir.parent.parent
        candidate = (project_root / config_file).resolve()
        candidate.parent.mkdir(parents=True, exist_ok=True)
        path = candidate
        print(f"配置文件将创建在: {path}")

    # Load existing
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Error reading {path}: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {}

    fields = [
        ("api_key",         args.api_key),
        ("base_url",        args.base_url),
        ("model",           args.model),
        ("timeout_seconds", args.timeout),
        ("temperature",     args.temperature),
    ]

    updated = False
    for json_key, arg_val in fields:
        if arg_val is not None:
            data[json_key] = arg_val
            updated = True
            print(f"  {json_key} -> {'***' if json_key == 'api_key' else str(arg_val)}")

    if not updated:
        print("No values to update.  Use --api-key, --base-url, --model, etc.")
        print("Run 'os-agent config show' to see current settings.")
        sys.exit(1)

    # Write back
    try:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nConfig saved to {path.resolve()}")
    except OSError as exc:
        print(f"Error writing {path}: {exc}", file=sys.stderr)
        sys.exit(1)


# -------------------------------------------------------------------
# Memory config handler
# -------------------------------------------------------------------

def _handle_memory_config(args: argparse.Namespace) -> None:
    from os_agent.memory.memory_config import MemoryConfig, parse_tuning_value

    cfg = MemoryConfig(getattr(args, "config_file", None))

    if args.memory_config_command == "show":
        values = cfg.all()
        prefix = getattr(args, "prefix", None)
        if prefix:
            values = {key: value for key, value in values.items() if key.startswith(prefix)}
        if getattr(args, "json", False):
            print(json.dumps(values, ensure_ascii=False, indent=2))
            return
        print(f"Memory config: {cfg.path}")
        for key in sorted(values):
            desc = cfg.describe(key)
            print(f"{key:<48} {values[key]!s:<12} {desc}")
        return

    if args.memory_config_command == "set":
        value = parse_tuning_value(args.value)
        try:
            cfg.set_value(args.key, value)
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            print("Run 'os-agent memory-config show' to list valid keys.", file=sys.stderr)
            sys.exit(1)
        path = cfg.save()
        MemoryConfig.reset_global()
        print(f"{args.key} -> {value}")
        print(f"Memory config saved to {path}")
        return

    print("Choose a memory-config subcommand: show or set", file=sys.stderr)
    sys.exit(1)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
