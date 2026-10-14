"""Portable, offline scenario replay through the real memory facade.

The input is a public simulation, not a benchmark or an LLM quality score.
No local credentials or system SDKs are loaded by this profile.
"""
from __future__ import annotations

import html
import json
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from os_agent.memory.embedding_provider import EmbeddingConfig, OpenAICompatibleEmbeddingProvider
from os_agent.memory.interface import ForgetRequest, RecallRequest, RememberRequest
from os_agent.memory.memory_config import MemoryConfig
from os_agent.memory.module import AgentMemoryModule


def _memory(root: Path, config_path: Path) -> AgentMemoryModule:
    return AgentMemoryModule(
        root, config=MemoryConfig(config_path),
        embedding_provider=OpenAICompatibleEmbeddingProvider(EmbeddingConfig()),
    )


def run_scenario(scenario_path: Path, output: Path) -> dict:
    """Replay an explicit synthetic workflow in a new, isolated directory."""
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    if scenario.get("schema_version") != 1 or not isinstance(scenario.get("steps"), list):
        raise ValueError("scenario must contain schema_version=1 and steps")
    output.mkdir(parents=True, exist_ok=True)
    run_root = Path(tempfile.mkdtemp(prefix="run-", dir=output))
    config_path = run_root / "memory-config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    root = run_root / "memory"
    memory = _memory(root, config_path)
    trace = []
    for step in scenario["steps"]:
        action = step["action"]
        arguments = step.get("arguments", {})
        started = perf_counter()
        if action == "remember":
            result = memory.remember(RememberRequest(**arguments)).to_dict()
        elif action == "recall":
            result = memory.recall(RecallRequest(**arguments)).to_dict()
        elif action == "forget":
            result = memory.forget(ForgetRequest(**arguments)).to_dict()
        elif action == "restart":
            memory = _memory(root, config_path)
            result = {"ok": True, "message": "reopened persisted memory"}
        elif action == "history":
            result = {"ok": True, "records": [f.to_dict() for f in memory.archival.get_fact_history(**arguments)]}
        else:
            raise ValueError(f"unsupported scenario action: {action}")
        checks = []
        for check in step.get("checks", []):
            values = [record.get("value") for record in result.get("records", [])]
            if check["kind"] == "contains_value":
                passed = check["value"] in values
            elif check["kind"] == "excludes_value":
                passed = check["value"] not in values
            elif check["kind"] == "core_missing":
                passed = memory.core.get(check["entity"], check["attribute"]) is None
            else:
                raise ValueError(f"unsupported check: {check['kind']}")
            checks.append({**check, "passed": passed})
        trace.append({"title": step["title"], "action": action, "arguments": arguments,
                      "elapsed_ms": round((perf_counter() - started) * 1000, 3),
                      "result": result, "checks": checks,
                      "passed": result["ok"] == step.get("expected_ok", True) and all(c["passed"] for c in checks)})
    checks = [check for step in trace for check in step["checks"]]
    report = {
        "schema_version": 1, "scenario": scenario["name"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "profile": "offline lexical retrieval; no LLM, embeddings or Kylin SDK",
        "data_kind": "public synthetic scenario; not customer data or quality benchmark",
        "run_directory": run_root.name,
        "passed": bool(checks) and all(c["passed"] for c in checks) and all(s["passed"] for s in trace),
        "checks_passed": sum(c["passed"] for c in checks), "checks_total": len(checks), "steps": trace,
    }
    (run_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_root / "report.html").write_text(render_report(report), encoding="utf-8")
    return report


def render_report(report: dict) -> str:
    escape = lambda value: html.escape(str(value))
    cards = []
    for step in report["steps"]:
        status = "通过" if step["passed"] else "失败"
        cards.append(f'<article><h2>{escape(step["title"])} · {status}</h2>'
                     f'<p>{escape(step["action"])} · {step["elapsed_ms"]} ms</p>'
                     f'<details><summary>输入、输出与检查证据</summary><pre>{escape(json.dumps(step, ensure_ascii=False, indent=2))}</pre></details></article>')
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>OSagent 行业场景验证</title><style>'
            'body{font:16px/1.7 system-ui;background:#f4f6fa;color:#192437;max-width:1000px;margin:40px auto;padding:20px}'
            'article{background:white;padding:20px;margin:16px 0;border-radius:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere}'
            'h1{line-height:1.3}summary{cursor:pointer}</style>'
            f'<h1>{escape(report["scenario"])}</h1>'
            f'<p>检查通过 {report["checks_passed"]}/{report["checks_total"]} · 总体：{"通过" if report["passed"] else "失败"}</p>'
            '<p>公开合成的制造运维交接场景。调用真实记忆模块，展示持久化、更新、历史证据、检索与遗忘；'
            '使用离线词法检索，不代表自然语言抽取准确率、语义检索质量或生产部署验收。</p>'
            f'<p>{escape(report["generated_at"])} · {escape(report["environment"])}</p>'
            + ''.join(cards) + '</html>')
