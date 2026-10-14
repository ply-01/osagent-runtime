from __future__ import annotations

import json
from pathlib import Path

from os_agent.eval.runner import ADAPTERS, run_benchmark


def test_eval_runner_supports_cxx_style_adapters(tmp_path: Path) -> None:
    payload = {
        "history": [
            {"role": "user", "content": "我喜欢导出 PDF"},
            {"role": "user", "content": "我更希望默认导出 Markdown"},
        ],
        "test_cases": [
            {
                "query": "帮我导出文档",
                "expected_preference": "Markdown",
                "preference_category": "export_format",
                "is_updated": True,
                "old_preference": "PDF",
            }
        ],
    }
    data_path = tmp_path / "perma.json"
    data_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    report = run_benchmark("perma", data_path, memory_root=str(tmp_path / "memory"))

    assert report.total == 1
    assert report.results[0].query == "帮我导出文档"
    assert "facts_created" in report.metrics
    assert set(ADAPTERS) == {
        "mem2act",
        "perma",
        "memory_agent_bench",
        "memora",
        "longmemeval_v2",
    }

