from __future__ import annotations

import subprocess

from benchmarks.memory_eval_pipeline import (
    build_report,
    nested_value,
    publish_guard,
    render_command,
    select_jobs,
    threshold_passed,
)


def test_nested_metric_and_thresholds() -> None:
    payload = {"summary": {"recall": {"p95": 12.5}}}
    assert nested_value(payload, "summary.recall.p95") == 12.5
    assert threshold_passed(0.9, {"min": 0.85}) is True
    assert threshold_passed(501, {"max": 500}) is False
    assert threshold_passed(1, None) is None


def test_explicit_selection_includes_optional_live_job() -> None:
    config = {
        "jobs": [
            {"id": "default", "enabled": True},
            {"id": "live", "enabled": False},
        ]
    }

    assert [job["id"] for job in select_jobs(config, set())] == ["default"]
    assert [job["id"] for job in select_jobs(config, {"live"})] == ["live"]


def test_render_command_exposes_comparison_runtime_paths(tmp_path) -> None:
    rendered = render_command(
        ["{bun}", "{oh_my_pi_runtime}"],
        root=tmp_path,
        output=tmp_path / "result.json",
    )

    assert "comparison-runtimes" in rendered[0]
    assert rendered[0].endswith(("bun", "bun.cmd"))
    assert rendered[1].endswith("oh-my-pi-mnemopi-17.3.4-fts")


def test_publish_guard_rejects_clean_tracked_third_party_source(tmp_path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    forbidden = tmp_path / "data" / "benchmark-sources" / "upstream.txt"
    forbidden.parent.mkdir(parents=True)
    forbidden.write_text("third-party source", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=OSagent Test",
            "-c",
            "user.email=osagent@example.invalid",
            "commit",
            "-m",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    result = publish_guard(
        tmp_path,
        [],
        ["data/benchmark-sources"],
    )

    assert not result["passed"]
    assert result["violations"] == ["data/benchmark-sources/upstream.txt"]


def test_report_separates_competition_and_diagnostic_metrics() -> None:
    manifest = {
        "evaluated_at": "2026-07-26T00:00:00+00:00",
        "publish_guard": {"passed": True, "violations": []},
        "jobs": [
            {
                "system": "OS Agent",
                "dataset": "competition",
                "dataset_revision": "v1",
                "evaluation_role": "regression",
                "metric_scope": "hard metrics",
                "metrics": [
                    {
                        "label": "Recall@5",
                        "value": 1.0,
                        "unit": "percent",
                        "passed": True,
                        "competition_metric": True,
                    },
                    {
                        "label": "coverage",
                        "value": 0.4,
                        "unit": "percent",
                        "passed": None,
                        "competition_metric": False,
                    },
                ],
            }
        ],
    }

    report = build_report(manifest)

    hard_section, diagnostic_section = report.split("## 外部数据集诊断")
    assert "Recall@5" in hard_section
    assert "coverage" not in hard_section
    assert "coverage" in diagnostic_section
