"""End-to-end public scenario replay and isolation regressions."""
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from os_agent.track1 import run_scenario

SCENARIO = Path(__file__).resolve().parents[1] / "examples/track1/maintenance-handover.json"


def test_real_lifecycle_reopens_and_physically_deletes(tmp_path):
    # Even a user's enabled remote embedding configuration must not be used.
    with patch.dict(os.environ, {"OS_AGENT_EMBEDDING_BACKEND": "openai",
                                 "OS_AGENT_API_KEY": "not-a-real-key"}), \
         patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")):
        report = run_scenario(SCENARIO, tmp_path)
    assert report["passed"]
    assert report["checks_passed"] == report["checks_total"] == 10
    run = tmp_path / report["run_directory"]
    assert json.loads((run / "report.json").read_text())["passed"]
    assert "输入、输出与检查证据" in (run / "report.html").read_text()
    core = json.loads((run / "memory/core_facts.json").read_text())
    # Forget must preserve the independent second device's current state.
    assert any(f["value"] == "SOP-B1" for f in core)
    assert all(f["value"] not in ("SOP-A1", "SOP-A2") for f in core)


def test_repeated_runs_do_not_reuse_state(tmp_path):
    first = run_scenario(SCENARIO, tmp_path)
    sentinel = tmp_path / first["run_directory"] / "do-not-touch.txt"
    sentinel.write_text("existing run")
    second = run_scenario(SCENARIO, tmp_path)
    assert first["run_directory"] != second["run_directory"]
    assert second["passed"]
    assert sentinel.read_text() == "existing run"


def test_failed_expectation_cannot_be_reported_as_success(tmp_path):
    scenario = json.loads(SCENARIO.read_text())
    scenario["steps"][3]["checks"][0]["value"] = "nonexistent-procedure"
    source = tmp_path / "scenario.json"
    source.write_text(json.dumps(scenario))
    report = run_scenario(source, tmp_path / "output")
    assert not report["passed"]
    assert report["checks_passed"] < report["checks_total"]


def test_report_escapes_untrusted_scenario_text(tmp_path):
    scenario = json.loads(SCENARIO.read_text())
    scenario["name"] = '<script>alert("x")</script>'
    source = tmp_path / "scenario.json"
    source.write_text(json.dumps(scenario))
    report = run_scenario(source, tmp_path / "output")
    page = (tmp_path / "output" / report["run_directory"] / "report.html").read_text()
    assert "<script>" not in page
    assert "&lt;script&gt;" in page


def test_unknown_action_is_rejected(tmp_path):
    source = tmp_path / "scenario.json"
    source.write_text(json.dumps({"schema_version": 1, "name": "invalid", "steps": [{"action": "shell"}]}))
    with pytest.raises(ValueError, match="unsupported scenario action"):
        run_scenario(source, tmp_path / "output")
