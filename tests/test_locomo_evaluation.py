from __future__ import annotations

from benchmarks.locomo_retrieval_eval import CaseResult, summarize_results
from benchmarks.oh_my_pi_locomo_retrieval_eval import (
    MNEMOPI_PACKAGE_INTEGRITY,
    _package_metadata,
)


def test_shared_locomo_summary_uses_first_evidence_rank() -> None:
    rows = [
        CaseResult("s1", 0, 1, 1, ["a"], 1, 2.0),
        CaseResult("s1", 1, 1, 2, ["x", "b"], 4, 4.0),
        CaseResult("s2", 0, 5, 1, ["x"], None, 10.0),
    ]

    summary = summarize_results(rows, [20.0, 40.0], conversations=2)

    assert summary["recall_at_1"] == 0.3333
    assert summary["recall_at_3"] == 0.3333
    assert summary["recall_at_5"] == 0.6667
    assert summary["mrr_at_5"] == 0.4167
    assert summary["recall_latency_ms"]["p50"] == 4.0
    assert summary["indexing_ms"]["p50"] == 30.0
    assert summary["by_category"]["1"]["recall_at_5"] == 1.0
    assert summary["by_category"]["5"]["recall_at_5"] == 0.0


def test_mnemopi_runtime_metadata_comes_from_installed_lockfile(tmp_path) -> None:
    package = tmp_path / "node_modules" / "@oh-my-pi" / "pi-mnemopi"
    package.mkdir(parents=True)
    (package / "package.json").write_text(
        '{"version":"17.3.4"}',
        encoding="utf-8",
    )
    (tmp_path / "package-lock.json").write_text(
        "{\"packages\":{\"node_modules/@oh-my-pi/pi-mnemopi\":{"
        f"\"version\":\"17.3.4\",\"integrity\":\"{MNEMOPI_PACKAGE_INTEGRITY}\""
        "}}}",
        encoding="utf-8",
    )

    version, integrity = _package_metadata(tmp_path)

    assert version == "17.3.4"
    assert integrity == MNEMOPI_PACKAGE_INTEGRITY
