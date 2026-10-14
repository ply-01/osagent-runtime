from __future__ import annotations

from os_agent.memory.interface import MemoryRecord

from benchmarks.public_memory_benchmarks_eval import (
    chunk_text,
    flatten_strings,
    support_rank,
    trajectory_summary,
)


def test_support_rank_scores_after_recall_without_labels_in_records() -> None:
    records = [
        MemoryRecord(memory_id="one", scope="ltm", content="irrelevant"),
        MemoryRecord(memory_id="two", scope="ltm", content="The current folder is reports/final."),
    ]

    assert support_rank(["reports/final"], records) == 2
    assert support_rank(["hidden gold value"], records) is None


def test_chunk_text_is_deterministic_and_overlapping() -> None:
    text = "A" * 2000 + "\n" + "B" * 2000
    first = chunk_text(text, max_chars=1000, overlap=100)
    second = chunk_text(text, max_chars=1000, overlap=100)

    assert first == second
    assert len(first) > 2
    assert all(len(item) <= 1000 for item in first)


def test_flatten_strings_ignores_numeric_scoring_metadata() -> None:
    payload = {"facts": ["keep this", {"value": "and this"}], "count": 2}

    assert flatten_strings(payload) == ["keep this", "and this"]


def test_trajectory_summary_excludes_screenshot_payloads() -> None:
    summary = trajectory_summary(
        {
            "goal": "Export the report",
            "outcome": "success",
            "states": [
                {
                    "step": 1,
                    "action": "click export",
                    "thought": "use the menu",
                    "accessibility_tree": "Export button",
                    "screenshot": "secret-image-path.png",
                }
            ],
        }
    )

    assert "Export button" in summary
    assert "secret-image-path.png" not in summary
