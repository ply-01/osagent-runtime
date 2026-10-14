import json
from pathlib import Path

from benchmarks.mem0_kylin_conflict_eval import (
    score_natural_conflicts,
    session_messages,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_session_messages_exposes_only_user_message_content() -> None:
    session = [
        {
            "event_type": "message_start",
            "payload": {"role": "user", "content": "  prefers cedar shelves  "},
        },
        {
            "event_type": "message_start",
            "payload": {"role": "assistant", "content": "acknowledged"},
        },
        {"event_type": "tool_result", "payload": {"content": "private output"}},
        {"event_type": "message_start", "payload": {"role": "user", "content": ""}},
    ]

    assert session_messages(session) == [
        {"role": "user", "content": "prefers cedar shelves"}
    ]


def test_conflict_scorer_uses_existing_match_contract_and_returns_only_aggregates(
    tmp_path: Path,
) -> None:
    labels = tmp_path / "labels.jsonl"
    inputs = tmp_path / "inputs.jsonl"
    _write_jsonl(
        labels,
        [
            {
                "case_id": "one",
                "expected_records": [["shelf", "cedar"]],
                "forbidden_records": [["shelf", "pine"]],
            },
            {
                "case_id": "two",
                "expected_records": [["lamp", "amber"]],
                "forbidden_records": [],
            },
        ],
    )
    _write_jsonl(
        inputs,
        [
            {"case_id": "one", "category": "change"},
            {"case_id": "two", "category": "independent"},
        ],
    )

    result = score_natural_conflicts(
        {
            "one": ["current shelf material cedar"],
            "two": ["unrelated memory"],
        },
        labels_path=labels,
        inputs_path=inputs,
    )

    assert result == {
        "cases": 2,
        "natural_ingress_accuracy": 0.5,
        "by_category": {
            "change": {"cases": 1, "natural_ingress_accuracy": 1.0},
            "independent": {"cases": 1, "natural_ingress_accuracy": 0.0},
        },
    }
    assert "case_id" not in json.dumps(result)
    assert "cedar" not in json.dumps(result)
