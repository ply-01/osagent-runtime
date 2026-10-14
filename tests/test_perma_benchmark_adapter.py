from __future__ import annotations

from dataclasses import replace

from benchmarks.perma_memory_eval import (
    CaseResult,
    PermaCase,
    parse_date,
    parse_label,
    public_case_result,
    select_cases,
    session_events,
    split_options,
)


def test_parse_perma_date_and_label() -> None:
    assert parse_date("2023-03-05 Evening").isoformat() == "2023-03-05"
    assert parse_label(" C\n") == "C"
    assert parse_label("The answer is H.") == "H"
    assert parse_label("unknown") == ""


def test_split_options_preserves_multiline_content() -> None:
    options = "A: first line\ncontinued\nB: second\nC: third"
    assert split_options(options) == {
        "A": "first line\ncontinued",
        "B": "second",
        "C": "third",
    }


def test_session_mapping_keeps_labels_out_of_events() -> None:
    session = {
        "date": "2023-04-20 Morning",
        "task_id": "event_1",
        "conversation": [
            {"role": "user", "content": "I prefer a quiet alarm."},
            {"role": "assistant", "content": "Understood."},
        ],
        "gold_label": "A",
    }
    events = session_events(session, "108")
    assert len(events) == 2
    assert events[0]["payload"]["content"] == "I prefer a quiet alarm."
    assert "gold_label" not in str(events)


def test_case_sampling_does_not_depend_on_gold_label() -> None:
    cases = [
        PermaCase(
            user_id=str(index % 2),
            task_id=f"task-{index}",
            question="question",
            question_date="2023-01-01",
            options="A: one\nB: two",
            gold_label="A" if index % 2 else "B",
            source_path=f"case-{index}.json",
        )
        for index in range(10)
    ]
    first = select_cases(cases, limit=4, seed=7)
    relabeled = [replace(case, gold_label="H") for case in cases]
    second = select_cases(relabeled, limit=4, seed=7)
    assert [case.task_id for case in first] == [case.task_id for case in second]


def test_public_case_result_omits_dataset_and_retrieval_text() -> None:
    row = CaseResult(
        user_id="user",
        task_id="task",
        question_date="2023-01-01",
        question="private question",
        source_path="private/source.json",
        gold_label="A",
        predicted_label="A",
        retrieved=["private recalled text"],
        correct=True,
    )

    public = public_case_result(row)

    assert public["correct"] is True
    assert set(public).isdisjoint(
        {
            "user_id",
            "task_id",
            "question_date",
            "question",
            "source_path",
            "gold_label",
            "predicted_label",
            "no_memory_label",
            "retrieved",
        }
    )
