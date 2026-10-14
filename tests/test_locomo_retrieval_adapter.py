from __future__ import annotations

from benchmarks.locomo_retrieval_eval import dialogue_facts, select_cases


def test_locomo_adapter_keeps_evidence_ids_out_of_queries() -> None:
    sample = {
        "sample_id": "sample-1",
        "conversation": {
            "session_1": [
                {"speaker": "A", "dia_id": "D1:1", "text": "I use Linux."}
            ],
            "session_1_date_time": "2023-01-01",
        },
        "qa": [
            {
                "question": "Which operating system do I use?",
                "answer": "Linux",
                "evidence": ["D1:1"],
                "category": 1,
            }
        ],
    }

    facts = dialogue_facts(sample)
    cases = select_cases([sample], limit=1, seed=7)

    assert facts[0].source_event_id == "D1:1"
    assert facts[0].value == "I use Linux."
    assert cases[0].question == "Which operating system do I use?"
    assert "D1:1" not in cases[0].question
