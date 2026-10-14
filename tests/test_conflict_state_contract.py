from copy import deepcopy

import pytest

from benchmarks.conflict_state_contract import VERSION, score_states


def record(**changes):
    return {"entity": "reader", "scope": "studio", "attribute": "display.theme",
            "value": "night", "status": "active", **changes}


def annotation():
    return {"schema_version": VERSION, "attribute_aliases": {"appearance": "display.theme"},
            "cases": [{"case_id": "independent-fixture", "expected_state": [record()]}]}


def score(rows):
    return score_states({"independent-fixture": {"natural_state": rows}}, annotation())


def test_equivalent_schema_and_embedded_scope():
    assert score([record(entity="reader::studio", attribute="appearance")])["accuracy"] == 1


@pytest.mark.parametrize("changes", [
    {"scope": "workshop"}, {"entity": "another-reader"}, {"value": "day"},
    {"status": "superseded"}, {"attribute": "display.contrast"},
    {"value": "day", "evidence": "previously night"},
])
def test_wrong_fields_fail(changes):
    assert score([record(**changes)])["accuracy"] == 0


def test_duplicates_and_extras_fail():
    assert score([record(), record()])["accuracy"] == 0
    assert score([record(), record(value="day")])["accuracy"] == 0


def test_missing_and_invalid_are_not_silently_excluded():
    assert score_states({}, annotation())["missing_predictions"] == 1
    assert score([{"value": "night"}])["invalid_predictions"] == 1
    assert score([record(entity="reader::elsewhere")])["invalid_predictions"] == 1
    assert score("not a list")["accuracy"] == 0


def test_duplicate_annotation_ids_are_rejected():
    labels = annotation()
    labels["cases"].append(deepcopy(labels["cases"][0]))
    with pytest.raises(ValueError):
        score_states({}, labels)


def test_value_types_and_punctuation_are_preserved():
    labels = annotation()
    labels["cases"][0]["expected_state"][0]["value"] = "1.5"
    assert score_states({"independent-fixture": {"natural_state": [record(value="15")]}}, labels)["accuracy"] == 0
    assert score_states({"independent-fixture": {"natural_state": [record(value=1.5)]}}, labels)["accuracy"] == 0
