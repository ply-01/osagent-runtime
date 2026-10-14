"""Versioned strict state scoring. No production parser or dataset vocabulary."""
from collections import Counter
import json

VERSION = "natural-conflict-state-v2-draft1"
FIELDS = ("entity", "scope", "attribute", "value", "status")


def state_key(row, aliases):
    if not isinstance(row, dict) or any(key not in row for key in FIELDS):
        raise ValueError("incomplete state record")
    if any(not isinstance(row[key], str) for key in FIELDS if key != "value"):
        raise ValueError("state identity fields must be strings")
    entity, scope = row["entity"], row["scope"]
    if "::" in entity:
        owner, embedded = entity.split("::", 1)
        if scope and scope != embedded:
            raise ValueError("contradictory embedded scope")
        entity, scope = owner, embedded
    attribute = aliases.get(row["attribute"], row["attribute"])
    # Preserve punctuation, case, JSON types and duplicate counts.
    value = json.dumps(row["value"], sort_keys=True, ensure_ascii=False, allow_nan=False)
    return entity, scope, attribute, value, row["status"]


def validate_annotation(annotation):
    if annotation.get("schema_version") != VERSION:
        raise ValueError("unsupported annotation version")
    aliases = annotation.get("attribute_aliases", {})
    if not isinstance(aliases, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in aliases.items()):
        raise ValueError("invalid alias vocabulary")
    if any(v in aliases and aliases[v] != v for v in aliases.values()):
        raise ValueError("alias chains are not allowed")
    cases = annotation.get("cases", [])
    ids = [case["case_id"] for case in cases]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("empty or duplicate annotation identifiers")
    for case in cases:
        keys = [state_key(row, aliases) for row in case["expected_state"]]
        if len(set(key[:3] for key in keys)) != len(keys):
            raise ValueError("duplicate expected active slot")
        if any(key[-1] != "active" for key in keys):
            raise ValueError("current state labels must be active")


def score_states(predictions, annotation):
    validate_annotation(annotation)
    aliases = annotation.get("attribute_aliases", {})
    correct = invalid = missing = 0
    for case in annotation["cases"]:
        prediction = predictions.get(case["case_id"])
        if prediction is None or "natural_state" not in prediction:
            missing += 1
            continue
        try:
            if not isinstance(prediction["natural_state"], list):
                raise ValueError("state must be a list")
            actual = Counter(state_key(row, aliases) for row in prediction["natural_state"])
        except (TypeError, ValueError):
            invalid += 1
            continue
        expected = Counter(state_key(row, aliases) for row in case["expected_state"])
        correct += actual == expected
    total = len(annotation["cases"])
    return {"protocol": VERSION, "annotated_cases": total, "correct": correct,
            "accuracy": correct / total, "invalid_predictions": invalid,
            "missing_predictions": missing, "history_evaluated": False}
