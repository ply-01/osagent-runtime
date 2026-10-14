"""Create a reviewable annotation draft from ONLY the authorized local sample."""
import json
from pathlib import Path

from conflict_state_contract import VERSION, validate_annotation

ROOT = Path(__file__).resolve().parent.parent
PRIVATE = ROOT / ".local-evaluation/authorized-conflict-20"

# Evaluation-only ontology, manually reviewed against the authorized sample.
# Values and case identities never form part of property matching.
PROPERTIES = {
    "backup_day": "automation.backup_schedule",
    "theme": "ui.theme",
    "office_app": "application.preferred_tool",
    "notification": "notification.delivery_mode",
    "output_format": "document.output_format",
    "reply_style": "response.style",
    "input_method": "input.preferred_method",
    "page_size": "document.page_size",
}


def main():
    target = PRIVATE / "state-v2-annotations.json"
    if target.exists():
        raise RuntimeError("Annotations already exist; do not overwrite a frozen contract")
    selection = json.loads((PRIVATE / "selection.json").read_text(encoding="utf-8"))
    rows = json.loads((PRIVATE / "development.json").read_text(encoding="utf-8"))
    if {r["input"]["case_id"] for r in rows} != set(selection["selected_ids"]):
        raise ValueError("development selection mismatch")
    cases = []
    for row in rows:
        expected = []
        for fact in row["label"]["expected_state"]:
            expected.append({"entity": "用户", "scope": fact["entity"] if row["input"]["category"] == "scoped" else "",
                             "attribute": PROPERTIES[fact["attribute"]], "value": fact["value"], "status": "active"})
        cases.append({"case_id": row["input"]["case_id"], "expected_state": expected})
    annotation = {"schema_version": VERSION, "role": "authorized development annotation draft; not an independent benchmark",
        "manifest_sha256": selection["manifest_sha256"],
        "annotation_basis": "Authorized natural-language samples reviewed by coding agent; structured expected values reused only after scope/owner interpretation. External human review pending.",
        "attribute_aliases": PROPERTIES, "cases": cases}
    validate_annotation(annotation)
    target.write_text(json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Created authorized development annotation draft: 20 cases")


if __name__ == "__main__":
    main()
