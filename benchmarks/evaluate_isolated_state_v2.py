"""Execute all unviewed cases, score only independently reviewed annotations."""
import argparse
import json
import platform
import tempfile
from pathlib import Path

import kylin_memory_eval as evaluation
from conflict_state_contract import score_states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Result exists; refusing to overwrite")
    _, files = evaluation.verify_dataset(args.dataset_root)
    # Selection contains identifiers only and is separate from the answer file.
    selection = json.loads(args.annotations.with_name("selection.json").read_text(encoding="utf-8"))
    manifest = evaluation.sha256(args.dataset_root / "manifest.json")
    if selection["manifest_sha256"] != manifest:
        raise RuntimeError("Selection mismatch")
    annotation_hash = evaluation.sha256(args.annotations)
    rows = [row for row in evaluation.read_jsonl(files["conflict_inputs"]) if row["case_id"] not in selection["selected_ids"]]
    ids = {row["case_id"] for row in rows}
    with tempfile.TemporaryDirectory(prefix="isolated-conflict-v2-") as root:
        root = Path(root)
        inputs = root / "inputs.jsonl"
        inputs.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        print(f"state=prediction_started cases={len(rows)}", flush=True)
        predictions, elapsed = evaluation.predict_conflicts(inputs, capture_natural_state=True)
        print("state=prediction_finished loading_labels=true", flush=True)
        annotation = json.loads(args.annotations.read_text(encoding="utf-8"))
        if annotation_hash != evaluation.sha256(args.annotations) or annotation["manifest_sha256"] != manifest:
            raise RuntimeError("Annotations changed")
        if not {case["case_id"] for case in annotation["cases"]}.issubset(ids):
            raise RuntimeError("Annotation outside isolated split")
        labels = root / "labels.jsonl"
        labels.write_text("\n".join(json.dumps(row) for row in evaluation.read_jsonl(files["conflict_labels"]) if row["case_id"] in ids), encoding="utf-8")
        legacy, _ = evaluation.score_conflicts(predictions, labels, inputs)
        reviewed = score_states(predictions, annotation) if annotation["cases"] else {"annotated_cases": 0, "correct": 0, "accuracy": None}
    accepted = reviewed["annotated_cases"]
    pending = len(rows) - accepted
    result = {"platform": platform.platform(), "role": "V11 frozen snapshot, unviewed reused split, automated annotations",
        "manifest_sha256": manifest, "annotation_sha256": annotation_hash, "total_executed": len(rows),
        "labels_loaded_after_prediction": True, "legacy": legacy, "state_v2_reviewed_subset": reviewed,
        "review_coverage": accepted / len(rows), "pending_review": pending,
        "full_split_accuracy": reviewed["accuracy"] if pending == 0 else None,
        "full_split_possible_range": [reviewed["correct"] / len(rows), (reviewed["correct"] + pending) / len(rows)],
        "annotation_model": annotation["model"], "review_counts": annotation["review_counts"],
        "elapsed_seconds": elapsed, "source_sha256": {str(p.relative_to(evaluation.ROOT)): evaluation.sha256(p)
          for p in sorted((evaluation.ROOT / "src/os_agent/memory").glob("*.py"))}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"state": "finished", "executed": len(rows), "reviewed": accepted, "correct": reviewed["correct"], "pending": pending}))


if __name__ == "__main__":
    main()
