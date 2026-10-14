"""Export only an authorized failure sample; score the remainder in isolation."""
import argparse
import json
import platform
import random
import tempfile
from pathlib import Path

import kylin_memory_eval as evaluation


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def subset_score(predictions, inputs, labels, ids):
    with tempfile.TemporaryDirectory(prefix="osagent-split-score-") as root:
        root = Path(root)
        for name, rows in (("inputs", inputs), ("labels", labels)):
            (root / name).write_text("\n".join(json.dumps(row) for row in rows if row["case_id"] in ids), encoding="utf-8")
        score, _ = evaluation.score_conflicts({key: value for key, value in predictions.items() if key in ids}, root / "labels", root / "inputs")
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--state-annotations", type=Path)
    args = parser.parse_args()
    private = evaluation.ROOT / ".local-evaluation/authorized-conflict-20"
    selection_path = private / "selection.json"
    if args.select and selection_path.exists():
        raise RuntimeError("Selection already frozen; do not resample")
    _, files = evaluation.verify_dataset(args.dataset_root)
    manifest = evaluation.sha256(args.dataset_root / "manifest.json")
    source_hashes = {str(p.relative_to(evaluation.ROOT)): evaluation.sha256(p)
                     for p in sorted((evaluation.ROOT / "src/os_agent/memory").glob("*.py"))}
    annotation_hash = evaluation.sha256(args.state_annotations) if args.state_annotations else None
    predictions, elapsed = evaluation.predict_conflicts(files["conflict_inputs"], capture_natural_state=bool(args.state_annotations))
    _, failures = evaluation.score_conflicts(predictions, files["conflict_labels"], files["conflict_inputs"])
    inputs = evaluation.read_jsonl(files["conflict_inputs"])
    labels = evaluation.read_jsonl(files["conflict_labels"])
    if args.select:
        failed_ids = sorted(row["case_id"] for row in failures if not row["natural_correct"])
        ids = sorted(random.Random(20260910).sample(failed_ids, 20))
        selection = {"seed": 20260910, "manifest_sha256": manifest, "selected_ids": ids,
                     "baseline_failure_count": len(failed_ids), "baseline_sources": source_hashes}
        write_json(selection_path, selection)
        input_by_id = {row["case_id"]: row for row in inputs}
        labels_by_id = {row["case_id"]: row for row in labels}
        write_json(private / "development.json", [
            {"input": input_by_id[key], "label": labels_by_id[key], "prediction": predictions[key]} for key in ids
        ])
    else:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection["manifest_sha256"] != manifest:
            raise RuntimeError("Dataset changed")
    selected = set(selection["selected_ids"])
    remaining = {row["case_id"] for row in inputs} - selected
    development = subset_score(predictions, inputs, labels, selected)
    isolated = subset_score(predictions, inputs, labels, remaining)
    state_result = {}
    if args.state_annotations:
        from conflict_state_contract import score_states
        if annotation_hash != evaluation.sha256(args.state_annotations):
            raise RuntimeError("Annotations changed during prediction")
        annotation = json.loads(args.state_annotations.read_text(encoding="utf-8"))
        if annotation["manifest_sha256"] != manifest or {row["case_id"] for row in annotation["cases"]} != selected:
            raise RuntimeError("Annotations must match only the authorized development split")
        state_result = {"state_v2_development": score_states(predictions, annotation),
                        "state_v2_unscored_cases": len(remaining),
                        "annotation_sha256": annotation_hash,
                        "scorer_sha256": evaluation.sha256(Path(__file__).with_name("conflict_state_contract.py"))}
    if not args.select:
        write_json(private / ("state-v2-development-predictions.json" if args.state_annotations else "candidate-development-predictions.json"), {
            key: predictions[key] for key in sorted(selected)
        })
    write_json(args.output, {"platform": platform.platform(), "manifest_sha256": manifest,
        "seed": selection["seed"], "selected_ids": selection["selected_ids"],
        "role": "authorized development / previously evaluated unviewed remainder",
        "labels_loaded_after_prediction": True, "development": development,
        "isolated": isolated, "elapsed_seconds": elapsed, "source_sha256": source_hashes, **state_result})


if __name__ == "__main__":
    main()
