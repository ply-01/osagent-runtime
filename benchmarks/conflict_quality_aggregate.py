"""Unmodified scorers with aggregate-only output for isolated quality checks."""
import argparse
import json
import platform
from pathlib import Path

import kylin_memory_eval as evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, files = evaluation.verify_dataset(args.dataset_root)
    pref, pref_time = evaluation.predict_preferences(files["preference_inputs"])
    conflicts, conflict_time = evaluation.predict_conflicts(files["conflict_inputs"])
    pref_score, _ = evaluation.score_preferences(pref, files["preference_labels"], files["preference_inputs"])
    conflict_score, _ = evaluation.score_conflicts(conflicts, files["conflict_labels"], files["conflict_inputs"])
    result = {
        "platform": platform.platform(), "role": "frozen reuse aggregate regression",
        "manifest_sha256": evaluation.sha256(args.dataset_root / "manifest.json"),
        "labels_loaded_after_prediction": True,
        "preference_extraction": pref_score, "conflict_handling": conflict_score,
        "elapsed_seconds": pref_time + conflict_time,
        "source_sha256": {str(path.relative_to(evaluation.ROOT)): evaluation.sha256(path)
                           for path in sorted((evaluation.ROOT / "src/os_agent/memory").glob("*.py"))},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
