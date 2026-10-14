"""Conflict-only regression using the unchanged competition scorer."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import os
from pathlib import Path

import kylin_memory_eval as evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, files = evaluation.verify_dataset(args.dataset_root)
    predictions, elapsed = evaluation.predict_conflicts(files["conflict_inputs"])
    aggregate, _ = evaluation.score_conflicts(
        predictions, files["conflict_labels"], files["conflict_inputs"]
    )
    result = {
        "manifest_sha256": evaluation.sha256(args.dataset_root / "manifest.json"),
        "platform": platform.platform(),
        "evaluation_role": "frozen reuse conflict-only regression",
        "labels_loaded_after_prediction": True,
        "memory_extractor": os.getenv("OS_AGENT_MEMORY_EXTRACTOR", "rules"),
        "extraction_runtime": evaluation.EXTRACTION_DIAGNOSTICS,
        "conflict_handling": aggregate,
        "elapsed_seconds": elapsed,
        "source_sha256": {
            name: hashlib.sha256((evaluation.ROOT / "src/os_agent/memory" / name).read_bytes()).hexdigest()
            for name in ("mem0_core.py", "mem0_pipeline.py")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
