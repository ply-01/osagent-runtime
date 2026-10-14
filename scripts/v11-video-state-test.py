"""Visible, repeatable real evaluation for recording; no fabricated progress."""
import json
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path


def main():
    root = Path.home() / ".cache/osagent-conflict-contract-v2-20260911"
    python = Path.home() / ".local/share/osagent/venv/bin/python"
    while True:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        output = root / "results" / ("video-" + run_id)
        output.mkdir(parents=True)
        result = output / "result.json"
        config = {
            "schema_version": 1, "sources": [],
            "jobs": [{"id": "video", "system": "OSagent Kylin V11 frozen rules",
                "dataset": "Unviewed reused 480", "dataset_revision": "v5",
                "evaluation_role": "v2 draft automated labels", "metric_scope": "natural state",
                "output": str(result), "command": [str(python), str(root / "benchmarks/evaluate_isolated_state_v2.py"),
                    "--dataset-root", str(Path.home() / ".cache/osagent-v11-eval-20260909/data/benchmarks/kylin-memory-eval-v5"),
                    "--annotations", str(root / "annotations/annotations.json"), "--output", str(result)],
                "env": {"OS_AGENT_MEMORY_EXTRACTOR": "rules", "OS_AGENT_EVAL_WORKERS": "1"},
                "metrics": [{"id": "state", "label": "State accuracy", "path": "full_split_accuracy", "unit": "percent"}]}],
            "forbidden_publish_paths": [".local-evaluation"], "forbidden_tracked_paths": ["data/benchmarks", ".local-evaluation"],
            "manifest_output": str(output / "pipeline.json"), "report_output": str(output / "report.md"),
        }
        config_path = output / "config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        print("\nOSagent | Kylin V11 | LIVE EVALUATION", flush=True)
        print(f"Run: {run_id}\n480 cases | rules | v2-draft1 | automated annotations, not human gold", flush=True)
        print("This is a frozen evaluation snapshot, not a live chat latency test.\n", flush=True)
        with (output / "console.log").open("w", encoding="utf-8") as log:
            # Reuse the unified job executor; this snapshot is not a Git checkout.
            # Repository publish checks remain in the Windows host workflow.
            execute = "import json,sys; from pathlib import Path; from memory_eval_pipeline import run_job; p=Path(sys.argv[1]); run_job(p.parents[2],json.loads(p.read_text())['jobs'][0],no_run=False)"
            process = subprocess.Popen([str(python), "-c", execute, str(config_path)], cwd=root / "benchmarks", stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        print(f"\nEvaluation job exit: {code} (repository publish check is separate)", flush=True)
        if code == 0 and result.exists():
            data = json.loads(result.read_text(encoding="utf-8"))
            state = data["state_v2_reviewed_subset"]
            print(f"\nRESULT: {state['correct']}/{data['total_executed']} correct", flush=True)
            if data["full_split_accuracy"] is not None:
                print(f"State accuracy: {100 * data['full_split_accuracy']:.2f}%", flush=True)
            print(f"Annotation coverage: {100 * data['review_coverage']:.2f}%", flush=True)
            print(f"Prediction batch time: {data['elapsed_seconds']:.3f} seconds", flush=True)
            print(f"Result: {result}", flush=True)
            print("Scoring revision, NOT evidence of algorithm improvement.", flush=True)
        try:
            choice = input("\nPress ENTER to run again for recording; type q then ENTER to close: ")
        except EOFError:
            break
        if choice.strip().lower() == "q":
            break


if __name__ == "__main__":
    main()
