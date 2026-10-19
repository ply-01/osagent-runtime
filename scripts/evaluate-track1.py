"""Archive the portable competition scenario for the optional evaluation pipeline."""
import argparse
import json
from pathlib import Path
from os_agent.track1 import run_scenario

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_scenario(Path(__file__).resolve().parents[1] / "examples/track1/maintenance-handover.json", args.output.parent / "runs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if report["passed"] else 1)
