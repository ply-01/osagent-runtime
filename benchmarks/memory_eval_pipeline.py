"""Run revision-pinned memory evaluations and generate a normalized report."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import venv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Metric:
    metric_id: str
    label: str
    path: str
    unit: str
    threshold: dict[str, float] | None = None
    competition_metric: bool = False


def nested_value(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"result does not contain metric path: {path}")
        value = value[part]
    return value


def threshold_passed(value: Any, threshold: dict[str, float] | None) -> bool | None:
    if threshold is None:
        return None
    if not isinstance(value, (int, float)):
        return False
    if "min" in threshold:
        return float(value) >= float(threshold["min"])
    if "max" in threshold:
        return float(value) <= float(threshold["max"])
    raise ValueError(f"unsupported threshold: {threshold}")


def ensure_git_source(root: Path, source: dict[str, Any], *, offline: bool) -> None:
    target = root / source["path"]
    revision = source["revision"]
    if not target.exists():
        if offline:
            raise FileNotFoundError(f"offline source is missing: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", source["url"], str(target)], check=True, cwd=root
        )
    current = subprocess.run(
        ["git", "-C", str(target), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    if current == revision:
        return
    if offline:
        raise RuntimeError(
            f"source {source['id']} is at {current}, expected {revision}"
        )
    subprocess.run(
        ["git", "-C", str(target), "fetch", "--depth", "1", "origin", revision],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(target), "checkout", "--detach", revision], check=True
    )


def ensure_python_runtime(root: Path, source: dict[str, Any], *, offline: bool) -> None:
    target = root / source["path"]
    python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    marker = target / ".osagent-runtime.json"
    expected = {"requirements": source["requirements"]}
    if (
        python.is_file()
        and marker.is_file()
        and json.loads(marker.read_text(encoding="utf-8")) == expected
    ):
        return
    if offline:
        raise RuntimeError(f"Python runtime is missing or stale: {target}")
    if not python.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(target)
    subprocess.run(
        [str(python), "-m", "pip", "install", *source["requirements"]],
        check=True,
        cwd=root,
    )
    marker.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")


def ensure_npm_runtime(root: Path, source: dict[str, Any], *, offline: bool) -> None:
    target = root / source["path"]
    marker = target / ".osagent-runtime.json"
    expected = {
        "packages": source["packages"],
        "omit_optional": bool(source.get("omit_optional", False)),
    }
    if marker.is_file() and json.loads(marker.read_text(encoding="utf-8")) == expected:
        return
    if offline:
        raise RuntimeError(f"npm runtime is missing or stale: {target}")
    target.mkdir(parents=True, exist_ok=True)
    npm = "npm.cmd" if os.name == "nt" else "npm"
    command = [npm, "install", "--prefix", str(target), *source["packages"]]
    if expected["omit_optional"]:
        command.append("--omit=optional")
    command.extend(["--no-audit", "--no-fund"])
    subprocess.run(command, check=True, cwd=root)
    marker.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")


def ensure_source(root: Path, source: dict[str, Any], *, offline: bool) -> None:
    source_type = source.get("type")
    if source_type == "git":
        ensure_git_source(root, source, offline=offline)
        return
    if source_type == "python_venv":
        ensure_python_runtime(root, source, offline=offline)
        return
    if source_type == "npm_runtime":
        ensure_npm_runtime(root, source, offline=offline)
        return
    raise ValueError(f"unsupported source type for {source['id']}: {source_type}")


def render_command(parts: list[str], *, root: Path, output: Path) -> list[str]:
    mem0_python = root / "data" / "comparison-runtimes" / "mem0-2.0.14"
    mem0_python /= "Scripts/python.exe" if os.name == "nt" else "bin/python"
    bun = root / "data" / "comparison-runtimes" / "oh-my-pi"
    bun /= "node_modules/.bin/bun.cmd" if os.name == "nt" else "node_modules/.bin/bun"
    oh_my_pi_runtime = (
        root
        / "data"
        / "comparison-runtimes"
        / "oh-my-pi-mnemopi-17.3.4-fts"
    )
    values = {
        "python": sys.executable,
        "root": str(root),
        "output": str(output),
        "mem0_python": str(mem0_python),
        "bun": str(bun),
        "oh_my_pi_runtime": str(oh_my_pi_runtime),
    }
    return [part.format(**values) for part in parts]


def run_job(root: Path, job: dict[str, Any], *, no_run: bool) -> dict[str, Any]:
    output = root / job["output"]
    if not no_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        command = render_command(job["command"], root=root, output=output)
        env = os.environ.copy()
        source_path = str(root / "src")
        env["PYTHONPATH"] = source_path + os.pathsep + env.get("PYTHONPATH", "")
        for key, value in job.get("env", {}).items():
            env[key] = str(value).format(root=str(root))
        completed = subprocess.run(
            command,
            cwd=root,
            env=env,
            check=False,
            timeout=int(job.get("timeout_seconds", 1800)),
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"job {job['id']} failed with exit code {completed.returncode}"
            )
    if not output.is_file():
        raise FileNotFoundError(f"job {job['id']} did not produce {output}")
    payload = json.loads(output.read_text(encoding="utf-8"))
    metrics = []
    for raw in job["metrics"]:
        metric = Metric(
            metric_id=raw["id"],
            label=raw["label"],
            path=raw["path"],
            unit=raw.get("unit", "number"),
            threshold=raw.get("threshold"),
            competition_metric=bool(raw.get("competition_metric", False)),
        )
        value = nested_value(payload, metric.path)
        metrics.append(
            {
                "id": metric.metric_id,
                "label": metric.label,
                "value": value,
                "unit": metric.unit,
                "threshold": metric.threshold,
                "passed": threshold_passed(value, metric.threshold),
                "competition_metric": metric.competition_metric,
            }
        )
    return {
        "id": job["id"],
        "system": job["system"],
        "dataset": job["dataset"],
        "dataset_revision": job["dataset_revision"],
        "evaluation_role": job["evaluation_role"],
        "metric_scope": job["metric_scope"],
        "output": job["output"],
        "metrics": metrics,
    }


def _format_value(metric: dict[str, Any]) -> str:
    value = metric["value"]
    unit = metric["unit"]
    if unit == "percent":
        return f"{float(value) * 100:.2f}%"
    if unit == "ms":
        return f"{float(value):.3f}ms"
    return str(value)


def build_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# 显式记忆系统自动评测报告",
        "",
        f"运行时间：{manifest['evaluated_at']}",
        "",
        "## 数据集与范围",
        "",
        "| 系统 | 数据集 | 固定版本 | 用途 | 指标范围 |",
        "|---|---|---|---|---|",
    ]
    for job in manifest["jobs"]:
        lines.append(
            f"| {job['system']} | {job['dataset']} | `{job['dataset_revision']}` "
            f"| {job['evaluation_role']} | {job['metric_scope']} |"
        )
    lines.extend(
        [
            "",
            "## 比赛四项硬指标",
            "",
            "只有标记为比赛指标的同协议结果进入本表。外部数据集的代理指标不会填入空位。",
            "",
            "| 系统 | 数据集 | 指标 | 结果 | 是否达标 |",
            "|---|---|---|---:|---|",
        ]
    )
    hard_rows = 0
    for job in manifest["jobs"]:
        for metric in job["metrics"]:
            if not metric["competition_metric"]:
                continue
            hard_rows += 1
            passed = "是" if metric["passed"] else "否"
            lines.append(
                f"| {job['system']} | {job['dataset']} | {metric['label']} "
                f"| {_format_value(metric)} | {passed} |"
            )
    if not hard_rows:
        lines.append("| - | - | - | - | 无同协议结果 |")
    lines.extend(
        [
            "",
            "## 外部数据集诊断",
            "",
            "这些结果用于跨系统横向比较或定位泛化问题，不等同于比赛四项分数。",
            "",
            "| 系统 | 数据集 | 指标 | 结果 |",
            "|---|---|---|---:|",
        ]
    )
    diagnostic_rows = 0
    for job in manifest["jobs"]:
        for metric in job["metrics"]:
            if metric["competition_metric"]:
                continue
            diagnostic_rows += 1
            lines.append(
                f"| {job['system']} | {job['dataset']} | {metric['label']} "
                f"| {_format_value(metric)} |"
            )
    if not diagnostic_rows:
        lines.append("| - | - | - | - |")
    lines.extend(
        [
            "",
            "## Git 发布边界",
            "",
            f"第三方源码/本地运行时发布检查：{'通过' if manifest['publish_guard']['passed'] else '失败'}。",
            "",
        ]
    )
    if manifest["publish_guard"]["violations"]:
        lines.extend(
            f"- `{path}`" for path in manifest["publish_guard"]["violations"]
        )
        lines.append("")
    lines.append(
        "报告由 `benchmarks/memory_eval_pipeline.py` 从各任务原始 JSON 自动生成。"
    )
    return "\n".join(lines) + "\n"


def publish_guard(
    root: Path,
    forbidden_paths: list[str],
    forbidden_tracked_paths: list[str] | None = None,
) -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    changed = [line[3:].replace("\\", "/") for line in status.stdout.splitlines()]
    tracked_paths = [line.replace("\\", "/") for line in tracked.stdout.splitlines()]
    changed_violations = {
        path
        for path in changed
        if any(path == prefix or path.startswith(prefix.rstrip("/") + "/") for prefix in forbidden_paths)
    }
    tracked_violations = {
        path
        for path in tracked_paths
        if any(
            path == prefix or path.startswith(prefix.rstrip("/") + "/")
            for prefix in forbidden_tracked_paths or []
        )
    }
    violations = sorted(changed_violations | tracked_violations)
    return {"passed": not violations, "violations": violations}


def select_jobs(
    config: dict[str, Any],
    selected: set[str],
) -> list[dict[str, Any]]:
    """Select defaults, or any explicitly named optional jobs."""
    return [
        job
        for job in config["jobs"]
        if (job["id"] in selected if selected else job.get("enabled", True))
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("config/memory_eval_pipeline.json")
    )
    parser.add_argument("--job", action="append", default=[])
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--no-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    selected = set(args.job)
    jobs = select_jobs(config, selected)
    unknown = selected - {job["id"] for job in config["jobs"]}
    if unknown:
        raise ValueError(f"unknown jobs: {', '.join(sorted(unknown))}")

    started = time.perf_counter()
    for source in config.get("sources", []):
        if any(source["id"] in job.get("sources", []) for job in jobs):
            ensure_source(root, source, offline=args.offline)
    job_results = [run_job(root, job, no_run=args.no_run) for job in jobs]
    guard = publish_guard(
        root,
        config.get("forbidden_publish_paths", []),
        config.get("forbidden_tracked_paths", []),
    )
    manifest = {
        "schema_version": 1,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "config": str(config_path.relative_to(root)).replace("\\", "/"),
        "jobs": job_results,
        "publish_guard": guard,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    manifest_path = root / config["manifest_output"]
    report_path = root / config["report_output"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path.write_text(build_report(manifest), encoding="utf-8")
    print(f"manifest: {manifest_path}")
    print(f"report: {report_path}")
    if not guard["passed"]:
        print("publish guard rejected forbidden local changes", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
