"""Render aggregate evaluation metrics without exposing benchmark examples."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    args = parser.parse_args()
    raw = args.result.read_bytes()
    data = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"], "font.size": 16})
    thresholds = data["competition_thresholds"]
    panels = [
        ("quality", "记忆质量指标", [
            ("偏好提取准确率", "preference_accuracy"),
            ("知识 Recall@5", "knowledge_recall_at_5"),
            ("自然语言冲突准确率", "natural_conflict_accuracy"),
        ], 100, "%"),
        ("latency", "完整在线检索延迟 · p95", [
            ("完整 Recall 请求", "online_recall_p95_ms"),
        ], 1, "ms"),
    ]
    for name, title, fields, scale, unit in panels:
        fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
        fig.subplots_adjust(left=.25, right=.9, top=.74, bottom=.25)
        values = [thresholds[key]["value"] * scale for _, key in fields]
        targets = [thresholds[key]["target"] * scale for _, key in fields]
        bars = ax.barh(range(len(fields)), values, height=.38, color=["#16826b" if thresholds[key]["passed"] else "#b85160" for _, key in fields])
        ax.set_yticks(range(len(fields)), [label for label, _ in fields])
        ax.invert_yaxis()
        ax.set_xlim(0, 110 if scale == 100 else max(values + targets) * 1.28)
        ax.set_xlabel(unit)
        for index, (bar, value, target) in enumerate(zip(bars, values, targets)):
            ax.text(value + (1 if scale == 100 else 15), index, f"{value:.1f}%" if scale == 100 else f"{value:.3f} ms", va="center", fontsize=21)
            ax.vlines(target, index - .29, index + .29, color="#30343b", linestyle="--", linewidth=2)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
        fig.text(.08, .91, title, fontsize=30, weight="bold")
        fig.text(.08, .85, args.run_label, fontsize=17, color="#555b64")
        target_text = " / ".join(f"{label} {'≥' if scale == 100 else '≤'} {target:g}{unit}" for (label, _), target in zip(fields, targets))
        fig.text(.08, .16, "虚线为内部验收阈值：" + target_text, fontsize=13)
        retrieval = data["knowledge_retrieval"]
        fig.text(.08, .115, f"冻结复用评测 | 偏好 {data['preference_extraction']['cases']} 条 | 冲突 {data['conflict_handling']['cases']} 条 | {retrieval['corpus_facts']:,} 事实 / {retrieval['queries']} 查询", fontsize=14)
        fig.text(.08, .065, "结果 SHA-256：" + digest[:24] + " | 原始 JSON 生成，不代表本轮修复后成绩", fontsize=12, color="#555b64")
        fig.savefig(args.output_dir / f"{name}.png", facecolor="white")
        plt.close(fig)
    (args.output_dir / "provenance.json").write_text(json.dumps({
        "source": str(args.result), "sha256": digest, "run_label": args.run_label,
        "metrics": thresholds,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
