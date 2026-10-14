"""Plot V11 aggregate state evaluation with annotation coverage attached."""
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.result.read_bytes()
    data = json.loads(raw)
    subset = data["state_v2_reviewed_subset"]
    total = data["total_executed"]
    accepted = subset["annotated_cases"]
    correct = subset["correct"]
    pending = data["pending_review"]
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"], "font.size": 17})
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    fig.subplots_adjust(left=.22, right=.92, top=.68, bottom=.28)
    counts = [correct, accepted - correct, pending]
    ax.barh([0, 1, 2], counts, color=["#16826b", "#b85160", "#818892"], height=.5)
    ax.set_yticks([0, 1, 2], ["已复核 · 状态正确", "已复核 · 状态不符", "标注待复核"])
    ax.invert_yaxis()
    ax.set_xlim(0, total * 1.15)
    for index, count in enumerate(counts):
        ax.text(count + total * .02, index, str(count), va="center", fontsize=24)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_xlabel("样例数")
    fig.text(.08, .91, "银河麒麟 V11 · 自然语言状态评测", fontsize=29, weight="bold")
    score = f"{100 * correct / accepted:.2f}%" if accepted else "无可评分标注"
    qualifier = "全 480 条" if not pending else "已复核子集"
    fig.text(.08, .82, f"{qualifier}准确率 {score}   |   标注覆盖 {accepted}/{total}", fontsize=24)
    fig.text(.08, .75, "2026-09-11 · v2-draft1 · 冻结快照实测 · 非算法提升对比", color="#555b64")
    fig.text(.08, .17, f"实际执行 {total} 条；旧协议通过率 {100 * data['legacy']['natural_ingress_accuracy']:.2f}%（不可跨口径比较）", fontsize=15)
    fig.text(.08, .115, "自动复核：同一模型两次独立标注 + 原标签一致性核对；非人工金标准", fontsize=14)
    fig.text(.08, .065, "原始结果 SHA-256：" + hashlib.sha256(raw).hexdigest()[:24], fontsize=13, color="#555b64")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
