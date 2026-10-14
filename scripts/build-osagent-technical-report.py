"""Build the OSagent technical solution and user manual as a styled DOCX.

The source of truth is the repository Markdown report. Charts are generated
from aggregate, already-published measurements only; no benchmark examples or
labels are opened by this script.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "doc/submission/2026-osagent-implementation-technical-solution.md"
ASSET_DIR = ROOT / "doc/submission/assets/technical-report"
OUTPUT = ROOT / "doc/submission/OSagent-技术方案与用户手册.docx"
TEMP_DIR = ROOT / "tmp/documents/osagent-technical-report"

NAVY = "16324F"
TEAL = "147D92"
GOLD = "D8A13A"
RED = "C94747"
INK = "1D242B"
MUTED = "66717B"
LIGHT = "EEF3F5"
PALE = "F6F8F9"
WHITE = "FFFFFF"
BLACK = "000000"


def configure_chinese_font() -> None:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/Deng.ttf"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            font_manager.fontManager.addfont(str(candidate))
            family = font_manager.FontProperties(fname=str(candidate)).get_name()
            plt.rcParams["font.family"] = family
            plt.rcParams["axes.unicode_minus"] = False
            return


def rounded_box(ax, x, y, w, h, text, *, color=NAVY, face=WHITE, size=10, weight="normal"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.2,
        edgecolor=f"#{color}",
        facecolor=f"#{face}",
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            color=f"#{INK}", weight=weight, linespacing=1.25)
    return box


def arrow(ax, start, end, *, color=TEAL):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                linewidth=1.2, color=f"#{color}"))


def save_figure(fig, name: str) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSET_DIR / name, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def architecture_chart() -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.965, "OSagent 六层架构", fontsize=16, weight="bold", color=f"#{NAVY}")
    rows = [
        ("交互层", "Open WebUI · MCP Host · 手机命令客户端", NAVY),
        ("接入层", "同源网关 · Bearer 鉴权 · 记忆 Filter", TEAL),
        ("运行层", "AgentLoop · 双 Planner · PlanValidator · Hook", "386A8C"),
        ("能力层", "文件/受控 Shell · 记忆 CRUD · 插件生命周期", GOLD),
        ("记忆层", "STM · Session/全局摘要 · Core · Archival · Recall", "4D8061"),
        ("平台层", "银河麒麟 Embedding SDK · C++ vector bridge · 本地数据", RED),
    ]
    y = 0.84
    for index, (label, detail, color) in enumerate(rows):
        rounded_box(ax, 0.03, y, 0.17, 0.095, label, color=color, face=WHITE, size=11, weight="bold")
        rounded_box(ax, 0.23, y, 0.74, 0.095, detail, color=color, face=PALE, size=9.2)
        if index < len(rows) - 1:
            arrow(ax, (0.50, y - 0.008), (0.50, y - 0.043), color=MUTED)
        y -= 0.135
    ax.text(0.03, 0.018, "自主研发边界：运行时、记忆、MCP、网关、部署与评测；Open WebUI 为开源前端框架。",
            fontsize=8.3, color=f"#{MUTED}")
    save_figure(fig, "architecture-layers.png")


def memory_lifecycle_chart() -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.93, "记忆生命周期：证据可追溯，索引可重建", fontsize=15, weight="bold", color=f"#{NAVY}")
    labels = ["事件", "Session", "结构化帧", "FactCard", "语义索引", "Recall"]
    colors = [NAVY, "386A8C", TEAL, "4D8061", GOLD, RED]
    xs = [0.02, 0.185, 0.35, 0.515, 0.68, 0.845]
    for i, (title, color, x) in enumerate(zip(labels, colors, xs)):
        rounded_box(ax, x, 0.54, 0.135, 0.18, title, color=color, face=WHITE, size=8.2,
                    weight="bold" if i in (0, 5) else "normal")
        if i < len(labels) - 1:
            arrow(ax, (x + 0.137, 0.63), (xs[i + 1] - 0.008, 0.63), color=MUTED)
    rounded_box(ax, 0.18, 0.18, 0.27, 0.18, "Core\n当前有效事实", color=TEAL, face=PALE, size=10, weight="bold")
    rounded_box(ax, 0.55, 0.18, 0.27, 0.18, "Archival + 冷事件\n被替代事实与完整证据", color=GOLD, face=PALE, size=10, weight="bold")
    arrow(ax, (0.58, 0.53), (0.34, 0.37), color=TEAL)
    arrow(ax, (0.61, 0.53), (0.67, 0.37), color=GOLD)
    ax.text(0.5, 0.06, "Prompt 压缩不改原始证据；存储压缩通过热窗口、gzip 冷档和跨 Session 全局摘要实现。",
            ha="center", fontsize=8.2, color=f"#{MUTED}")
    save_figure(fig, "memory-lifecycle.png")


def recall_chart() -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.94, "五来源召回：结构过滤在前，语义检索在后", fontsize=15, weight="bold", color=f"#{NAVY}")
    rounded_box(ax, 0.02, 0.70, 0.19, 0.15, "Query Plan\n原始查询 / 有界变体", color=NAVY, face=PALE, size=8.0, weight="bold")
    rounded_box(ax, 0.27, 0.70, 0.22, 0.15, "结构过滤\nentity / scope / time", color=TEAL, face=PALE, size=8.0, weight="bold")
    arrow(ax, (0.215, 0.775), (0.265, 0.775))
    sources = [
        ("Core lexical", NAVY), ("Core semantic", TEAL), ("Session semantic", "386A8C"),
        ("Archival", GOLD), ("Raw events", RED),
    ]
    for i, (name, color) in enumerate(sources):
        y = 0.68 - i * 0.125
        rounded_box(ax, 0.57, y, 0.19, 0.085, name, color=color, face=WHITE, size=8.7)
        arrow(ax, (0.495, 0.775), (0.56, y + 0.042), color=color)
        arrow(ax, (0.765, y + 0.042), (0.815, 0.50), color=color)
    rounded_box(ax, 0.82, 0.42, 0.15, 0.16, "RRF 融合\n证据配额\n结果蒸馏", color=RED, face=PALE, size=9.2, weight="bold")
    ax.text(0.02, 0.47, "执行语义", fontsize=10, weight="bold", color=f"#{INK}")
    ax.text(0.02, 0.39, "• Core 词法与语义先融合", fontsize=8.6, color=f"#{MUTED}")
    ax.text(0.02, 0.32, "• Core 不足再级联 Archival", fontsize=8.6, color=f"#{MUTED}")
    ax.text(0.02, 0.25, "• 深度召回时加入 Session/Raw", fontsize=8.6, color=f"#{MUTED}")
    ax.text(0.02, 0.18, "• 多条 Query 路由共享批量 Embedding", fontsize=8.6, color=f"#{MUTED}")
    ax.text(0.02, 0.08, "“五路”表示五种候选来源；不是五个线程无条件同时运行。", fontsize=8.2, color=f"#{RED}")
    save_figure(fig, "five-source-recall.png")


def bridge_chart() -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.93, "C++ vector bridge", fontsize=14.2, weight="bold", color=f"#{NAVY}")
    ax.text(0.02, 0.865, "把不稳定 C++ ABI 收敛为稳定 C ABI v2", fontsize=9.3, color=f"#{MUTED}")
    blocks = [
        (0.02, "Python", NAVY),
        (0.22, "ctypes", "386A8C"),
        (0.42, "C ABI v2", TEAL),
        (0.62, "C++ SDK", GOLD),
        (0.82, "Runtime", RED),
    ]
    for i, (x, label, color) in enumerate(blocks):
        rounded_box(ax, x, 0.57, 0.16, 0.16, label, color=color, face=PALE, size=8.8,
                    weight="bold" if i in (0, 2, 4) else "normal")
        if i < len(blocks) - 1:
            arrow(ax, (x + 0.162, 0.65), (blocks[i + 1][0] - 0.008, 0.65), color=MUTED)
    notes = [
        "FNV-1a 数值主键 + 原始 VARCHAR key",
        "FLAT / COSINE，支持 allowed key 过滤",
        "C++ malloc 的结果由 bridge free 释放",
        "Core JSON 是权威事实源；系统向量只负责加速",
    ]
    for i, note in enumerate(notes):
        x = 0.03 + (i % 2) * 0.49
        y = 0.34 - (i // 2) * 0.13
        ax.text(x, y, f"• {note}", fontsize=8.7, color=f"#{INK}")
    ax.text(0.03, 0.07, "实机状态：768 维 write/search/filter/update/delete 真实往返通过，已接入目标机 runtime。",
            fontsize=8.5, color=f"#{TEAL}", weight="bold")
    save_figure(fig, "kylin-vector-bridge.png")


def deployment_chart() -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.65))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.93, "银河麒麟 V11 部署拓扑", fontsize=15, weight="bold", color=f"#{NAVY}")
    rounded_box(ax, 0.03, 0.61, 0.18, 0.19, "Windows 浏览器\n127.0.0.1:33001", color=NAVY, face=PALE, size=8, weight="bold")
    rounded_box(ax, 0.29, 0.61, 0.20, 0.19, "VirtualBox NAT\n目标端口 10200", color="386A8C", face=PALE, size=9)
    rounded_box(ax, 0.57, 0.61, 0.18, 0.19, "Open WebUI\n+ Gateway", color=TEAL, face=PALE, size=9.2, weight="bold")
    rounded_box(ax, 0.80, 0.61, 0.17, 0.19, "OSagent\n127.0.0.1:8765", color=GOLD, face=PALE, size=9.2, weight="bold")
    for a, b in [((0.215, 0.705), (0.285, 0.705)), ((0.495, 0.705), (0.565, 0.705)), ((0.755, 0.705), (0.795, 0.705))]:
        arrow(ax, a, b, color=MUTED)
    rounded_box(ax, 0.58, 0.25, 0.18, 0.17, "Docker 容器\n仅负责 UI", color=TEAL, face=WHITE, size=9)
    rounded_box(ax, 0.80, 0.25, 0.17, 0.17, "V11 用户态\n记忆与工具\n系统 SDK", color=RED, face=WHITE, size=8.5)
    arrow(ax, (0.66, 0.60), (0.66, 0.43), color=TEAL)
    arrow(ax, (0.88, 0.60), (0.88, 0.43), color=RED)
    ax.text(0.03, 0.30, "安全边界", fontsize=10, weight="bold", color=f"#{INK}")
    ax.text(0.03, 0.22, "• 默认单用户 UI 仅通过主机 Loopback 暴露", fontsize=8.5, color=f"#{MUTED}")
    ax.text(0.03, 0.15, "• Gateway 注入内部 Bearer，浏览器不持有服务密钥", fontsize=8.5, color=f"#{MUTED}")
    ax.text(0.03, 0.08, "• Kylin SDK 使用用户 runtime socket，因此 OSagent 不放入容器", fontsize=8.5, color=f"#{MUTED}")
    save_figure(fig, "v11-deployment-topology.png")


def competition_chart() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6.2, 5.1))
    fig.suptitle("当前保留结果与内部比赛阈值", fontsize=15, weight="bold", color=f"#{NAVY}", y=0.98)
    entries = [
        ("偏好准确率", 96.6, 85.0, 100.0, "%", True),
        ("知识 Recall@5", 70.0, 85.0, 100.0, "%", True),
        ("10k 在线 p95", 440.464, 500.0, 600.0, "ms", False),
        ("自然语言冲突", 19.8, 88.0, 100.0, "%", True),
    ]
    for ax, (title, actual, threshold, maximum, unit, higher) in zip(axes.flat, entries):
        passed = actual >= threshold if higher else actual <= threshold
        color = TEAL if passed else RED
        ax.barh([0], [actual], height=0.34, color=f"#{color}")
        ax.axvline(threshold, color=f"#{GOLD}", linewidth=2, linestyle="--")
        ax.set_xlim(0, maximum)
        ax.set_yticks([])
        ax.set_ylim(-0.65, 0.65)
        ax.set_title(title, fontsize=10.2, weight="bold", loc="left", pad=10)
        ax.text(actual, 0, f" {actual:g}{unit}", va="center", fontsize=9, color=f"#{INK}")
        ax.text(threshold, -0.27, f"阈值 {threshold:g}{unit}", ha="center", va="top", fontsize=7.6, color=f"#{MUTED}")
        ax.text(0.98, 0.88, "通过" if passed else "未通过", transform=ax.transAxes,
                ha="right", va="top", fontsize=8.5, color=f"#{color}", weight="bold")
        ax.grid(axis="x", color="#DCE3E7", linewidth=0.6)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(0.02, 0.015, "环境：Windows 11 / CPU / 固定 KylinMemoryEval-v5 复用回归。V11 完整结果单独报告。",
             fontsize=8, color=f"#{MUTED}")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.13, hspace=0.72, wspace=0.20)
    save_figure(fig, "competition-metrics.png")


def v11_competition_chart() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6.2, 5.1))
    fig.suptitle("银河麒麟 V11 目标机实测与内部阈值", fontsize=15, weight="bold", color=f"#{NAVY}", y=0.98)
    entries = [
        ("偏好准确率", 90.0, 85.0, 100.0, "%", True),
        ("知识 Recall@5", 75.6, 85.0, 100.0, "%", True),
        ("10k 在线 p95", 6174.272, 500.0, 7000.0, "ms", False),
        ("自然语言冲突", 20.0, 88.0, 100.0, "%", True),
    ]
    for ax, (title, actual, threshold, maximum, unit, higher) in zip(axes.flat, entries):
        passed = actual >= threshold if higher else actual <= threshold
        color = TEAL if passed else RED
        ax.barh([0], [actual], height=0.34, color=f"#{color}")
        ax.axvline(threshold, color=f"#{GOLD}", linewidth=2, linestyle="--")
        ax.set_xlim(0, maximum)
        ax.set_yticks([])
        ax.set_ylim(-0.65, 0.65)
        ax.set_title(title, fontsize=10.2, weight="bold", loc="left", pad=10)
        ax.text(actual, 0, f" {actual:g}{unit}", va="center", fontsize=9, color=f"#{INK}")
        ax.text(threshold, -0.27, f"阈值 {threshold:g}{unit}", ha="center", va="top", fontsize=7.6, color=f"#{MUTED}")
        ax.text(0.98, 0.88, "通过" if passed else "未通过", transform=ax.transAxes,
                ha="right", va="top", fontsize=8.5, color=f"#{color}", weight="bold")
        ax.grid(axis="x", color="#DCE3E7", linewidth=0.6)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(0.02, 0.015, "环境：银河麒麟 V11 虚拟机 / KylinSystemEmbeddingProvider / 单 worker / 低 CPU 与 I/O 优先级。",
             fontsize=8, color=f"#{MUTED}")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.13, hspace=0.72, wspace=0.20)
    save_figure(fig, "v11-competition-metrics.png")


def comparison_chart() -> None:
    systems = ["OSagent", "Mnemopi\nFTS fallback", "Mem0 OSS\nlimited extras"]
    recall = [45.0, 43.0, 22.0]
    p95 = [43.806, 7.109, 88.179]
    colors = [f"#{TEAL}", f"#{GOLD}", f"#{MUTED}"]
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.65))
    fig.suptitle("显式记忆后端：LoCoMo 固定子集", fontsize=15, weight="bold", color=f"#{NAVY}", y=0.98)
    axes[0].bar(systems, recall, color=colors, width=0.62)
    axes[0].set_title("Evidence R@5（越高越好）", fontsize=10, weight="bold")
    axes[0].set_ylim(0, 55)
    axes[0].set_ylabel("%")
    axes[1].bar(systems, p95, color=colors, width=0.62)
    axes[1].set_title("查询 p95（越低越好）", fontsize=10, weight="bold")
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("ms")
    for ax, values in zip(axes, (recall, p95)):
        for index, value in enumerate(values):
            ax.text(index, value + 1.5, f"{value:g}", ha="center", fontsize=8.5)
        ax.grid(axis="y", color="#DCE3E7", linewidth=0.6)
        ax.tick_params(axis="x", labelsize=7.3)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(0.02, 0.015, "仅为固定 100 条 evidence 检索对照；Provider、可选依赖和能力边界不完全等价。",
             fontsize=8, color=f"#{MUTED}")
    fig.tight_layout(rect=[0, 0.06, 1, 0.92], w_pad=2.2)
    save_figure(fig, "explicit-memory-comparison.png")


def generate_charts() -> None:
    configure_chinese_font()
    architecture_chart()
    memory_lifecycle_chart()
    recall_chart()
    bridge_chart()
    deployment_chart()
    competition_chart()
    v11_competition_chart()
    comparison_chart()


def set_east_asia(run, name: str) -> None:
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shading = tc_pr.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        tc_pr.append(shading)
    shading.set(qn("w:fill"), fill)


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_table_borders(table) -> None:
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = borders.find(qn(f"w:{edge}"))
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "4")
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), "C9D2D8")


def add_page_field(paragraph) -> None:
    paragraph.add_run("第 ")
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, separate, end])
    paragraph.add_run(" 页")


def style_reference(reference_path: Path) -> None:
    raw = subprocess.check_output(["pandoc", "--print-default-data-file", "reference.docx"])
    reference_path.write_bytes(raw)
    doc = Document(reference_path)
    for section in doc.sections:
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
        section.left_margin = Cm(3.0)
        section.right_margin = Cm(2.0)
        section.top_margin = Cm(3.0)
        section.bottom_margin = Cm(2.5)
        section.header_distance = Cm(1.4)
        section.footer_distance = Cm(1.4)
        section.different_first_page_header_footer = True

        header = section.header
        paragraph = header.paragraphs[0]
        paragraph.text = "OSagent  技术方案与用户手册"
        paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        for run in paragraph.runs:
            set_east_asia(run, "Microsoft YaHei")
            run.font.size = Pt(8.5)
            run.font.color.rgb = RGBColor.from_string(MUTED)

        footer = section.footer
        paragraph = footer.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.text = ""
        add_page_field(paragraph)
        for run in paragraph.runs:
            set_east_asia(run, "Microsoft YaHei")
            run.font.size = Pt(8.5)
            run.font.color.rgb = RGBColor.from_string(MUTED)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "SimSun"
    normal._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "SimSun")
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor.from_string(INK)
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    normal.paragraph_format.space_after = Pt(4)
    normal.paragraph_format.widow_control = True

    style_specs = {
        "Title": ("Microsoft YaHei", 29, True, BLACK, WD_ALIGN_PARAGRAPH.CENTER),
        "Subtitle": ("Microsoft YaHei", 15, False, BLACK, WD_ALIGN_PARAGRAPH.CENTER),
        "Author": ("Microsoft YaHei", 11, False, MUTED, WD_ALIGN_PARAGRAPH.CENTER),
        "Date": ("Microsoft YaHei", 10.5, False, MUTED, WD_ALIGN_PARAGRAPH.CENTER),
        "Heading 1": ("Microsoft YaHei", 16, True, BLACK, WD_ALIGN_PARAGRAPH.LEFT),
        "Heading 2": ("Microsoft YaHei", 13.5, True, BLACK, WD_ALIGN_PARAGRAPH.LEFT),
        "Heading 3": ("Microsoft YaHei", 11.5, True, BLACK, WD_ALIGN_PARAGRAPH.LEFT),
        "Caption": ("Microsoft YaHei", 9, False, MUTED, WD_ALIGN_PARAGRAPH.CENTER),
        "TOC Heading": ("Microsoft YaHei", 18, True, BLACK, WD_ALIGN_PARAGRAPH.LEFT),
    }
    for style_name, (font_name, size, bold, color, align) in style_specs.items():
        if style_name not in styles:
            continue
        style = styles[style_name]
        style.font.name = font_name
        style._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), font_name)
        style.font.size = Pt(size)
        style.font.bold = bold
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.alignment = align
        if style_name.startswith("Heading"):
            style.paragraph_format.keep_with_next = True
            style.paragraph_format.keep_together = True
            style.paragraph_format.space_before = Pt(13 if style_name == "Heading 1" else 9)
            style.paragraph_format.space_after = Pt(6)

    if "Source Code" in styles:
        code = styles["Source Code"]
        code.font.name = "Consolas"
        code.font.size = Pt(8)
        code.font.color.rgb = RGBColor.from_string(INK)
        code.paragraph_format.line_spacing = 1.05

    doc.save(reference_path)


def prepare_markdown(path: Path) -> None:
    text = SOURCE.read_text(encoding="utf-8")
    required_sections = (
        "# OSagent 技术方案与用户手册",
        "## 摘要",
        "## 11. 结论",
        "2026-09-10-kylin-v11-system-memory-evaluation.md",
    )
    missing = [section for section in required_sections if section not in text]
    if missing:
        raise RuntimeError(f"report source is missing required content: {missing}")
    body_at = text.find("## 摘要")
    body = text[body_at:]
    front_matter = """---
title: "OSagent 技术方案与用户手册"
subtitle: "面向银河麒麟桌面操作系统 V11 的可解释长期记忆智能体"
author: "OSagent 项目组"
date: "2026 年 9 月 10 日 · V2.3"
lang: zh-CN
---

"""
    path.write_text(front_matter + body, encoding="utf-8")


def paragraph_contains_page_break(paragraph) -> bool:
    return bool(paragraph._element.xpath(".//w:br[@w:type='page']"))


def postprocess_docx(path: Path) -> None:
    doc = Document(path)
    doc.core_properties.title = "OSagent 技术方案与用户手册"
    doc.core_properties.subject = "软件架构、算法原理、实现方案、部署指南、用户手册与效果验证报告"
    doc.core_properties.author = "OSagent 项目组"
    doc.core_properties.keywords = "OSagent, 银河麒麟V11, 长期记忆, C++ vector bridge, MCP"

    for section in doc.sections:
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
        section.left_margin = Cm(3.0)
        section.right_margin = Cm(2.0)
        section.top_margin = Cm(3.0)
        section.bottom_margin = Cm(2.5)

    first_heading = True
    paragraphs = list(doc.paragraphs)
    for index, paragraph in enumerate(paragraphs):
        name = paragraph.style.name if paragraph.style else ""
        if paragraph.text.strip() == "摘要":
            paragraph.paragraph_format.page_break_before = True
        if name == "TOC Heading":
            paragraph.paragraph_format.page_break_before = True
        if name == "Heading 1":
            paragraph.paragraph_format.keep_with_next = True
            paragraph.paragraph_format.widow_control = True
            if first_heading:
                paragraph.paragraph_format.page_break_before = True
                first_heading = False
            elif paragraph.text.startswith(("3.", "5.", "9.", "11.", "16.", "附录")):
                paragraph.paragraph_format.page_break_before = True
        if name == "Caption":
            paragraph.paragraph_format.keep_with_next = False
            paragraph.paragraph_format.keep_together = True
        if name == "Source Code":
            paragraph.paragraph_format.keep_together = True
            next_name = ""
            if index + 1 < len(paragraphs) and paragraphs[index + 1].style:
                next_name = paragraphs[index + 1].style.name
            paragraph.paragraph_format.keep_with_next = next_name == "Source Code"
        for run in paragraph.runs:
            if name in {"Title", "Subtitle", "Author", "Date", "Heading 1", "Heading 2", "Heading 3", "Caption", "TOC Heading"}:
                set_east_asia(run, "Microsoft YaHei")
            elif name != "Source Code":
                set_east_asia(run, "Microsoft YaHei" if run.bold else "SimSun")

    for table in doc.tables:
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.style = "Table"
        table.autofit = True
        set_table_borders(table)
        if not table.rows:
            continue
        set_repeat_table_header(table.rows[0])
        for row_index, row in enumerate(table.rows):
            for cell in row.cells:
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                if row_index == 0:
                    set_cell_shading(cell, NAVY)
                elif row_index % 2 == 0:
                    set_cell_shading(cell, PALE)
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.space_after = Pt(1.5)
                    paragraph.paragraph_format.line_spacing = 1.15
                    for run in paragraph.runs:
                        set_east_asia(run, "Microsoft YaHei" if row_index == 0 else "SimSun")
                        run.font.size = Pt(8.5 if len(table.columns) >= 4 else 9)
                        if row_index == 0:
                            run.font.bold = True
                            run.font.color.rgb = RGBColor.from_string(WHITE)

    max_width = Inches(6.15)
    for shape in doc.inline_shapes:
        if shape.width > max_width:
            ratio = max_width / shape.width
            shape.width = max_width
            shape.height = int(shape.height * ratio)

    settings = doc.settings._element
    update_fields = settings.find(qn("w:updateFields"))
    if update_fields is None:
        update_fields = OxmlElement("w:updateFields")
        settings.append(update_fields)
    # Pandoc already materializes the TOC. Asking Word to update all fields on
    # open makes this image-rich report repaginate for several minutes.
    update_fields.set(qn("w:val"), "false")
    doc.save(path)


def build() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    generate_charts()
    reference_path = TEMP_DIR / "reference.docx"
    markdown_path = TEMP_DIR / "report.md"
    style_reference(reference_path)
    prepare_markdown(markdown_path)
    command = [
        "pandoc",
        str(markdown_path),
        "--from=markdown+pipe_tables+fenced_code_blocks+link_attributes",
        "--to=docx",
        "--standalone",
        "--toc",
        "--toc-depth=3",
        "--metadata=toc-title:目录",
        f"--reference-doc={reference_path}",
        f"--resource-path={ROOT / 'doc/submission'};{ROOT}",
        f"--output={OUTPUT}",
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if completed.returncode != 0:
        print(completed.stdout, file=sys.stderr)
        print(completed.stderr, file=sys.stderr)
        raise SystemExit(completed.returncode)
    postprocess_docx(OUTPUT)
    print(OUTPUT)


def main() -> None:
    global SOURCE, OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    SOURCE = args.source.resolve()
    OUTPUT = args.output.resolve()
    build()


if __name__ == "__main__":
    main()
