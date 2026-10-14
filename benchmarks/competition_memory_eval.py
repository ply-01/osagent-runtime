"""Reproducible competition-oriented evaluation for the OS Agent memory module.

The runner intentionally uses only the Python standard library and the public
AgentMemoryModule / ExtractUpdatePipeline entry points. It evaluates the memory
layer without an LLM so retrieval and memory-policy behavior are not confounded
by a particular backbone model.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import re
import statistics
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

from os_agent.memory.interface import (
    ForgetRequest,
    RecallRequest,
    RememberRequest,
    UpdateRequest,
)
from os_agent.memory.mem0_archival import ArchivalMemoryStore
from os_agent.memory.mem0_core import CoreMemoryStore
from os_agent.memory.mem0_pipeline import ExtractUpdatePipeline
from os_agent.memory.mem0_schemas import FactCard
from os_agent.memory.memory_config import MemoryConfig
from os_agent.memory.module import AgentMemoryModule

SEED = 20260726


@dataclass(frozen=True)
class PreferenceCase:
    case_id: str
    text: str
    expected: bool
    category: str


@dataclass(frozen=True)
class RetrievalCase:
    case_id: str
    entity: str
    attribute: str
    value: str
    query: str
    category: str


PREFERENCE_CASES = [
    PreferenceCase("p01", "我喜欢把最终报告导出为 PDF。", True, "explicit"),
    PreferenceCase("p02", "我习惯先预览，再发送邮件。", True, "explicit"),
    PreferenceCase("p03", "我通常把项目文件放在工作区目录。", True, "explicit"),
    PreferenceCase("p04", "我希望回答简洁一些。", True, "explicit"),
    PreferenceCase("p05", "以后用 WPS 打开 docx 文件。", True, "explicit"),
    PreferenceCase("p06", "从现在起使用深色主题。", True, "explicit"),
    PreferenceCase("p07", "还是改成每周五归档吧。", True, "explicit"),
    PreferenceCase("p08", "I prefer concise Markdown reports.", True, "explicit"),
    PreferenceCase("p09", "Please always ask before sending contracts.", True, "explicit"),
    PreferenceCase("p10", "From now on, use the local office suite.", True, "explicit"),
    PreferenceCase("p11", "默认保存目录设为 D:/Work。", True, "explicit_no_marker"),
    PreferenceCase("p12", "邮件附件统一转成 PDF。", True, "explicit_no_marker"),
    PreferenceCase("p13", "最近五次都选择了静默安装。", True, "behavioral"),
    PreferenceCase("p14", "连续三次把结果保存为 Markdown。", True, "behavioral"),
    PreferenceCase("p15", "工具记录显示用户总在发送前预览。", True, "behavioral"),
    PreferenceCase("p16", "过去一周每天 18:00 执行备份。", True, "behavioral"),
    PreferenceCase("p17", "配置文件中 output_format=pdf。", True, "configuration"),
    PreferenceCase("p18", "user.theme is set to dark in settings.json.", True, "configuration"),
    PreferenceCase("p19", "最近的导出操作全部使用了 A4 页面。", True, "behavioral"),
    PreferenceCase("p20", "三次工具调用都选用了只读模式。", True, "behavioral"),
    PreferenceCase("n01", "请打开 README.md。", False, "negative"),
    PreferenceCase("n02", "PDF 文件已经生成。", False, "negative"),
    PreferenceCase("n03", "深色主题是什么？", False, "negative"),
    PreferenceCase("n04", "列出工作区中的文件。", False, "negative"),
    PreferenceCase("n05", "用户偏好表目前为空。", False, "negative"),
    PreferenceCase("n06", "这个工具支持 Markdown 输出。", False, "negative"),
    PreferenceCase("n07", "我喜欢这个方案吗？", False, "question"),
    PreferenceCase("n08", "他说‘我喜欢 PDF’，但那不是我的偏好。", False, "quoted"),
    PreferenceCase("n09", "示例句是：我习惯先预览。", False, "quoted"),
    PreferenceCase("n10", "不要假设我喜欢深色主题。", False, "negated"),
    PreferenceCase("n11", "I do not prefer PDF; this is a test sentence.", False, "negated"),
    PreferenceCase("n12", "Does the user prefer concise answers?", False, "question"),
    PreferenceCase("n13", "以后用什么软件需要再讨论。", False, "uncertain"),
    PreferenceCase("n14", "配置文件没有指定 output_format。", False, "negative"),
    PreferenceCase("n15", "工具偶然选择了一次只读模式。", False, "negative"),
    PreferenceCase("n16", "系统默认主题是浅色。", False, "negative"),
    PreferenceCase("n17", "报告可以导出为 PDF 或 DOCX。", False, "negative"),
    PreferenceCase("n18", "请解释‘我希望回答简洁’这句话。", False, "quoted"),
    PreferenceCase("n19", "No preference has been configured.", False, "negative"),
    PreferenceCase("n20", "The application always writes an audit log.", False, "system_behavior"),
]


RETRIEVAL_CASES = [
    RetrievalCase("r01", "project_alpha", "save_dir", "D:/Work/Alpha", "project_alpha save_dir", "exact_keyword"),
    RetrievalCase("r02", "user_beta", "output_format", "PDF", "user_beta output_format", "exact_keyword"),
    RetrievalCase("r03", "mail_gamma", "approval_rule", "confirm_first", "mail_gamma approval_rule", "exact_keyword"),
    RetrievalCase("r04", "backup_delta", "schedule", "Friday_1800", "backup_delta schedule", "exact_keyword"),
    RetrievalCase("r05", "office_epsilon", "preferred_app", "WPS", "office_epsilon preferred_app", "exact_keyword"),
    RetrievalCase("r06", "display_zeta", "theme", "dark", "display_zeta theme", "exact_keyword"),
    RetrievalCase("r07", "archive_eta", "retention_days", "30", "archive_eta retention_days", "exact_keyword"),
    RetrievalCase("r08", "security_theta", "send_policy", "manual_review", "security_theta send_policy", "exact_keyword"),
    RetrievalCase("r09", "report_iota", "page_size", "A4", "report_iota page_size", "exact_keyword"),
    RetrievalCase("r10", "shell_kappa", "mode", "read_only", "shell_kappa mode", "exact_keyword"),
    RetrievalCase("r11", "项目甲", "默认保存目录", "D:/Work/Alpha", "甲项目的文件应当放到哪里", "chinese_paraphrase"),
    RetrievalCase("r12", "用户乙", "输出格式", "PDF", "乙用户交付文档要用哪种格式", "chinese_paraphrase"),
    RetrievalCase("r13", "邮件任务丙", "审批规则", "发送前确认", "丙任务能否直接发出邮件", "chinese_paraphrase"),
    RetrievalCase("r14", "备份任务丁", "执行时间", "周五十八点", "丁任务什么时候跑备份", "chinese_paraphrase"),
    RetrievalCase("r15", "办公场景戊", "首选应用", "WPS", "戊场景用哪个办公软件", "chinese_paraphrase"),
    RetrievalCase("r16", "显示设置己", "主题", "深色", "己的界面外观采用什么模式", "chinese_paraphrase"),
    RetrievalCase("r17", "归档规则庚", "保留天数", "三十天", "庚规则需要保存多久", "chinese_paraphrase"),
    RetrievalCase("r18", "安全策略辛", "发送策略", "人工复核", "辛策略要求谁检查后才能发送", "chinese_paraphrase"),
    RetrievalCase("r19", "报告模板壬", "纸张尺寸", "A4", "壬模板使用多大的页面", "chinese_paraphrase"),
    RetrievalCase("r20", "命令策略癸", "执行模式", "只读", "癸策略允不允许修改文件", "chinese_paraphrase"),
    RetrievalCase("r21", "workspace_orchid", "root_path", "D:/Cases/Orchid", "请问 workspace_orchid 当前 root_path 是什么", "noisy_keyword"),
    RetrievalCase("r22", "export_maple", "target_type", "PDF_A", "帮我查一下 export_maple 使用的 target_type", "noisy_keyword"),
    RetrievalCase("r23", "meeting_cedar", "room", "C-302", "根据历史记录 meeting_cedar 应预订哪个 room", "noisy_keyword"),
    RetrievalCase("r24", "network_pine", "profile", "office_lan", "请恢复 network_pine 对应的 profile", "noisy_keyword"),
    RetrievalCase("r25", "printer_birch", "duplex", "enabled", "printer_birch 的 duplex 配置是否开启", "noisy_keyword"),
    RetrievalCase("r26", "scan_lotus", "resolution", "300dpi", "执行 scan_lotus 时 resolution 应设多少", "noisy_keyword"),
    RetrievalCase("r27", "calendar_iris", "timezone", "Asia/Shanghai", "calendar_iris 使用哪个 timezone", "noisy_keyword"),
    RetrievalCase("r28", "sync_rose", "conflict_mode", "keep_newest", "sync_rose 遇到冲突时 conflict_mode 是什么", "noisy_keyword"),
    RetrievalCase("r29", "terminal_elm", "encoding", "UTF-8", "terminal_elm 的 encoding 配置", "noisy_keyword"),
    RetrievalCase("r30", "document_ash", "naming_rule", "date_prefix", "document_ash 应采用哪个 naming_rule", "noisy_keyword"),
]


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def classification_metrics(labels: list[bool], predictions: list[bool]) -> dict[str, float | int]:
    tp = sum(label and prediction for label, prediction in zip(labels, predictions))
    tn = sum(not label and not prediction for label, prediction in zip(labels, predictions))
    fp = sum(not label and prediction for label, prediction in zip(labels, predictions))
    fn = sum(label and not prediction for label, prediction in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "samples": len(labels),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": round((tp + tn) / len(labels), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def make_event(text: str, case_id: str) -> dict[str, Any]:
    return {
        "event_type": "message_start",
        "session_id": f"pref-{case_id}",
        "task_id": f"task-{case_id}",
        "event_id": f"event-{case_id}",
        "timestamp": "2026-07-26T00:00:00+00:00",
        "payload": {"role": "user", "content": text},
    }


def evaluate_preference_extraction() -> dict[str, Any]:
    labels: list[bool] = []
    predictions: list[bool] = []
    failures: list[dict[str, Any]] = []
    by_category: dict[str, dict[str, int]] = {}
    with TemporaryDirectory() as tmp:
        base = Path(tmp)
        for case in PREFERENCE_CASES:
            case_root = base / case.case_id
            config = MemoryConfig(case_root / "config.json")
            core = CoreMemoryStore(case_root / "memory", config=config)
            archival = ArchivalMemoryStore(case_root / "memory", config=config)
            pipeline = ExtractUpdatePipeline(core, archival, config=config)
            facts = pipeline.process_session([make_event(case.text, case.case_id)], f"pref-{case.case_id}")
            prediction = any("preference" in fact.tags for fact in facts)
            labels.append(case.expected)
            predictions.append(prediction)
            counts = by_category.setdefault(case.category, {"samples": 0, "correct": 0})
            counts["samples"] += 1
            counts["correct"] += int(prediction == case.expected)
            if prediction != case.expected:
                failures.append({**asdict(case), "predicted": prediction})
    category_metrics = {
        name: {**counts, "accuracy": round(counts["correct"] / counts["samples"], 4)}
        for name, counts in sorted(by_category.items())
    }
    return {
        **classification_metrics(labels, predictions),
        "by_category": category_metrics,
        "failures": failures,
    }


def build_retrieval_module(root: Path) -> AgentMemoryModule:
    config = MemoryConfig(root / "config.json")
    module = AgentMemoryModule(root / "memory", config=config)
    for case in RETRIEVAL_CASES:
        module.remember(
            RememberRequest(
                content=f"{case.entity}.{case.attribute} = {case.value}",
                entity=case.entity,
                attribute=case.attribute,
                value=case.value,
                evidence=f"benchmark:{case.case_id}",
                tags=["benchmark", case.category],
                confidence=0.9,
                metadata={"importance": 0.8},
            )
        )
    for index in range(200):
        module.remember(
            RememberRequest(
                content=f"distractor_{index}.note = unrelated_value_{index}",
                entity=f"distractor_{index}",
                attribute="note",
                value=f"unrelated_value_{index}",
                evidence="benchmark distractor",
                tags=["distractor"],
                confidence=0.6,
            )
        )
    return module


def retrieval_corpus() -> list[RetrievalCase]:
    corpus = list(RETRIEVAL_CASES)
    corpus.extend(
        RetrievalCase(
            case_id=f"d{index:03d}",
            entity=f"distractor_{index}",
            attribute="note",
            value=f"unrelated_value_{index}",
            query="",
            category="distractor",
        )
        for index in range(200)
    )
    return corpus


def is_expected_record(record: Any, case: RetrievalCase) -> bool:
    return record.entity == case.entity and record.attribute == case.attribute and record.value == case.value


def retrieval_metrics(ranks: list[int | None]) -> dict[str, float | int]:
    total = len(ranks)
    return {
        "queries": total,
        "recall_at_1": round(sum(rank == 1 for rank in ranks) / total, 4),
        "recall_at_3": round(sum(rank is not None and rank <= 3 for rank in ranks) / total, 4),
        "recall_at_5": round(sum(rank is not None and rank <= 5 for rank in ranks) / total, 4),
        "mrr": round(sum(1 / rank for rank in ranks if rank is not None) / total, 4),
    }


def tokenize_bigram(text: str) -> set[str]:
    lowered = text.lower()
    terms = set(re.findall(r"[a-z0-9_./\\-]{2,}", lowered))
    for sequence in re.findall(r"[\u4e00-\u9fff]+", lowered):
        terms.update(sequence[index : index + 2] for index in range(max(len(sequence) - 1, 0)))
    return terms


def keyword_rag_rank(case: RetrievalCase, corpus: Iterable[RetrievalCase], limit: int = 5) -> int | None:
    query_terms = tokenize_bigram(case.query)
    scored: list[tuple[int, str, RetrievalCase]] = []
    for item in corpus:
        text = f"{item.entity} {item.attribute} {item.value}"
        score = len(query_terms & tokenize_bigram(text))
        if score:
            scored.append((score, item.case_id, item))
    scored.sort(key=lambda row: (-row[0], row[1]))
    for rank, (_, _, item) in enumerate(scored[:limit], 1):
        if item.case_id == case.case_id:
            return rank
    return None


def evaluate_retrieval() -> dict[str, Any]:
    interface_ranks: list[int | None] = []
    keyword_ranks: list[int | None] = []
    category_ranks: dict[str, list[int | None]] = {}
    failures: list[dict[str, Any]] = []
    baseline_corpus = retrieval_corpus()
    with TemporaryDirectory() as tmp:
        module = build_retrieval_module(Path(tmp))
        for case in RETRIEVAL_CASES:
            result = module.recall(RecallRequest(query=case.query, scope="ltm", limit=5))
            rank = next(
                (index for index, record in enumerate(result.records, 1) if is_expected_record(record, case)),
                None,
            )
            interface_ranks.append(rank)
            category_ranks.setdefault(case.category, []).append(rank)
            keyword_ranks.append(keyword_rag_rank(case, baseline_corpus))
            if rank is None:
                failures.append({**asdict(case), "returned": [record.content for record in result.records]})
    return {
        "current_agent_memory": {
            **retrieval_metrics(interface_ranks),
            "by_category": {
                category: retrieval_metrics(ranks) for category, ranks in sorted(category_ranks.items())
            },
        },
        "baselines": {
            "no_memory": retrieval_metrics([None] * len(RETRIEVAL_CASES)),
            "keyword_rag_char_bigram": retrieval_metrics(keyword_ranks),
        },
        "failures": failures,
    }


def make_pipeline(root: Path) -> tuple[ExtractUpdatePipeline, CoreMemoryStore, ArchivalMemoryStore]:
    config = MemoryConfig(root / "config.json")
    core = CoreMemoryStore(root / "memory", config=config)
    archival = ArchivalMemoryStore(root / "memory", config=config)
    return ExtractUpdatePipeline(core, archival, config=config), core, archival


def evaluate_conflicts() -> dict[str, Any]:
    structured_passes = 0
    structured_total = 10
    with TemporaryDirectory() as tmp:
        module = AgentMemoryModule(Path(tmp) / "memory", config=MemoryConfig(Path(tmp) / "config.json"))
        for index in range(structured_total):
            entity = f"structured_user_{index}"
            attribute = f"format_pref_{index}"
            old_value = f"DOCX_{index}"
            new_value = f"PDF_{index}"
            module.remember(RememberRequest(content=old_value, entity=entity, attribute=attribute, value=old_value))
            module.update(UpdateRequest(content=new_value, entity=entity, attribute=attribute, value=new_value))
            result = module.recall(RecallRequest(query=f"{entity} {attribute}", scope="ltm", limit=5))
            values = [record.value for record in result.records if record.entity == entity]
            structured_passes += int(new_value in values and old_value not in values)

    true_conflict_passes = 0
    independent_passes = 0
    natural_failures: list[dict[str, Any]] = []
    true_conflicts = [
        ("我喜欢导出 DOCX。", "以后改成使用 PDF。"),
        ("我习惯用浅色主题。", "从现在起使用深色主题。"),
        ("我通常周一备份。", "以后改成周五备份。"),
        ("我希望回复详细。", "接下来改成回复简洁。"),
        ("I prefer WPS for documents.", "From now on, use LibreOffice."),
    ]
    independent_pairs = [
        ("我喜欢导出 PDF。", "我习惯使用深色主题。"),
        ("我希望回复简洁。", "我通常周五执行备份。"),
        ("I prefer WPS for documents.", "I always preview mail before sending."),
        ("我习惯文件名带日期。", "我希望扫描分辨率使用三百 DPI。"),
        ("我通常使用 A4 页面。", "我喜欢把附件放到项目目录。"),
    ]
    for index, (old_text, new_text) in enumerate(true_conflicts):
        with TemporaryDirectory() as tmp:
            pipeline, core, _ = make_pipeline(Path(tmp))
            pipeline.process_session([make_event(old_text, f"tc-{index}-a")], f"tc-{index}-a")
            pipeline.process_session([make_event(new_text, f"tc-{index}-b")], f"tc-{index}-b")
            facts = core.list_by_entity("用户")
            passed = len(facts) == 1 and facts[0].value == new_text
            true_conflict_passes += int(passed)
            if not passed:
                natural_failures.append({"type": "true_conflict", "old": old_text, "new": new_text})
    for index, (first_text, second_text) in enumerate(independent_pairs):
        with TemporaryDirectory() as tmp:
            pipeline, core, _ = make_pipeline(Path(tmp))
            pipeline.process_session([make_event(first_text, f"ip-{index}-a")], f"ip-{index}-a")
            pipeline.process_session([make_event(second_text, f"ip-{index}-b")], f"ip-{index}-b")
            values = {fact.value for fact in core.list_by_entity("用户")}
            passed = first_text in values and second_text in values
            independent_passes += int(passed)
            if not passed:
                natural_failures.append({
                    "type": "independent_preferences",
                    "first": first_text,
                    "second": second_text,
                    "retained": sorted(str(value) for value in values),
                })
    natural_total = len(true_conflicts) + len(independent_pairs)
    natural_passes = true_conflict_passes + independent_passes
    return {
        "structured_interface_update_accuracy": round(structured_passes / structured_total, 4),
        "structured_cases": structured_total,
        "natural_pipeline_balanced_accuracy": round(natural_passes / natural_total, 4),
        "natural_pipeline_cases": natural_total,
        "true_conflict_latest_wins_accuracy": round(true_conflict_passes / len(true_conflicts), 4),
        "independent_preference_preservation_accuracy": round(independent_passes / len(independent_pairs), 4),
        "failures": natural_failures,
    }


def evaluate_forgetting() -> dict[str, Any]:
    logical_passes = 0
    physical_passes = 0
    hard_flag_honored = 0
    total = 10
    failures: list[dict[str, Any]] = []
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        module = AgentMemoryModule(root / "memory", config=MemoryConfig(root / "config.json"))
        for index in range(total):
            entity = f"forget_user_{index}"
            attribute = f"secret_{index}"
            value = f"sensitive_value_{index}"
            module.remember(RememberRequest(content=value, entity=entity, attribute=attribute, value=value))
            forgotten = module.forget(
                ForgetRequest(scope="ltm", entity=entity, attribute=attribute, hard=True)
            )
            recalled = module.recall(RecallRequest(query=f"{entity} {attribute}", scope="ltm", limit=5))
            logical = forgotten.ok and all(record.value != value for record in recalled.records)
            logical_passes += int(logical)
            core_text = (root / "memory" / "core_facts.json").read_text(encoding="utf-8")
            archive_path = root / "memory" / "archive_facts.json"
            archive_text = archive_path.read_text(encoding="utf-8") if archive_path.exists() else ""
            physical = value not in core_text and value not in archive_text
            physical_passes += int(physical)
            hard_flag_honored += int(physical)
            if not logical or not physical:
                failures.append({
                    "case_id": f"f{index + 1:02d}",
                    "logical_recall_blocked": logical,
                    "physically_erased": physical,
                })
    return {
        "logical_forgetting_success_rate": round(logical_passes / total, 4),
        "physical_erasure_success_rate": round(physical_passes / total, 4),
        "hard_delete_flag_honored_rate": round(hard_flag_honored / total, 4),
        "cases": total,
        "failures": failures,
    }


def fact_payload(index: int) -> dict[str, Any]:
    return FactCard(
        entity=f"entity_{index:05d}",
        attribute=f"attribute_{index % 97:02d}",
        value=f"value_{index:05d}",
        evidence="synthetic latency fixture",
        source_session_id="latency",
        source_event_id=f"latency-{index}",
        confidence=0.8,
        tags=["latency", f"bucket_{index % 31}"],
        importance=0.6,
    ).to_dict()


def evaluate_latency() -> dict[str, Any]:
    rng = random.Random(SEED)
    results: dict[str, Any] = {}
    with TemporaryDirectory() as tmp:
        base = Path(tmp)
        for size in (100, 1000, 10000):
            root = base / str(size)
            memory_root = root / "memory"
            memory_root.mkdir(parents=True, exist_ok=True)
            (memory_root / "core_facts.json").write_text(
                json.dumps([fact_payload(index) for index in range(size)], ensure_ascii=False),
                encoding="utf-8",
            )
            module = AgentMemoryModule(memory_root, config=MemoryConfig(root / "config.json"))
            query_indices = [rng.randrange(size) for _ in range(120)]
            for index in query_indices[:10]:
                module.recall(RecallRequest(query=f"entity_{index:05d}", scope="ltm", limit=5))
            samples: list[float] = []
            hits = 0
            for index in query_indices:
                start = perf_counter()
                result = module.recall(
                    RecallRequest(query=f"entity_{index:05d}", scope="ltm", limit=5)
                )
                samples.append((perf_counter() - start) * 1000)
                hits += int(any(record.entity == f"entity_{index:05d}" for record in result.records))
            results[str(size)] = {
                "queries": len(samples),
                "hit_rate": round(hits / len(samples), 4),
                "latency_ms": {
                    "mean": round(statistics.fmean(samples), 3),
                    "p50": round(percentile(samples, 0.50), 3),
                    "p95": round(percentile(samples, 0.95), 3),
                    "p99": round(percentile(samples, 0.99), 3),
                    "max": round(max(samples), 3),
                },
                "under_500ms": percentile(samples, 0.95) <= 500,
            }
    return results


def git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return completed.stdout.strip()

    status = run("git", "status", "--short")
    return {
        "commit": run("git", "rev-parse", "--short", "HEAD"),
        "dirty": bool(status),
        "changed_path_count": len(status.splitlines()) if status else 0,
    }


def evaluate_all() -> dict[str, Any]:
    started = perf_counter()
    return {
        "protocol": {
            "name": "OS Agent competition memory evaluation",
            "version": "1.0",
            "seed": SEED,
            "evaluated_at": datetime.now(UTC).isoformat(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "git": git_metadata(),
            "llm_used": False,
            "notes": [
                "Memory behavior is evaluated independently from a backbone LLM.",
                "Retrieval fixtures are written directly only for the 100/1k/10k latency setup; all measured queries use AgentMemoryModule.recall.",
            ],
        },
        "preference_extraction": evaluate_preference_extraction(),
        "knowledge_retrieval": evaluate_retrieval(),
        "conflict_handling": evaluate_conflicts(),
        "forgetting": evaluate_forgetting(),
        "latency_by_active_fact_count": evaluate_latency(),
        "elapsed_seconds": round(perf_counter() - started, 3),
    }


def sanitize_evaluation_output(value: Any) -> Any:
    """Remove per-case failures so published reports contain aggregates only."""
    if isinstance(value, dict):
        return {
            key: sanitize_evaluation_output(item)
            for key, item in value.items()
            if key != "failures"
        }
    if isinstance(value, list):
        return [sanitize_evaluation_output(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("doc/evaluation/results/competition-memory-eval.json"),
        help="Path for the machine-readable result JSON.",
    )
    args = parser.parse_args()
    results = evaluate_all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    public_results = sanitize_evaluation_output(results)
    args.output.write_text(
        json.dumps(public_results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(public_results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
