"""Generate a sealed, competition-aligned Kylin OS memory evaluation set."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


VERSION = "KylinMemoryEval-v1"
DEFAULT_ROOT = Path("data/kylin-memory-eval-v1")

PREFERENCE_CATEGORIES = (
    "explicit_zh",
    "explicit_en",
    "explicit_avoid",
    "configuration",
    "repeated_behavior",
    "question",
    "quoted_or_reported",
    "one_off_request",
    "uncertain",
    "system_statement",
)

PREFERENCE_DIMENSIONS = (
    ("document.output_format", "文档输出格式", "document output format", ("PDF", "ODT", "DOCX")),
    ("ui.theme", "界面主题", "interface theme", ("深色", "浅色", "高对比度")),
    ("application.preferred_tool", "办公软件", "office application", ("WPS", "LibreOffice", "OnlyOffice")),
    ("response.style", "回复风格", "response style", ("简洁", "详细", "分步骤")),
    ("document.page_size", "页面尺寸", "page size", ("A4", "A5", "Letter")),
    ("scanner.resolution", "扫描分辨率", "scan resolution", ("300DPI", "600DPI", "1200DPI")),
    ("language.preferred_language", "回复语言", "reply language", ("中文", "English", "双语")),
    ("notification.delivery_mode", "通知方式", "notification mode", ("静音", "横幅", "声音")),
    ("input.preferred_method", "输入方式", "input method", ("拼音", "五笔", "英文键盘")),
    ("operation.execution_mode", "执行模式", "execution mode", ("只读", "安全", "静默")),
)

RETRIEVAL_ATTRIBUTES = (
    ("output_format", "交付文件使用什么格式", "delivery file format", ("PDF", "ODT", "DOCX", "XPS")),
    ("save_directory", "文件保存到哪个目录", "file save directory", ("/home/user/Documents", "/data/projects", "/srv/archive", "/home/user/Desktop")),
    ("preferred_application", "应使用哪个应用程序", "preferred application", ("WPS", "LibreOffice", "Dolphin", "Okular")),
    ("approval_policy", "执行前采用什么审批规则", "approval policy", ("confirm_first", "manual_review", "two_person_check", "auto_approve")),
    ("backup_schedule", "备份在什么时间执行", "backup schedule", ("周五18:00", "每天23:00", "周一09:00", "每月1日02:00")),
    ("network_profile", "应该连接哪个网络配置", "network profile", ("office_lan", "secure_vpn", "offline", "guest_wifi")),
    ("naming_rule", "文件采用什么命名规则", "file naming rule", ("date_prefix", "project_prefix", "sequence_suffix", "unchanged")),
    ("retention_days", "记录需要保留多少天", "retention period", ("7", "30", "90", "365")),
    ("printer_mode", "打印机使用什么模式", "printer mode", ("duplex", "single_sided", "grayscale", "booklet")),
    ("encoding", "文本使用什么编码", "text encoding", ("UTF-8", "GB18030", "UTF-16LE", "ASCII")),
)

CONFLICT_DIMENSIONS = (
    ("output_format", "文档输出格式", ("PDF", "ODT", "DOCX")),
    ("theme", "界面主题", ("深色", "浅色", "高对比度")),
    ("office_app", "办公软件", ("WPS", "LibreOffice", "OnlyOffice")),
    ("reply_style", "回复风格", ("简洁", "详细", "分步骤")),
    ("page_size", "页面尺寸", ("A4", "A5", "Letter")),
    ("backup_day", "备份时间", ("周一", "周五", "每天")),
    ("notification", "通知方式", ("静音", "横幅", "声音")),
    ("input_method", "输入法", ("拼音", "五笔", "英文键盘")),
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def message_event(case_id: str, session: int, index: int, text: str) -> dict[str, Any]:
    return {
        "event_type": "message_start",
        "session_id": f"{case_id}-s{session}",
        "task_id": f"{case_id}-task",
        "event_id": f"{case_id}-s{session}-e{index}",
        "timestamp": f"2026-06-{session + 1:02d}T08:{index:02d}:00+00:00",
        "payload": {"role": "user", "content": text},
    }


def tool_event(
    case_id: str,
    session: int,
    index: int,
    tool: str,
    parameter: str,
    value: str,
) -> dict[str, Any]:
    return {
        "event_type": "tool_result",
        "session_id": f"{case_id}-s{session}",
        "task_id": f"{case_id}-task",
        "event_id": f"{case_id}-s{session}-e{index}",
        "timestamp": f"2026-06-{session + 1:02d}T09:{index:02d}:00+00:00",
        "payload": {
            "tool_name": tool,
            "status": "success",
            "input": {parameter: value},
            "output": {"ok": True},
        },
    }


def preference_case(
    rng: random.Random,
    case_id: str,
    category: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    attribute, zh_name, en_name, values = rng.choice(PREFERENCE_DIMENSIONS)
    value = rng.choice(values)
    sessions: list[list[dict[str, Any]]]
    expected = category in PREFERENCE_CATEGORIES[:5]
    expected_value: str | None = value if expected else None
    if category == "explicit_zh":
        text = rng.choice((
            f"我希望以后{zh_name}固定使用{value}。",
            f"我通常把{zh_name}设置为{value}，后续也请保持。",
            f"从现在开始，我偏好{zh_name}采用{value}。",
        ))
        sessions = [[message_event(case_id, 0, 0, text)]]
    elif category == "explicit_en":
        text = rng.choice((
            f"I prefer {value} for the {en_name} in future tasks.",
            f"Please always keep the {en_name} set to {value}.",
            f"My usual {en_name} is {value}; use it from now on.",
        ))
        sessions = [[message_event(case_id, 0, 0, text)]]
    elif category == "explicit_avoid":
        text = rng.choice((
            f"我不喜欢{zh_name}使用{value}，以后请避开它。",
            f"I want to avoid {value} for the {en_name} in future work.",
            f"我的长期要求是不要把{zh_name}设成{value}。",
        ))
        sessions = [[message_event(case_id, 0, 0, text)]]
    elif category == "configuration":
        text = rng.choice((
            f"用户配置中 default_{attribute.replace('.', '_')}={value}。",
            f"配置文件已经把{zh_name}的默认值设为{value}。",
            f"The user profile sets the default {en_name} to {value}.",
        ))
        sessions = [[message_event(case_id, 0, 0, text)]]
    elif category == "repeated_behavior":
        tool = rng.choice(("office.export", "desktop.configure", "workflow.execute"))
        parameter = attribute.replace(".", "_")
        sessions = [
            [tool_event(case_id, session, 0, tool, parameter, value)]
            for session in range(3)
        ]
    elif category == "question":
        text = rng.choice((
            f"我以前是不是喜欢把{zh_name}设为{value}？",
            f"Does the user prefer {value} for the {en_name}?",
            f"请检查是否有人要求{zh_name}使用{value}。",
        ))
        sessions = [[message_event(case_id, 0, 0, text)]]
    elif category == "quoted_or_reported":
        text = rng.choice((
            f"示例文档写着“我希望{zh_name}使用{value}”，这不是我的设置。",
            f"客户说他偏好{en_name}使用{value}，不要记成我的偏好。",
            f"The manual uses 'I prefer {value}' as an example sentence only.",
        ))
        sessions = [[message_event(case_id, 0, 0, text)]]
    elif category == "one_off_request":
        text = rng.choice((
            f"只把这一次的{zh_name}临时设为{value}。",
            f"Use {value} for the {en_name} in this task only.",
            f"本次操作先用{value}，不要保存为长期设置。",
        ))
        sessions = [[message_event(case_id, 0, 0, text)]]
    elif category == "uncertain":
        text = rng.choice((
            f"以后{zh_name}也许用{value}，但我还没有决定。",
            f"I might use {value} for the {en_name}, but this is undecided.",
            f"{value}只是候选方案，等确认后再记录偏好。",
        ))
        sessions = [[message_event(case_id, 0, 0, text)]]
    else:
        text = rng.choice((
            f"系统默认的{zh_name}是{value}。",
            f"The application supports {value} as a possible {en_name}.",
            f"管理员文档建议将{zh_name}设为{value}。",
        ))
        sessions = [[message_event(case_id, 0, 0, text)]]
    return (
        {
            "case_id": case_id,
            "category": category,
            "source_type": "tool_behavior" if category == "repeated_behavior" else "user_message",
            "sessions": sessions,
        },
        {
            "case_id": case_id,
            "expected_preference": expected,
            "expected_attribute": attribute if expected else None,
            "expected_value": expected_value,
        },
    )


def generate_preferences(rng: random.Random) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    serial = 0
    for category in PREFERENCE_CATEGORIES:
        for _ in range(50):
            serial += 1
            pairs.append(preference_case(rng, f"pref-{serial:04d}", category))
    rng.shuffle(pairs)
    return [row[0] for row in pairs], [row[1] for row in pairs]


def retrieval_value(entity_index: int, attribute_index: int, values: tuple[str, ...]) -> str:
    base = values[(entity_index * 3 + attribute_index) % len(values)]
    if attribute_index == 7:
        return str(int(base) + (entity_index % 4) * 7)
    return base


def generate_retrieval(
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    corpus: list[dict[str, Any]] = []
    for entity_index in range(1000):
        entity = f"kylin_task_{entity_index:04d}"
        display = f"麒麟任务-{entity_index:04d}"
        for attribute_index, (attribute, zh_prompt, en_prompt, values) in enumerate(RETRIEVAL_ATTRIBUTES):
            value = retrieval_value(entity_index, attribute_index, values)
            fact_id = f"kmr-{entity_index:04d}-{attribute_index:02d}"
            evidence = (
                f"{display}：{zh_prompt.replace('什么', value).replace('哪个', value).replace('哪里', value).replace('多少天', value)}。"
                if entity_index % 2 == 0
                else f"For {display}, {en_prompt} is {value}."
            )
            corpus.append(
                {
                    "fact_id": fact_id,
                    "entity": entity,
                    "attribute": attribute,
                    "value": value,
                    "evidence": evidence,
                    "display_name": display,
                    "source_type": ("user_configuration", "tool_result", "behavior_log")[
                        (entity_index + attribute_index) % 3
                    ],
                }
            )

    selected_entities = rng.sample(range(1000), 500)
    categories = [category for category in ("exact", "zh_paraphrase", "en_paraphrase", "cross_lingual", "multi_fact") for _ in range(100)]
    rng.shuffle(categories)
    inputs: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []
    for query_index, (entity_index, category) in enumerate(zip(selected_entities, categories), start=1):
        entity = f"kylin_task_{entity_index:04d}"
        display = f"麒麟任务-{entity_index:04d}"
        attribute_index = (entity_index * 7 + query_index) % len(RETRIEVAL_ATTRIBUTES)
        attribute, zh_prompt, en_prompt, _ = RETRIEVAL_ATTRIBUTES[attribute_index]
        relevant = [f"kmr-{entity_index:04d}-{attribute_index:02d}"]
        if category == "exact":
            query = f"{entity} {attribute} current value"
        elif category == "zh_paraphrase":
            query = f"根据历史配置，{display}{zh_prompt}？不要参考其他任务。"
        elif category == "en_paraphrase":
            query = f"According to memory, for {display}, what is the {en_prompt}?"
        elif category == "cross_lingual":
            query = (
                f"For {display}, recover the stored {en_prompt}."
                if entity_index % 2 == 0
                else f"请查找{display}的历史设置：{zh_prompt}？"
            )
        else:
            second_index = (attribute_index + 3) % len(RETRIEVAL_ATTRIBUTES)
            second = RETRIEVAL_ATTRIBUTES[second_index]
            relevant.append(f"kmr-{entity_index:04d}-{second_index:02d}")
            query = f"请同时找出{display}的{zh_prompt}，以及{second[1]}。"
        case_id = f"ret-{query_index:04d}"
        inputs.append({"case_id": case_id, "category": category, "query": query})
        qrels.append({"case_id": case_id, "relevant_fact_ids": sorted(relevant)})
    return corpus, inputs, qrels


def structured_operation(action: str, entity: str, attribute: str, value: str) -> dict[str, str]:
    return {"action": action, "entity": entity, "attribute": attribute, "value": value}


def natural_session(case_id: str, index: int, text: str) -> list[dict[str, Any]]:
    return [message_event(case_id, index, 0, text)]


def generate_conflicts(rng: random.Random) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    categories = [category for category in ("latest_override", "independent", "scoped", "duplicate", "revert") for _ in range(100)]
    rng.shuffle(categories)
    inputs: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for index, category in enumerate(categories, start=1):
        case_id = f"conf-{index:04d}"
        attr_a, zh_a, values_a = rng.choice(CONFLICT_DIMENSIONS)
        attr_b, zh_b, values_b = rng.choice([row for row in CONFLICT_DIMENSIONS if row[0] != attr_a])
        old, new = rng.sample(list(values_a), 2)
        other = rng.choice(values_b)
        entity = f"user_{index:04d}"
        operations: list[dict[str, str]]
        sessions: list[list[dict[str, Any]]]
        expected_state: list[dict[str, str]]
        expected_records: list[list[str]]
        forbidden_records: list[list[str]]
        if category == "latest_override":
            operations = [
                structured_operation("remember", entity, attr_a, old),
                structured_operation("update", entity, attr_a, new),
            ]
            sessions = [
                natural_session(case_id, 0, f"我希望以后{zh_a}使用{old}。"),
                natural_session(case_id, 1, f"我改变主意了，现在更喜欢{zh_a}使用{new}。"),
            ]
            expected_state = [{"entity": entity, "attribute": attr_a, "value": new}]
            expected_records = [[zh_a, new]]
            forbidden_records = [[zh_a, old]]
        elif category == "independent":
            operations = [
                structured_operation("remember", entity, attr_a, old),
                structured_operation("remember", entity, attr_b, other),
            ]
            sessions = [
                natural_session(case_id, 0, f"我希望以后{zh_a}使用{old}。"),
                natural_session(case_id, 1, f"另外，我通常把{zh_b}设为{other}。"),
            ]
            expected_state = [
                {"entity": entity, "attribute": attr_a, "value": old},
                {"entity": entity, "attribute": attr_b, "value": other},
            ]
            expected_records = [[zh_a, old], [zh_b, other]]
            forbidden_records = []
        elif category == "scoped":
            scope_a = f"项目甲{index:03d}"
            scope_b = f"项目乙{index:03d}"
            operations = [
                structured_operation("remember", scope_a, attr_a, old),
                structured_operation("remember", scope_b, attr_a, new),
            ]
            sessions = [
                natural_session(case_id, 0, f"在{scope_a}中，我希望{zh_a}使用{old}。"),
                natural_session(case_id, 1, f"在{scope_b}中，我希望{zh_a}使用{new}。"),
            ]
            expected_state = [
                {"entity": scope_a, "attribute": attr_a, "value": old},
                {"entity": scope_b, "attribute": attr_a, "value": new},
            ]
            expected_records = [[scope_a, old], [scope_b, new]]
            forbidden_records = []
        elif category == "duplicate":
            operations = [
                structured_operation("remember", entity, attr_a, old),
                structured_operation("remember", entity, attr_a, old),
            ]
            text = f"我希望以后{zh_a}使用{old}。"
            sessions = [natural_session(case_id, 0, text), natural_session(case_id, 1, text)]
            expected_state = [{"entity": entity, "attribute": attr_a, "value": old}]
            expected_records = [[zh_a, old]]
            forbidden_records = []
        else:
            operations = [
                structured_operation("remember", entity, attr_a, old),
                structured_operation("update", entity, attr_a, new),
                structured_operation("update", entity, attr_a, old),
            ]
            sessions = [
                natural_session(case_id, 0, f"我希望以后{zh_a}使用{old}。"),
                natural_session(case_id, 1, f"暂时改成{zh_a}使用{new}。"),
                natural_session(case_id, 2, f"最终还是恢复为{zh_a}使用{old}。"),
            ]
            expected_state = [{"entity": entity, "attribute": attr_a, "value": old}]
            expected_records = [[zh_a, old]]
            forbidden_records = [[zh_a, new]]
        inputs.append(
            {
                "case_id": case_id,
                "category": category,
                "structured_operations": operations,
                "natural_sessions": sessions,
            }
        )
        labels.append(
            {
                "case_id": case_id,
                "expected_state": sorted(expected_state, key=lambda row: (row["entity"], row["attribute"])),
                "expected_records": expected_records,
                "forbidden_records": forbidden_records,
            }
        )
    return inputs, labels


def validate(
    preferences: tuple[list[dict[str, Any]], list[dict[str, Any]]],
    retrieval: tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]],
    conflicts: tuple[list[dict[str, Any]], list[dict[str, Any]]],
) -> None:
    preference_inputs, preference_labels = preferences
    corpus, retrieval_inputs, qrels = retrieval
    conflict_inputs, conflict_labels = conflicts
    assert len(preference_inputs) == len(preference_labels) == 500
    assert len(corpus) == 10000
    assert len(retrieval_inputs) == len(qrels) == 500
    assert len(conflict_inputs) == len(conflict_labels) == 500
    assert sum(row["expected_preference"] for row in preference_labels) == 250
    corpus_ids = {row["fact_id"] for row in corpus}
    assert len(corpus_ids) == len(corpus)
    assert all(set(row["relevant_fact_ids"]) <= corpus_ids for row in qrels)
    for inputs, labels in (
        (preference_inputs, preference_labels),
        (retrieval_inputs, qrels),
        (conflict_inputs, conflict_labels),
    ):
        assert {row["case_id"] for row in inputs} == {row["case_id"] for row in labels}
    serialized_inputs = json.dumps(
        {"preference": preference_inputs, "retrieval": retrieval_inputs, "conflict": conflict_inputs},
        ensure_ascii=False,
    )
    for forbidden in ("expected_preference", "expected_state", "relevant_fact_ids", "forbidden_records"):
        assert forbidden not in serialized_inputs


def generate(root: Path, *, seed: str, force: bool = False) -> dict[str, Any]:
    if root.exists():
        if not force:
            raise FileExistsError(f"dataset already exists: {root}; pass --force to replace")
        shutil.rmtree(root)
    rng = random.Random(seed)
    preferences = generate_preferences(rng)
    retrieval = generate_retrieval(rng)
    conflicts = generate_conflicts(rng)
    validate(preferences, retrieval, conflicts)

    input_root = root / "hidden-test" / "inputs"
    label_root = root / "hidden-test" / "labels"
    files = {
        "preference_inputs": input_root / "preference.jsonl",
        "preference_labels": label_root / "preference.jsonl",
        "retrieval_corpus": input_root / "retrieval_corpus.jsonl",
        "retrieval_queries": input_root / "retrieval_queries.jsonl",
        "retrieval_qrels": label_root / "retrieval_qrels.jsonl",
        "conflict_inputs": input_root / "conflict.jsonl",
        "conflict_labels": label_root / "conflict.jsonl",
    }
    write_jsonl(files["preference_inputs"], preferences[0])
    write_jsonl(files["preference_labels"], preferences[1])
    write_jsonl(files["retrieval_corpus"], retrieval[0])
    write_jsonl(files["retrieval_queries"], retrieval[1])
    write_jsonl(files["retrieval_qrels"], retrieval[2])
    write_jsonl(files["conflict_inputs"], conflicts[0])
    write_jsonl(files["conflict_labels"], conflicts[1])
    manifest = {
        "dataset": VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "sealed": True,
        "seed_commitment": hashlib.sha256(seed.encode("utf-8")).hexdigest(),
        "protocol": {
            "preference_cases": 500,
            "preference_positive": 250,
            "retrieval_corpus_facts": 10000,
            "retrieval_queries": 500,
            "conflict_cases": 500,
            "latency_queries": 500,
            "top_k": 5,
            "labels_visible_to_agent": False,
            "production_changes_after_freeze": False,
        },
        "files": {
            name: {"path": path.relative_to(root).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}
            for name, path in files.items()
        },
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seed", required=True, help="Secret generation seed; only its SHA256 is stored.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = generate(args.root, seed=args.seed, force=args.force)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
