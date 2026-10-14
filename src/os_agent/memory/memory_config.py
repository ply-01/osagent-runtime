"""Tunable memory runtime configuration."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar


class MemoryConfig:
    """Load memory tuning parameters from JSON with code defaults."""

    _instance: MemoryConfig | None = None

    DEFAULT_SCHEMA: ClassVar[dict[str, Any]] = {
        "core": {
            "default_confidence": {"value": 0.55, "desc": "Default confidence for new Core facts."},
            "default_importance": {"value": 0.5, "desc": "Default importance for new Core facts."},
            "confidence_boost": {"value": 0.05, "desc": "Confidence increment for repeated facts."},
            "search_top_k": {"value": 3, "desc": "Default Core search result limit."},
            "search_relative_cutoff": {
                "value": 0.65,
                "desc": "Drop Core hits scoring below this fraction of the top hit. 0 disables.",
            },
            "semantic_min_similarity": {
                "value": 0.60,
                "desc": "Absolute cosine floor for semantic Core hits.",
            },
            "semantic_relative_cutoff": {
                "value": 0.99,
                "desc": (
                    "Drop semantic hits below this fraction of the best similarity. "
                    "Sentence embeddings compress same-domain facts into a narrow "
                    "band, so the useful signal is the gap to the best hit rather "
                    "than the absolute value. 0 disables."
                ),
            },
            "recency_window_days": {"value": 21.0, "desc": "Core recency scoring window."},
            "recency_floor": {"value": 0.2, "desc": "Minimum Core recency factor."},
            "composite_keyword_weight": {"value": 0.55, "desc": "Keyword score weight."},
            "composite_confidence_weight": {"value": 0.15, "desc": "Confidence score weight."},
            "composite_importance_weight": {"value": 0.2, "desc": "Importance score weight."},
            "composite_recency_weight": {"value": 0.1, "desc": "Recency score weight."},
        },
        "archival": {
            "search_facts_top_k": {"value": 6, "desc": "Archived fact search result limit."},
            "search_raw_events_top_k": {"value": 8, "desc": "Raw event fallback result limit."},
            "recency_window_days": {"value": 60.0, "desc": "Archival recency scoring window."},
            "recency_floor": {"value": 0.05, "desc": "Minimum Archival recency factor."},
            "composite_keyword_weight": {"value": 0.75, "desc": "Keyword score weight."},
            "composite_recency_weight": {"value": 0.25, "desc": "Recency score weight."},
            "raw_event_hot_max_events": {
                "value": 10000,
                "desc": "Maximum raw events kept in the hot JSONL window before gzip archival.",
            },
            "raw_event_archive_enabled": {
                "value": True,
                "desc": "Move evicted raw events to a compressed cold archive.",
            },
            "global_summary_max_chars": {
                "value": 6000,
                "desc": "Maximum extractive summary size across sessions.",
            },
            "global_summary_max_sessions": {
                "value": 64,
                "desc": "Maximum sessions represented in the global summary.",
            },
        },
        "recall": {
            "min_core_results": {"value": 2, "desc": "Core hits needed before skipping Archival."},
            "max_total": {"value": 8, "desc": "Maximum facts returned by recall."},
            "raw_event_default_confidence": {"value": 0.25, "desc": "Raw fallback confidence."},
            "raw_event_default_importance": {"value": 0.15, "desc": "Raw fallback importance."},
            "include_legacy_markdown_context": {
                "value": False,
                "desc": "Include legacy Markdown retrieval alongside Mem0 recall.",
            },
        },
        "pipeline": {
            "extractor": {
                "value": "rules",
                "desc": "User-memory extractor: rules, llm, or semantic_state (explicit opt-in).",
            },
            "extractor_max_tokens": {
                "value": 2048,
                "desc": "Grounded semantic mention output token limit.",
            },
            "state_resolver_context_limit": {
                "value": 12,
                "desc": "Maximum owner/scope-matched property slots sent to relation classification.",
            },
            "semantic_fallback_to_rules": {
                "value": True,
                "desc": "Use conservative rules when semantic mention extraction is unavailable or rejects output.",
            },
            "session_dedup": {"value": True, "desc": "Deduplicate facts within one session."},
            "confidence_threshold_legacy_write": {
                "value": 0.5,
                "desc": "Minimum confidence for legacy Markdown backfill.",
            },
            "backfill_legacy_markdown": {
                "value": True,
                "desc": "Maintain Markdown as a readable mirror of active Core facts.",
            },
            "skip_memory_file_facts": {"value": True, "desc": "Skip facts from memory metadata files."},
            "per_file_attributes": {"value": True, "desc": "Use per-file content attributes."},
            "confidence_boost": {"value": 0.05, "desc": "Pipeline unchanged-fact confidence boost."},
            "archive_raw_events": {
                "value": True,
                "desc": "Persist complete session events for deep recall fallback.",
            },
            "behavior_enabled": {"value": True, "desc": "Aggregate repeated tool parameters."},
            "behavior_min_observations": {
                "value": 3,
                "desc": "Repeated observations required before inferring a behavior preference.",
            },
            "session_preference_consolidation_enabled": {
                "value": True,
                "desc": "Keep one low-confidence implicit preference observation per session.",
            },
            "session_preference_confidence": {
                "value": 0.58,
                "desc": "Confidence assigned to implicit session preference observations.",
            },
            "session_preference_importance": {
                "value": 0.55,
                "desc": "Importance assigned to implicit session preference observations.",
            },
            "preference_reinforcement_enabled": {
                "value": True,
                "desc": "Promote matching observations from independent sessions.",
            },
            "preference_reinforcement_min_sessions": {
                "value": 2,
                "desc": "Independent sessions required before preference promotion.",
            },
            "preference_reinforcement_similarity": {
                "value": 0.6,
                "desc": "Minimum lexical overlap for observation reinforcement.",
            },
            "preference_reinforcement_confidence": {
                "value": 0.72,
                "desc": "Initial confidence for a reinforced preference.",
            },
            "preference_reinforcement_importance": {
                "value": 0.7,
                "desc": "Importance for cross-session reinforced preferences.",
            },
            "semantic_alignment_threshold": {
                "value": 0.68,
                "desc": (
                    "Minimum explainable entity/scope/attribute alignment score "
                    "before natural-language facts share a version chain."
                ),
            },
            "rule_importance": {
                "read_file_success": {"value": 0.6, "desc": "Read-file importance."},
                "list_dir_success": {"value": 0.35, "desc": "List-dir importance."},
                "shell_success": {"value": 0.3, "desc": "Shell success importance."},
                "tool_failure": {"value": 0.55, "desc": "Tool failure importance."},
                "user_preference": {"value": 0.75, "desc": "Preference importance."},
                "user_safety": {"value": 0.9, "desc": "Safety importance."},
                "read_pdf_success": {"value": 0.65, "desc": "PDF importance."},
            },
            "rule_confidence": {
                "read_file_success": {"value": 0.6, "desc": "Read-file confidence."},
                "list_dir_success": {"value": 0.45, "desc": "List-dir confidence."},
                "shell_success": {"value": 0.5, "desc": "Shell success confidence."},
                "tool_failure": {"value": 0.65, "desc": "Tool failure confidence."},
                "user_preference": {"value": 0.65, "desc": "Preference confidence."},
                "user_safety": {"value": 0.7, "desc": "Safety confidence."},
                "read_pdf_success": {"value": 0.65, "desc": "PDF confidence."},
            },
        },
        "query_rewrite": {
            "enabled": {"value": True, "desc": "Use conservative query planning before Mem0 recall."},
        },
        "agentic_policy": {
            "distillation_candidate_limit": {
                "value": 24,
                "desc": "Recall candidates collected before policy distillation.",
            },
            "trace_enabled": {
                "value": False,
                "desc": "Append agentic memory action traces for later training.",
            },
            "trace_file": {
                "value": "policy-traces.jsonl",
                "desc": "Trace file path relative to the memory root.",
            },
        },
    }

    def __init__(self, config_path: str | Path | None = None) -> None:
        raw_path = config_path or os.getenv("OS_AGENT_MEMORY_CONFIG", "config/memory_config.json")
        self._configured_path = Path(raw_path)
        self.path = self._resolve_path(self._configured_path)
        self._raw: dict[str, Any] = {}
        self._flat: dict[str, Any] = {}
        self._descriptions: dict[str, str] = {}
        self._paths: dict[str, list[str]] = {}
        self.reload()

    @classmethod
    def global_config(cls) -> MemoryConfig:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_global(cls) -> None:
        cls._instance = None

    @staticmethod
    def _resolve_path(path: Path) -> Path:
        if path.is_absolute():
            return path
        if path.exists():
            return path.resolve()
        project_root = Path(__file__).resolve().parents[3]
        candidate = project_root / path
        if candidate.exists():
            return candidate.resolve()
        return (Path.cwd() / path).resolve()

    def reload(self) -> None:
        self._raw = deepcopy(self.DEFAULT_SCHEMA)
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self._raw = self._merge_schema(self._raw, loaded)
            except (OSError, json.JSONDecodeError):
                pass
        self._flat = {}
        self._descriptions = {}
        self._paths = {}
        self._flatten(self._raw)

    def all(self) -> dict[str, Any]:
        return dict(self._flat)

    def describe(self, key: str) -> str:
        return self._descriptions.get(key, "")

    def get(self, key: str, default: Any = None) -> Any:
        return self._flat.get(key, default)

    def set_value(self, key: str, value: Any) -> None:
        if key not in self._paths:
            raise KeyError(f"unknown memory config key: {key}")
        node: Any = self._raw
        for part in self._paths[key][:-1]:
            node = node[part]
        node[self._paths[key][-1]]["value"] = value
        self._flat[key] = value

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._raw, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return self.path

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self.get(name, 0)

    def _flatten(self, node: dict[str, Any], prefix: str = "", path: list[str] | None = None) -> None:
        current_path = path or []
        for key, value in node.items():
            if key.startswith("_"):
                continue
            flat_key = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict) and "value" in value:
                self._flat[flat_key] = value["value"]
                self._descriptions[flat_key] = str(value.get("desc", ""))
                self._paths[flat_key] = current_path + [key]
            elif isinstance(value, dict):
                self._flatten(value, flat_key, current_path + [key])

    @classmethod
    def _merge_schema(cls, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        merged = deepcopy(base)
        for key, value in override.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
                and "value" not in merged[key]
            ):
                merged[key] = cls._merge_schema(merged[key], value)
            else:
                merged[key] = value
        return merged


def parse_tuning_value(raw: str) -> Any:
    """Parse a CLI tuning value into bool/int/float/string."""
    clean = raw.strip()
    lowered = clean.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    try:
        return int(clean)
    except ValueError:
        pass
    try:
        return float(clean)
    except ValueError:
        return clean
