"""Conservative structured interpretation for memory statements and queries.

The parser composes generic language-act signals with caller metadata. It does
not contain benchmark entities, expected values, case identifiers, or a second
query-only attribute rule table.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal, cast

from os_agent.memory.preference_extractor import PreferenceExtractor, memory_statement_text

MemoryIntent = Literal[
    "set",
    "avoid",
    "update",
    "revert",
    "temporary",
    "query",
    "reported",
    "uncertain",
    "ignore",
]
Certainty = Literal["certain", "uncertain", "reported"]
TemporalStatus = Literal["stable", "temporary", "reverted"]


@dataclass(frozen=True, slots=True)
class MemoryFrame:
    intent: MemoryIntent
    entity: str
    scope: str
    attribute: str
    value: str
    certainty: Certainty
    temporal_status: TemporalStatus
    polarity: str = "prefer"
    should_persist: bool = True
    confidence: float = 0.7

    @property
    def storage_entity(self) -> str:
        return f"{self.entity}::{self.scope}" if self.scope else self.entity

    @property
    def tags(self) -> list[str]:
        tags = [
            "user_message",
            "preference",
            f"intent:{self.intent}",
            f"certainty:{self.certainty}",
            f"temporal:{self.temporal_status}",
            f"polarity:{self.polarity}",
            f"attribute:{self.attribute}",
        ]
        if self.scope:
            tags.append(f"scope:{self.scope}")
        return tags


@dataclass(frozen=True, slots=True)
class QueryConstraints:
    entity: str | None = None
    entities: tuple[str, ...] = ()
    scope: str = ""
    attribute: str | None = None
    days: int | None = None


class MemoryFrameParser:
    """Map generic language acts and explicit metadata to a memory frame."""

    _QUESTION = re.compile(r"[?？]")
    _PROJECT_STATEMENT = re.compile(
        r"(?:^|[：:，,])\s*(?P<scope>[\w\u4e00-\u9fff-]{1,40}(?:项目|工作区|任务))的"
        r"(?P<attribute>[\w\u4e00-\u9fff-]{2,40}?)"
        r"(?:现在|目前)?(?:改为|设置为|设为|是|为)\s*(?P<value>[^，,。！？?;；]{1,120})"
    )
    _ONE_SHOT = re.compile(
        r"(?:帮我|替我|请).{0,20}(?:创建|新建|删除|读取|打开|下载|发送)|"
        r"\bplease\s+(?:create|delete|read|open|download|send)\b", re.IGNORECASE
    )
    _DURABLE = re.compile(r"(?:默认|每次|总是|以后|今后|偏好|习惯)|\b(?:default|always|prefer|every time)\b", re.IGNORECASE)
    _REPORTED = re.compile(
        r"\b(?:according to|reported|quoted|said|says)\b|"
        r"(?:据|转述|引用).{0,20}(?:说|表示|记载)|(?:他说|她说|对方说)",
        re.IGNORECASE,
    )
    _UNCERTAIN = re.compile(
        r"(?:也许|或许|可能|大概|尚未|还没).{0,20}(?:决定|确定|确认)?|"
        r"\b(?:maybe|perhaps|possibly|might|undecided|unconfirmed|tbd)\b",
        re.IGNORECASE,
    )
    _TEMPORARY = re.compile(
        r"(?:暂时|临时|本次|这次|仅此一次)|"
        r"\b(?:temporarily|for now|this (?:time|task|session) only|one[- ]off)\b",
        re.IGNORECASE,
    )
    _REVERT = re.compile(
        r"(?:恢复|改回|换回|回到)|\b(?:revert|restore|switch back|go back)\b",
        re.IGNORECASE,
    )
    _UPDATE = re.compile(
        r"(?:改成|换成|改用|不再|现在改为)|"
        r"\b(?:switch|change|replace|move)\b.{0,80}\b(?:to|with)\b|"
        r"\bno longer\b",
        re.IGNORECASE,
    )
    _AVOID = re.compile(
        r"(?:不喜欢|讨厌|避开|避免)|"
        r"\b(?:avoid|dislike|hate|do not want|don't want)\b",
        re.IGNORECASE,
    )
    _SCOPE_PATTERNS = (
        re.compile(
            r"(?:在|对于|针对)(?P<scope>[A-Za-z0-9_.\-\u4e00-\u9fff]{2,40}?)"
            r"(?:中|里|内|场景)(?:[，,。.]|$)"
        ),
        re.compile(
            r"\b(?:in|for|within)\s+"
            r"(?P<scope>(?:project|workspace|task)\s+[a-z0-9_.-]{1,40})\b",
            re.IGNORECASE,
        ),
    )
    _DAYS = re.compile(
        r"(?:最近|过去)\s*(?P<cn>\d{1,4})\s*天|"
        r"\b(?:last|past)\s+(?P<en>\d{1,4})\s+days?\b",
        re.IGNORECASE,
    )
    _ASSIGNMENT_SLOT = re.compile(
        r"^\s*(?P<slot>[^=\n]{1,80})\s*=\s*[^=\n]{1,200}\s*$"
    )
    _CN_MUTATION_SLOT = re.compile(
        r"(?:把|将)?\s*(?P<slot>[A-Za-z0-9_.\-\u4e00-\u9fff ]{2,80}?)\s*"
        r"(?:改成|换成|改回|改用|切换为|改为|设置为|设为|换为)"
    )
    _EN_MUTATION_SLOT = re.compile(
        r"\b(?:set|change|switch|replace|move|restore)\s+(?:the\s+)?"
        r"(?P<slot>[a-z][a-z0-9_.\- ]{1,80}?)\s+(?:to|with)\b|"
        r"\b(?:the\s+)?(?P<passive>[a-z][a-z0-9_.\- ]{1,80}?)\s+"
        r"(?:is|was)\s+(?:now\s+)?set\s+to\b",
        re.IGNORECASE,
    )
    _SLOT_PREFIX = re.compile(
        r"^(?:(?:我|我们|用户)?(?:现在|以后|今后|接下来|希望|想要|想|需要|决定|请)+|"
        r"(?:please|i|we|now|from now on)\s+)+",
        re.IGNORECASE,
    )
    _ATTRIBUTE_ACTION = re.compile(
        r"(?:喜欢|偏好|希望|想要|需要|改成|换成|改用|设置|切换|恢复|为什么|怎么|如何)|"
        r"\b(?:prefer|want|need|set|change|switch|replace|restore|why|how)\b",
        re.IGNORECASE,
    )
    _INTENTS = frozenset(
        {"set", "avoid", "update", "revert", "temporary", "query", "reported", "uncertain", "ignore"}
    )
    _CERTAINTIES = frozenset({"certain", "uncertain", "reported"})
    _TEMPORAL_STATUSES = frozenset({"stable", "temporary", "reverted"})

    def __init__(self, preference_extractor: PreferenceExtractor | None = None) -> None:
        self.preference_extractor = preference_extractor or PreferenceExtractor()

    def parse(
        self,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryFrame | None:
        clean = memory_statement_text(text)
        if not clean:
            return None
        if self._QUESTION.search(clean):
            return None
        if self._ONE_SHOT.search(clean) and not self._DURABLE.search(clean):
            return None
        metadata = metadata or {}
        entity = self._entity(metadata)
        scope = self.extract_scope(clean, metadata=metadata)
        project_statement = self._PROJECT_STATEMENT.search(clean)
        if project_statement and not scope:
            scope = self.normalize_scope(project_statement.group("scope"))
        signal = self.preference_extractor.extract(clean)
        observation = self.preference_extractor.extract_observation(clean)
        raw_value = metadata.get("value")
        value = self._infer_value(clean, raw_value)[:300]
        if project_statement and raw_value is None:
            value = project_statement.group("value").strip()
        attribute_input = str(metadata.get("attribute", "")).strip()
        attribute = (
            self.canonical_attribute(attribute_input)
            if attribute_input
            else self.canonical_attribute(project_statement.group("attribute"))
            if project_statement
            else self.preference_extractor.match_attribute(clean)
        )
        if not attribute:
            attribute = self.infer_open_attribute(clean)

        explicit_intent = self._metadata_choice(
            metadata.get("intent"), self._INTENTS
        )
        explicit_frame = bool(
            attribute_input
            and any(
                key in metadata
                for key in ("intent", "certainty", "temporal_status", "value")
            )
        )
        certainty = self._certainty(clean, metadata)
        temporal_status = self._temporal_status(clean, metadata)
        intent = cast(
            MemoryIntent,
            explicit_intent or self._infer_intent(clean, certainty, temporal_status),
        )

        if signal is None:
            if observation is not None:
                return None
            if not attribute or (
                intent not in {"avoid", "update", "revert"} and not explicit_frame and not project_statement
            ):
                return None

        attribute = attribute or (
            signal.attribute
            if signal is not None
            else self.preference_extractor.normalize_attribute(clean)
        )
        persistent = (
            certainty == "certain"
            and temporal_status != "temporary"
            and intent not in {"query", "reported", "uncertain", "ignore", "temporary"}
        )
        confidence = signal.confidence if signal is not None else 0.7
        if not persistent:
            confidence = min(confidence, 0.2 if certainty != "certain" else 0.6)
        return MemoryFrame(
            intent=intent,
            entity=entity,
            scope=scope,
            attribute=attribute,
            value=value,
            certainty=certainty,
            temporal_status=temporal_status,
            polarity="avoid" if intent == "avoid" else "prefer",
            should_persist=persistent,
            confidence=confidence,
        )

    def query_constraints(
        self,
        text: str,
        *,
        known_entities: list[str] | None = None,
        known_attributes: list[str] | None = None,
    ) -> QueryConstraints:
        clean = " ".join(text.strip().split())
        scope = self.extract_scope(clean)
        entities = self._match_known_entities(clean, known_entities or [], scope)
        entity = entities[0] if len(entities) == 1 else None
        attribute = self.preference_extractor.match_attribute(clean)
        if attribute is None:
            attribute = self._match_known_attribute(clean, known_attributes or [])
        if attribute and known_attributes and attribute not in known_attributes:
            attribute = None
        days_match = self._DAYS.search(clean)
        days = int(days_match.group("cn") or days_match.group("en")) if days_match else None
        if entity and "::" in entity:
            scope = entity.split("::", 1)[1]
        return QueryConstraints(
            entity=entity,
            entities=entities,
            scope=scope,
            attribute=attribute,
            days=days,
        )

    def canonical_attribute(self, attribute: str) -> str | None:
        """Align natural language through the product ontology only."""
        clean = " ".join(attribute.strip().split())
        if not clean:
            return None
        if re.fullmatch(r"[a-z][a-z0-9_.-]*", clean):
            return clean
        matched = self.preference_extractor.match_attribute(clean)
        if matched is not None:
            return matched
        if len(clean) <= 64 and not self._ATTRIBUTE_ACTION.search(clean):
            return self._slot_attribute(clean)
        return None

    @classmethod
    def infer_open_attribute(cls, text: str) -> str | None:
        """Infer a stable open-domain slot from assignment/update grammar."""
        clean = " ".join(text.strip().split())
        match = cls._ASSIGNMENT_SLOT.match(clean)
        slot = match.group("slot") if match else ""
        if not slot:
            match = cls._CN_MUTATION_SLOT.search(clean)
            slot = match.group("slot") if match else ""
        if not slot:
            match = cls._EN_MUTATION_SLOT.search(clean)
            slot = (match.group("slot") or match.group("passive")) if match else ""
        slot = cls._SLOT_PREFIX.sub("", slot).strip(" ，,。.;；:：")
        slot = re.sub(r"^(?:把|将)\s*", "", slot)
        if len(slot) < 2:
            return None
        return cls._slot_attribute(slot)

    @staticmethod
    def _slot_attribute(slot: str) -> str | None:
        normalized = unicodedata.normalize("NFKC", slot).casefold()
        normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "_", normalized)
        normalized = normalized.strip("_")
        if len(normalized) < 2:
            return None
        return f"preference.open.{normalized[:64]}"

    def extract_scope(
        self,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        metadata = metadata or {}
        for key in ("scope", "project_id", "workspace_id", "task_scope"):
            value = str(metadata.get(key, "")).strip()
            if value:
                return self.normalize_scope(value)
        for pattern in self._SCOPE_PATTERNS:
            match = pattern.search(text)
            if match:
                return self.normalize_scope(match.group("scope"))
        return ""

    @staticmethod
    def normalize_scope(value: str) -> str:
        return re.sub(r"\s+", " ", value.strip(" ，,。.;；"))[:80]

    @classmethod
    def _certainty(cls, text: str, metadata: dict[str, Any]) -> Certainty:
        explicit = cls._metadata_choice(metadata.get("certainty"), cls._CERTAINTIES)
        if explicit:
            return cast(Certainty, explicit)
        if cls._REPORTED.search(text):
            return "reported"
        if cls._UNCERTAIN.search(text):
            return "uncertain"
        return "certain"

    @classmethod
    def _temporal_status(
        cls, text: str, metadata: dict[str, Any]
    ) -> TemporalStatus:
        explicit = cls._metadata_choice(
            metadata.get("temporal_status"), cls._TEMPORAL_STATUSES
        )
        if explicit:
            return cast(TemporalStatus, explicit)
        if cls._TEMPORARY.search(text):
            return "temporary"
        if cls._REVERT.search(text):
            return "reverted"
        return "stable"

    @classmethod
    def _infer_intent(
        cls,
        text: str,
        certainty: Certainty,
        temporal_status: TemporalStatus,
    ) -> MemoryIntent:
        if certainty == "reported":
            return "reported"
        if certainty == "uncertain":
            return "uncertain"
        if temporal_status == "temporary":
            return "temporary"
        if temporal_status == "reverted":
            return "revert"
        if cls._QUESTION.search(text):
            return "query"
        if cls._AVOID.search(text):
            return "avoid"
        if cls._UPDATE.search(text):
            return "update"
        return "set"

    @staticmethod
    def _metadata_choice(value: object, allowed: frozenset[str]) -> str:
        clean = str(value or "").strip().lower()
        return clean if clean in allowed else ""

    @classmethod
    def _infer_value(cls, text: str, explicit: object) -> str:
        if explicit is not None and str(explicit).strip():
            return str(explicit).strip()
        clean = " ".join(text.strip().split())
        assignment = re.match(
            r"^\s*(?P<left>[^=\n]{1,80})\s*=\s*(?P<right>[^=\n]{1,200})\s*$",
            clean,
        )
        if assignment:
            left = assignment.group("left").strip()
            right = assignment.group("right").strip()
            if left and right:
                candidate = cls._value_candidate(right)
                if candidate:
                    return candidate
        set_to = re.search(
            r"(?:\b(?:is|was)\s+set\s+to\b|\b(?:set|switch|change|move)\s+to\b|"
            r"(?:设置|改成|换成|改回|改用|切换为|改为|设置为|换为))(?:\s+)?(?P<value>[^。.!?;，,]{1,120})",
            clean,
            re.IGNORECASE,
        )
        if set_to:
            candidate = cls._value_candidate(set_to.group("value"))
            if candidate:
                return candidate
        object_update = re.search(
            r"\b(?:set|switch|change|move|restore)\s+(?:the\s+)?"
            r"[^.!?;,]{1,80}?\s+to\s+(?P<value>[^.!?;,]{1,120})",
            clean,
            re.IGNORECASE,
        )
        if object_update:
            candidate = cls._value_candidate(object_update.group("value"))
            if candidate:
                return candidate
        if re.search(
            r"(?:希望|想|要|请|设为|设成|设置|改成|改为|换成|切换为|使用|采用|变成|变为)",
            clean,
        ):
            trailing_value = re.search(
                r"(?:为|成|到|用|作)\s*(?P<value>[^。.!?;，,]{1,120})",
                clean,
            )
            if trailing_value:
                candidate = cls._value_candidate(trailing_value.group("value"))
                if candidate:
                    return candidate
        return clean

    @staticmethod
    def _value_candidate(value: object) -> str:
        clean = str(value or "").strip()
        if not clean:
            return ""
        action_prefixes = (
            "导出",
            "输出",
            "保存",
            "设置",
            "改成",
            "换成",
            "切换",
            "改用",
            "使用",
            "采用",
            "变成",
            "改为",
            "设置为",
            "切换为",
            "换为",
        )
        if any(clean.startswith(prefix) for prefix in action_prefixes):
            return ""
        return clean

    @staticmethod
    def _entity(metadata: dict[str, Any]) -> str:
        for key in ("user_id", "entity", "actor_id"):
            value = str(metadata.get(key, "")).strip()
            if value:
                return value
        return "用户"

    @classmethod
    def _match_known_entity(
        cls,
        query: str,
        entities: list[str],
        scope: str,
    ) -> str | None:
        matches = cls._match_known_entities(query, entities, scope)
        return matches[0] if matches else None

    @staticmethod
    def _match_known_entities(
        query: str,
        entities: list[str],
        scope: str,
    ) -> tuple[str, ...]:
        """Return every explicitly mentioned entity in textual order."""
        lowered = query.casefold()
        matches: list[tuple[int, int, str]] = []
        for entity in entities:
            aliases = {
                entity,
                entity.split("::", 1)[-1],
                entity.replace("_", " "),
                entity.replace("_", "-"),
            }
            positions = [
                lowered.find(alias.casefold())
                for alias in aliases
                if alias and alias.casefold() in lowered
            ]
            if positions:
                scoped_priority = 0 if scope and entity.endswith(f"::{scope}") else 1
                matches.append((min(positions), scoped_priority, entity))
        if scope:
            scoped = [item for item in matches if item[2].endswith(f"::{scope}")]
            if scoped:
                matches = scoped
        matches.sort(key=lambda item: (item[0], item[1], -len(item[2]), item[2]))
        return tuple(dict.fromkeys(item[2] for item in matches))

    @staticmethod
    def _match_known_attribute(query: str, attributes: list[str]) -> str | None:
        lowered = query.lower()
        matches: list[str] = []
        for attribute in attributes:
            aliases = {
                attribute.lower(),
                attribute.lower().replace("_", " "),
                attribute.lower().replace("_", "-"),
                attribute.lower().replace(".", " "),
                attribute.lower().rsplit(".", 1)[-1].replace("_", " "),
            }
            if any(alias and alias in lowered for alias in aliases):
                matches.append(attribute)
        return max(matches, key=len) if matches else None
