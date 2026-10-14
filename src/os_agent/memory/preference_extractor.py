"""General preference extraction and attribute normalization.

Rules describe language acts and stable OS-agent preference domains. They do
not contain benchmark entities, identifiers, or expected values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import blake2b


def memory_statement_text(text: str) -> str:
    """Discard an outer presentation fence, never its enclosed statement."""
    clean = text.strip()
    fenced = re.fullmatch(r"(`{3,}|~{3,})[^\r\n]*\r?\n(.*?)\r?\n\1", clean, re.DOTALL)
    if fenced:
        clean = fenced.group(2)
    return " ".join(clean.split())


@dataclass(frozen=True, slots=True)
class PreferenceSignal:
    """One normalized preference candidate with provenance metadata."""

    attribute: str
    value: str
    source_type: str
    confidence: float
    tags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _AttributeRule:
    attribute: str
    pattern: re.Pattern[str]


class PreferenceExtractor:
    """Extract asserted preferences without treating every marker as truth."""

    _QUESTION = re.compile(
        r"[?？]|(?:吗|么|什么|哪个|哪种|是否|能否|为什么|怎么|如何|需要再讨论)[。！!]*$|"
        r"\b(?:does|do|is|are|which|what|why|how)\b.*\?$",
        re.IGNORECASE,
    )
    _QUOTED_OR_META = re.compile(
        r"(?:示例句|例句|请解释|这句话|他说|她说|引用|文档中写着|记录中写着|"
        r"quote|example sentence|the phrase|someone said|documentation says)",
        re.IGNORECASE,
    )
    _NO_PREFERENCE = re.compile(
        r"(?:没有|无|不设|无需)(?:任何|特别|固定)?(?:偏好|要求|倾向)|"
        r"(?:不要|别)(?:假设|推断|记录)(?:我的)?(?:偏好|习惯)|"
        r"\b(?:no|without)\s+(?:particular\s+|strong\s+)?preference\b|"
        r"\b(?:do not|don't)\s+(?:assume|infer|record)\s+(?:my\s+)?preference",
        re.IGNORECASE,
    )
    _EXPLICIT = re.compile(
        r"我(?:喜欢|偏好|习惯|一般|通常|总是|每次都?|希望|想要|需要|讨厌)|"
        r"(?:以后|今后|往后|接下来|从现在起|算了|还是|改成|换成|改回|改用|不用|别用)"
        r"[\s，,]*(?:用|使用|采用|改成|换成|保存|导出|回复|执行)?|"
        r"(?:默认|统一)(?:\S{0,16})(?:设为|设置为|使用|采用|转成|保存|导出)|"
        r"\bI\s+(?:prefer|like|always|usually|typically|want|need|hate|love)\b|"
        r"\b(?:from now on|going forward|please always|switch to|change to|move to)\b",
        re.IGNORECASE,
    )
    _CONFIGURATION = re.compile(
        r"\b[a-z][a-z0-9_.-]{1,48}\s*=\s*[^\s]+|"
        r"\buser\.[a-z0-9_.-]+\s+is\s+set\s+to\b|"
        r"(?:用户)?配置(?:项|文件)?(?:中|为|设置)",
        re.IGNORECASE,
    )
    _BEHAVIORAL = re.compile(
        r"(?:连续|最近|过去)(?:\S{0,12})(?:次|每天|每周|全部|总是)|"
        r"(?:两|三|四|五|六|七|八|九|十|\d+)(?:次|回)(?:\S{0,12})(?:都|总是)|"
        r"(?:工具|操作|行为)(?:记录|日志)(?:\S{0,12})(?:用户|操作者)|"
        r"(?:最近的|历史)(?:\S{0,12})(?:全部|均|总是)",
        re.IGNORECASE,
    )
    _BEHAVIORAL_EN = re.compile(
        r"\b(?:user|operator).*(?:consecutive|last\s+\d+|every time|repeatedly)\b",
        re.IGNORECASE,
    )
    _IMPLICIT = re.compile(
        r"我(?:在找|正在找|想找|有兴趣|感兴趣|很在意|重视|更倾向于|比较喜欢|希望能|需要)|"
        r"(?:对我来说|最好|理想情况下|如果可以|如果可能).{0,40}(?:更好|理想|重要|方便|合适)|"
        r"\bI(?:'m| am)\s+(?:looking for|interested in|planning|hoping|curious about|leaning toward)\b|"
        r"\bI(?:'d| would)\s+(?:like|love|rather)\b|"
        r"\bI\s+(?:enjoy|value|appreciate|favor|favour|care about)\b|"
        r"\b(?:important for me|works perfectly for me|would be (?:ideal|great|best)|"
        r"ideally|if possible|if available)\b",
        re.IGNORECASE,
    )
    _SINGLE_OR_ACCIDENTAL = re.compile(r"(?:偶然|仅|只)(?:\S{0,4})(?:一次|一回)")
    _SYSTEM_SUBJECT = re.compile(
        r"^(?:系统|应用(?:程序)?|软件|服务|the\s+(?:system|application|service))",
        re.IGNORECASE,
    )

    _ATTRIBUTE_RULES = (
        _AttributeRule(
            "communication.confirmation_policy",
            re.compile(r"(?:发送|邮件|合同|消息).*(?:确认|复核|预览)|(?:确认|复核|预览).*(?:发送|邮件)"),
        ),
        _AttributeRule(
            "document.output_format",
            re.compile(
                r"(?:导出|输出|报告|文档|附件|格式).*(?:PDF|DOCX|Markdown|HTML|文本)|"
                r"(?:PDF|DOCX|Markdown|HTML).*(?:导出|输出|报告|文档|附件|格式)|"
                r"\boutput[_ .-]?format\b|\b(?:PDF|DOCX|ODT|ODP|ODS|Markdown|HTML|TXT|CSV|PNG|JPEG)\b",
                re.IGNORECASE,
            ),
        ),
        _AttributeRule(
            "filesystem.save_location",
            re.compile(
                r"(?:保存|存放|放到|目录|路径|文件夹)|\b(?:save[_ .-]?(?:dir|path)|root[_ .-]?path)\b",
                re.IGNORECASE,
            ),
        ),
        _AttributeRule(
            "application.preferred_tool",
            re.compile(
                r"(?:软件|应用|工具|打开).*(?:使用|采用|首选)|(?:使用|采用|改用).*(?:软件|应用|工具)|"
                r"\b(?:WPS|LibreOffice|PowerShell)\b",
                re.IGNORECASE,
            ),
        ),
        _AttributeRule(
            "ui.theme",
            re.compile(
                r"(?:主题|界面|外观).*(?:深色|浅色|模式)|(?:深色|浅色|模式).*(?:主题|界面|外观)|"
                r"\b(?:theme|dark mode|light mode)\b",
                re.IGNORECASE,
            ),
        ),
        _AttributeRule(
            "automation.backup_schedule",
            re.compile(
                r"(?:备份|归档).*(?:每天|每周|周[一二三四五六日天]|\d{1,2}[:：]\d{2}|时间)|"
                r"(?:每天|每周|周[一二三四五六日天]).*(?:备份|归档)"
            ),
        ),
        _AttributeRule(
            "response.style",
            re.compile(
                r"(?:回答|回复|响应).*(?:简洁|详细|示例|格式)|(?:简洁|详细).*(?:回答|回复|响应)|"
                r"\breply[_ .-]?style\b",
                re.IGNORECASE,
            ),
        ),
        _AttributeRule(
            "document.naming_rule",
            re.compile(r"(?:文件名|命名).*(?:日期|前缀|后缀|规则)|\bnaming[_ .-]?rule\b", re.IGNORECASE),
        ),
        _AttributeRule(
            "document.page_size",
            re.compile(
                r"(?:页面|纸张|版面).*(?:A[0-9]|尺寸)|(?:A[0-9]).*(?:页面|纸张|版面)|\bpage[_ .-]?size\b",
                re.IGNORECASE,
            ),
        ),
        _AttributeRule(
            "scanner.resolution",
            re.compile(r"(?:扫描|分辨率|DPI)|\bresolution\b", re.IGNORECASE),
        ),
        _AttributeRule(
            "operation.execution_mode",
            re.compile(r"(?:只读|静默|安全|执行).*(?:模式|安装|操作)?|\b(?:read[_ -]?only|silent mode)\b", re.IGNORECASE),
        ),
        _AttributeRule(
            "language.preferred_language",
            re.compile(
                r"(?:回复|回答|响应|语言).*(?:中文|英文|双语)|"
                r"(?:中文|英文|双语).*(?:回复|回答|响应|语言)|"
                r"\b(?:reply|response|language).*(?:Chinese|English|bilingual)\b|"
                r"\b(?:Chinese|English|bilingual).*(?:reply|response|language)\b",
                re.IGNORECASE,
            ),
        ),
        _AttributeRule(
            "notification.delivery_mode",
            re.compile(r"(?:通知|提醒).*(?:静音|声音|弹窗|横幅|关闭|开启)|\bnotification\b", re.IGNORECASE),
        ),
        _AttributeRule(
            "ui.window_animation",
            re.compile(r"(?:窗口|界面).*(?:动画|动效)|(?:动画|动效).*(?:窗口|界面)"),
        ),
        _AttributeRule(
            "input.preferred_method",
            re.compile(r"(?:输入法|键盘布局|快捷键)|\b(?:input method|keyboard layout|shortcut)\b", re.IGNORECASE),
        ),
        _AttributeRule(
            "audio.volume_policy",
            re.compile(r"(?:音量|声音).*(?:大小|静音|级别|降低|提高)|\bvolume\b", re.IGNORECASE),
        ),
    )

    @staticmethod
    def _assertion_before_question(text: str) -> str:
        """Keep completed assertions before a separate follow-up question."""
        sentences = re.split(r"(?<=[.!。！])\s+|(?<=[。！])", text)
        prefix = []
        for sentence in sentences:
            if "?" in sentence or "？" in sentence:
                return " ".join(prefix) if prefix else text
            prefix.append(sentence)
        return text

    def extract(self, text: str, *, source_type: str = "user_message") -> PreferenceSignal | None:
        clean = self._assertion_before_question(memory_statement_text(text))
        if not clean:
            return None

        explicit = bool(self._EXPLICIT.search(clean))
        configuration = bool(self._CONFIGURATION.search(clean))
        behavioral = bool(
            self._BEHAVIORAL.search(clean) or self._BEHAVIORAL_EN.search(clean)
        ) and not self._SINGLE_OR_ACCIDENTAL.search(clean)
        if not (explicit or configuration or behavioral):
            return None
        if self._is_non_assertion(clean, has_preference_signal=True):
            return None

        evidence_type = (
            "configuration" if configuration else "behavior" if behavioral and not explicit else source_type
        )
        confidence = 0.72 if explicit else 0.68 if configuration else 0.62
        attribute = self.normalize_attribute(clean)
        return PreferenceSignal(
            attribute=attribute,
            value=clean[:200],
            source_type=evidence_type,
            confidence=confidence,
            tags=(
                "user_message",
                "preference",
                f"attribute:{attribute}",
                f"source:{evidence_type}",
            ),
        )

    def extract_observation(self, text: str) -> PreferenceSignal | None:
        """Extract one implicit, task-scoped user preference observation."""
        clean = self._assertion_before_question(memory_statement_text(text))
        if not clean or not self._IMPLICIT.search(clean):
            return None
        if self._is_non_assertion(clean, has_preference_signal=True):
            return None
        if self.extract(clean) is not None:
            return None
        attribute = self.normalize_attribute(clean)
        return PreferenceSignal(
            attribute=attribute,
            value=clean[:300],
            source_type="session_observation",
            confidence=0.58,
            tags=(
                "user_message",
                "preference",
                "preference_observation",
                f"attribute:{attribute}",
                "source:session_observation",
            ),
        )

    def normalize_attribute(self, text: str) -> str:
        matched = self.match_attribute(text)
        if matched is not None:
            return matched
        digest = blake2b(text.lower().encode("utf-8"), digest_size=6).hexdigest()
        return f"preference.unclassified.{digest}"

    def match_attribute(self, text: str) -> str | None:
        """Return a known canonical attribute without inventing a hash key."""
        for rule in self._ATTRIBUTE_RULES:
            if rule.pattern.search(text):
                return rule.attribute
        return None

    def _is_non_assertion(self, text: str, *, has_preference_signal: bool = False) -> bool:
        system_statement = bool(self._SYSTEM_SUBJECT.search(text)) and not bool(
            re.search(r"(?:请|please|set|change|switch|配置|设置)", text, re.IGNORECASE)
        )
        return bool(
            self._QUESTION.search(text)
            or self._QUOTED_OR_META.search(text)
            or self._NO_PREFERENCE.search(text)
            or system_statement
        )
