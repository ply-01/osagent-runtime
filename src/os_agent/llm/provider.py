"""OpenAI-compatible LLM provider used by the first-stage agent."""

from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_MODEL = "gpt-4o-mini"


class LLMError(RuntimeError):
    """Raised when the configured LLM provider cannot return a response."""


@dataclass(slots=True)
class ChatMessage:
    """A minimal chat message."""

    role: str
    content: str


@dataclass(slots=True)
class LLMConfig:
    """Runtime configuration for an OpenAI-compatible chat provider."""

    api_key: str | None = None
    base_url: str = _DEFAULT_BASE_URL
    model: str = _DEFAULT_MODEL
    timeout_seconds: int = 60
    temperature: float = 0.0
    context_max_chars: int = 12000
    replan_context_max_chars: int = 8000
    response_context_max_chars: int = 6000
    max_tokens: int | None = None
    thinking: str | None = None
    max_retries: int = 2
    retry_backoff_seconds: float = 0.25

    @classmethod
    def from_env(cls) -> LLMConfig:
        file_config = cls._from_local_file()
        return cls(
            api_key=(
                os.getenv("OPENAI_API_KEY")
                or os.getenv("OS_AGENT_API_KEY")
                or file_config.get("api_key")
            ),
            base_url=(
                os.getenv("OPENAI_BASE_URL")
                or os.getenv("OS_AGENT_BASE_URL")
                or str(file_config.get("base_url") or _DEFAULT_BASE_URL)
            ),
            model=(
                os.getenv("OS_AGENT_MODEL")
                or os.getenv("OPENAI_MODEL")
                or str(file_config.get("model") or _DEFAULT_MODEL)
            ),
            timeout_seconds=int(
                os.getenv("OS_AGENT_TIMEOUT_SECONDS")
                or file_config.get("timeout_seconds")
                or "60"
            ),
            temperature=float(
                os.getenv("OS_AGENT_TEMPERATURE")
                or file_config.get("temperature")
                or "0"
            ),
            context_max_chars=max(
                int(
                    os.getenv("OS_AGENT_CONTEXT_MAX_CHARS")
                    or file_config.get("context_max_chars")
                    or "12000"
                ),
                256,
            ),
            replan_context_max_chars=max(
                int(
                    os.getenv("OS_AGENT_REPLAN_CONTEXT_MAX_CHARS")
                    or file_config.get("replan_context_max_chars")
                    or "8000"
                ),
                256,
            ),
            response_context_max_chars=max(
                int(
                    os.getenv("OS_AGENT_RESPONSE_CONTEXT_MAX_CHARS")
                    or file_config.get("response_context_max_chars")
                    or "6000"
                ),
                256,
            ),
            max_retries=min(
                max(
                    int(
                        os.getenv("OS_AGENT_LLM_MAX_RETRIES")
                        or file_config.get("max_retries")
                        or "2"
                    ),
                    0,
                ),
                5,
            ),
            retry_backoff_seconds=max(
                float(
                    os.getenv("OS_AGENT_LLM_RETRY_BACKOFF_SECONDS")
                    or file_config.get("retry_backoff_seconds")
                    or "0.25"
                ),
                0.0,
            ),
        )

    @classmethod
    def _from_local_file(cls) -> dict[str, Any]:
        raw = os.getenv("OS_AGENT_LLM_CONFIG", "config/llm.local.json")
        path = Path(raw)
        # Resolve relative path: try cwd first, then project root (package parent)
        if not path.is_absolute() and not path.exists():
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            candidate = project_root / raw
            if candidate.exists():
                path = candidate
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LLMError(f"Invalid LLM config file {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise LLMError(f"Invalid LLM config file {path}: root must be an object")
        return data


class OpenAICompatibleProvider:
    """Small chat-completions client with no third-party dependency."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_env()
        self.last_usage: dict[str, int] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.config.api_key)

    def complete(self, messages: list[ChatMessage], response_format: str = "text") -> str:
        self.last_usage = {}
        if not self.config.api_key:
            raise LLMError("OPENAI_API_KEY or OS_AGENT_API_KEY is not set")
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "temperature": self.config.temperature,
        }
        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}
        if self.config.max_tokens is not None:
            payload["max_tokens"] = self.config.max_tokens
        if self.config.thinking is not None:
            payload["thinking"] = {"type": self.config.thinking}

        body = json.dumps(payload).encode("utf-8")
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        data: dict[str, Any] | None = None
        for attempt in range(self.config.max_retries + 1):
            request = urllib.request.Request(
                url,
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.config.timeout_seconds
                ) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                retryable = exc.code in {408, 425, 429} or 500 <= exc.code < 600
                if retryable and attempt < self.config.max_retries:
                    self._retry_delay(attempt)
                    continue
                detail = exc.read().decode("utf-8", errors="replace")
                raise LLMError(f"LLM HTTP error {exc.code}: {detail}") from exc
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                http.client.IncompleteRead,
            ) as exc:
                if attempt < self.config.max_retries:
                    self._retry_delay(attempt)
                    continue
                if isinstance(exc, TimeoutError):
                    raise LLMError("LLM request timed out") from exc
                raise LLMError(f"LLM connection error: {exc}") from exc

        if data is None:
            raise LLMError("LLM request did not return a response")

        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
            if choice.get("finish_reason") == "length" or not isinstance(content, str):
                raise LLMError("LLM returned truncated or non-text content")
            usage = data.get("usage", {})
            if isinstance(usage, dict):
                self.last_usage = {
                    key: value
                    for key, value in usage.items()
                    if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
                    and type(value) is int
                    and value >= 0
                }
            return content
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected LLM response: {data}") from exc

    def _retry_delay(self, attempt: int) -> None:
        delay = self.config.retry_backoff_seconds * (2**attempt)
        if delay > 0:
            time.sleep(delay)
