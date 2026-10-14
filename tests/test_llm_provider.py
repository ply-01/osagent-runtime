from __future__ import annotations

import http.client
import io
import json
import urllib.error
from unittest.mock import patch

import pytest

from os_agent.llm.provider import (
    ChatMessage,
    LLMConfig,
    LLMError,
    OpenAICompatibleProvider,
)


class _Response:
    def __init__(self, payload: dict) -> None:
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> bool:
        return False

    def read(self) -> bytes:
        return self.body


def _provider() -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        LLMConfig(
            api_key="test-key",
            max_retries=2,
            retry_backoff_seconds=0,
        )
    )


def test_provider_retries_an_incomplete_response() -> None:
    response = _Response(
        {
            "choices": [
                {"message": {"content": "complete"}, "finish_reason": "stop"}
            ],
            "usage": {"total_tokens": 7},
        }
    )
    with patch(
        "urllib.request.urlopen",
        side_effect=[http.client.IncompleteRead(b"", 1), response],
    ) as opener:
        result = _provider().complete([ChatMessage("user", "hello")])

    assert result == "complete"
    assert opener.call_count == 2


def test_provider_does_not_retry_a_nontransient_http_error() -> None:
    error = urllib.error.HTTPError(
        "https://api.openai.com/v1/chat/completions",
        400,
        "bad request",
        {},
        io.BytesIO(b'{"error":"bad request"}'),
    )
    with (
        patch("urllib.request.urlopen", side_effect=error) as opener,
        pytest.raises(LLMError, match="HTTP error 400"),
    ):
        _provider().complete([ChatMessage("user", "hello")])

    assert opener.call_count == 1
