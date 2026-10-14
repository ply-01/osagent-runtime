import importlib
import json
from pathlib import Path


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    return importlib.import_module("annotate_isolated_conflicts")


def test_calls_are_blind_and_independent(monkeypatch):
    target = module(monkeypatch)
    calls = []
    class Provider:
        def __init__(self, config):
            pass
        def complete(self, messages, response_format):
            calls.append(json.loads(messages[1].content))
            return json.dumps({"facts": [{"entity": "用户", "scope": "", "attribute": "ui.theme", "value": "night", "status": "active"}]})
    monkeypatch.setattr(target, "OpenAICompatibleProvider", Provider)
    row = {"natural_sessions": [[{"payload": {"role": "user", "content": "I prefer the night theme."}}]],
           "structured_operations": [{"value": "never-send-this-reference"}]}
    result = target.annotate(row, None)
    assert result["status"] == "agreed"
    assert len(calls) == 2 and calls[0] == calls[1]
    assert "never-send-this-reference" not in json.dumps(calls)


def test_provider_errors_do_not_disclose_text(monkeypatch):
    target = module(monkeypatch)
    class Provider:
        def __init__(self, config):
            raise RuntimeError("private example text")
    monkeypatch.setattr(target, "OpenAICompatibleProvider", Provider)
    result = target.annotate({"natural_sessions": []}, None)
    assert result == {"status": "error", "error_type": "RuntimeError"}
