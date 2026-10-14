"""Check configured semantic extraction without benchmark content or secrets."""
import json
import tempfile
import re
from pathlib import Path
from urllib.parse import urlsplit

from os_agent.llm.provider import LLMConfig
from os_agent.memory.memory_config import MemoryConfig
from os_agent.memory.semantic_extractor import create_semantic_extractor


def main():
    config = LLMConfig.from_env()
    result = {"model": config.model, "host": urlsplit(config.base_url).hostname,
              "credential_present": bool(config.api_key)}
    if not config.api_key:
        print(json.dumps(result))
        return 2
    with tempfile.TemporaryDirectory() as root:
        try:
            extractor = create_semantic_extractor(MemoryConfig(Path(root) / "config.json"))
            if extractor is None:
                raise RuntimeError("semantic mode not enabled")
            facts = extractor.extract([{
                "event_type": "message_start", "event_id": "development-smoke",
                "payload": {"role": "user", "content": "For my personal archive, I prefer plain text as the document format."},
            }], "independent-development-smoke")
            result.update(fact_count=len(facts), stats=dict(extractor.stats), connected=True)
        except Exception as exc:
            result.update(connected=False, error_type=type(exc).__name__)
            chain = []
            current = exc
            while current is not None and len(chain) < 4:
                message = str(current)
                status = re.search(r"HTTP error (\d{3})", message)
                chain.append({"type": type(current).__name__,
                              "http_status": status.group(1) if status else None,
                              "timeout": "timed out" in message.lower(),
                              "certificate": "certificate" in message.lower(),
                              "validation": message if type(current).__name__ == "SemanticExtractionError" else None})
                current = current.__cause__ or current.__context__
            result["error_chain"] = chain
    print(json.dumps(result))
    return 0 if result.get("connected") and result.get("fact_count") else 1


if __name__ == "__main__":
    raise SystemExit(main())
