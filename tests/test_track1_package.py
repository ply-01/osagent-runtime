"""Release package tests ensure private runtime state is not redistributed."""
import hashlib
import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("track1_package", ROOT / "scripts/package-track1.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_archive_allowlist_and_exact_hashes(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    for name in module.TOP_LEVEL:
        (root / name).write_text("public source")
    for name in ["src/os_agent/example.py", "config/llm.example.json",
                 "config/llm.local.json", "data/events/private.jsonl",
                 "memory/core_facts.json", "src/__pycache__/example.pyc"]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("private" if "local" in name else "content")
    archive = module.package(tmp_path / "release", root=root)
    with ZipFile(archive) as bundle:
        names = bundle.namelist()
        assert "osagent/src/os_agent/example.py" in names
        assert "osagent/config/llm.example.json" in names
        assert not any("local.json" in n or "/memory/" in n or "/data/" in n or "__pycache__" in n for n in names)
        manifest = json.loads(bundle.read("osagent/SOURCE-MANIFEST.json"))
        assert set(names) == {"osagent/" + n for n in manifest["sha256"]} | {"osagent/SOURCE-MANIFEST.json"}
        for name, digest in manifest["sha256"].items():
            assert hashlib.sha256(bundle.read("osagent/" + name)).hexdigest() == digest
