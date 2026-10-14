#!/usr/bin/env python3
"""Build an allowlisted source archive, with a fresh per-file SHA-256 manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = {
    "README.md", "SUBMISSION-README.md", "pyproject.toml", "uv.lock", "LICENSE",
    "THIRD_PARTY_NOTICES.md", "CONTRIBUTING.md", "GOVERNANCE.md", "ROADMAP.md",
    "SECURITY.md", "CHANGELOG.md", ".gitignore", ".gitmodules", ".dockerignore",
}
DIRECTORIES = {"src", "tests", "benchmarks", "scripts", "native", "integrations", "deploy", "examples", "skills", ".github"}
EXCLUDED_PARTS = {"__pycache__", "node_modules", ".git", ".venv", ".pytest_cache"}


def collect_files(root: Path) -> dict[str, bytes]:
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        name = relative.as_posix()
        allowed = (name in TOP_LEVEL or relative.parts[0] in DIRECTORIES
                   or (relative.parts[0] == "config" and path.suffix == ".json")
                   or name.startswith("doc/track1/") or name.startswith("doc/evaluation/")
                   or name in {"doc/legacy-kylin-readme.md", "doc/legacy-source-manifest.json"}
                   or name == "vendor/README.md" or name.startswith("vendor/patches/"))
        if not allowed or path.is_dir():
            continue
        if path.is_symlink():
            raise ValueError(f"symlink refused: {name}")
        if path.name.endswith((".pyc", ".pyo", ".local.json")) or path.name.startswith(".env"):
            continue
        if path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError(f"unexpected large file: {name}")
        files[name] = path.read_bytes()
    required = TOP_LEVEL - {".gitmodules", ".dockerignore"}
    missing = required - files.keys()
    if missing:
        raise ValueError(f"missing release files: {sorted(missing)}")
    return files


def package(output: Path, root: Path = ROOT) -> Path:
    files = collect_files(root)
    manifest = {
        "schema_version": 1,
        "description": "OpenAtom 2026 track-one source snapshot; hashes cover exact archive file bytes",
        "baseline_source_commit": "6ca20ca8155ae148cf95d874e068cf287e00cadf",
        "baseline_note": "Original archive provenance only; modified release history is tracked separately in the project repository",
        "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    }
    encoded = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "osagent-openatom2026-track1.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
        for name, data in files.items():
            bundle.writestr("osagent/" + name, data)
        bundle.writestr("osagent/SOURCE-MANIFEST.json", encoded)
    # Verify the delivered bytes, not only the original files.
    with ZipFile(archive) as bundle:
        for name, digest in manifest["sha256"].items():
            if hashlib.sha256(bundle.read("osagent/" + name)).hexdigest() != digest:
                raise ValueError(f"archive verification failed: {name}")
        if bundle.testzip() is not None:
            raise ValueError("archive CRC check failed")
    (output / "SOURCE-MANIFEST.json").write_bytes(encoded)
    (output / (archive.name + ".sha256")).write_text(hashlib.sha256(archive.read_bytes()).hexdigest() + "  " + archive.name + "\n")
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    print(package(args.output))
