"""Import the official OpenAI plugin marketplace as local OSagent bundles.

Imported bundles remain data-only. The marketplace can copy, enable, disable,
and remove them, but OSagent does not import or execute their code.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .marketplace import _copy_directory_safely, _remove_path


OPENAI_PLUGIN_REPOSITORY = "https://github.com/openai/plugins"
OPENAI_MARKETPLACE = "openai-curated"
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def import_openai_catalog(
    source: Path | str,
    plugin_root: Path | str,
    *,
    revision: str | None = None,
) -> dict[str, object]:
    checkout = Path(source).expanduser().resolve(strict=True)
    destination_root = Path(plugin_root).expanduser().resolve()
    marketplace_path = checkout / ".agents" / "plugins" / "marketplace.json"
    payload = json.loads(marketplace_path.read_text(encoding="utf-8"))
    entries = payload.get("plugins", []) if isinstance(payload, dict) else []
    if not isinstance(entries, list):
        raise ValueError("OpenAI marketplace plugins must be an array")

    checkout_revision = _git_revision(checkout).strip().lower()
    source_revision = (revision or checkout_revision).strip().lower()
    if not REVISION_PATTERN.fullmatch(source_revision):
        raise ValueError("source revision must be a full 40-character Git commit")
    if source_revision != checkout_revision:
        raise ValueError("source revision does not match the checkout HEAD")

    source_root = destination_root / "sources"
    imported_root_name = f"openai-plugins-{source_revision[:12]}"
    imported_root = source_root / imported_root_name
    source_root.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(tempfile.mkdtemp(prefix=".openai-plugins-", dir=source_root))
    staging_root = staging_parent / imported_root_name
    staging_root.mkdir()

    imported: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    try:
        for index, entry in enumerate(entries):
            try:
                catalog_entry, plugin_source = _translate_entry(
                    checkout,
                    entry,
                    imported_root_name,
                    source_revision,
                    featured=index < 8,
                )
                relative_source = plugin_source.relative_to(checkout)
                plugin_destination = staging_root / relative_source
                try:
                    _copy_directory_safely(plugin_source, plugin_destination)
                except (OSError, shutil.Error) as error:
                    if plugin_destination.exists() or plugin_destination.is_symlink():
                        _remove_path(plugin_destination)
                    raise ValueError(
                        f"could not copy plugin bundle: {type(error).__name__}"
                    ) from error
                imported.append(catalog_entry)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                name = str(entry.get("name", "unknown")) if isinstance(entry, dict) else "unknown"
                skipped.append({"name": name, "reason": str(error)})

        if imported_root.exists() or imported_root.is_symlink():
            _remove_path(imported_root)
        staging_root.replace(imported_root)
    finally:
        if staging_parent.exists():
            shutil.rmtree(staging_parent)

    catalog_path = destination_root / "catalog.json"
    existing = _read_existing_catalog(catalog_path)
    custom = [
        item
        for item in existing
        if str(item.get("source_marketplace", "")) != OPENAI_MARKETPLACE
    ]
    custom_names = {str(item.get("name", "")) for item in custom}
    merged = custom + [item for item in imported if str(item["name"]) not in custom_names]
    catalog_payload = {
        "version": 1,
        "source": {
            "marketplace": OPENAI_MARKETPLACE,
            "repository": OPENAI_PLUGIN_REPOSITORY,
            "revision": source_revision,
        },
        "plugins": merged,
    }
    destination_root.mkdir(parents=True, exist_ok=True)
    temporary_catalog = catalog_path.with_name(f".{catalog_path.name}.tmp")
    temporary_catalog.write_text(
        json.dumps(catalog_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_catalog.replace(catalog_path)

    return {
        "catalog": str(catalog_path),
        "revision": source_revision,
        "imported": len(imported),
        "skipped": skipped,
        "preserved_custom": len(custom),
    }


def _translate_entry(
    checkout: Path,
    entry: object,
    imported_root_name: str,
    revision: str,
    *,
    featured: bool,
) -> tuple[dict[str, object], Path]:
    if not isinstance(entry, dict):
        raise ValueError("marketplace entry must be an object")
    source_value = entry.get("source")
    if not isinstance(source_value, dict):
        raise ValueError("marketplace entry has no source object")
    source_kind = str(source_value.get("source", "unknown"))
    if source_kind != "local":
        raise ValueError(f"unsupported marketplace source type: {source_kind}")
    source_path = str(source_value.get("path", "")).strip()
    if not source_path:
        raise ValueError("marketplace entry has no local source path")
    relative = Path(source_path)
    if relative.is_absolute():
        raise ValueError("marketplace source path must be relative")
    plugin_source = (checkout / relative).resolve(strict=True)
    if checkout not in plugin_source.parents or not plugin_source.is_dir():
        raise ValueError("marketplace source path escapes the checkout")

    plugin_manifest_path = plugin_source / ".codex-plugin" / "plugin.json"
    if not plugin_manifest_path.is_file():
        raise ValueError("missing .codex-plugin/plugin.json")
    manifest = json.loads(plugin_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("plugin manifest must be an object")
    name = str(manifest.get("name", entry.get("name", ""))).strip()
    version = str(manifest.get("version", "")).strip()
    if not name or not version:
        raise ValueError("plugin manifest requires name and version")
    if str(entry.get("name", "")).strip() not in {"", name}:
        raise ValueError("marketplace name does not match plugin manifest")

    interface = manifest.get("interface", {})
    if not isinstance(interface, dict):
        interface = {}
    author_value = manifest.get("author", {})
    author = (
        str(author_value.get("name", ""))
        if isinstance(author_value, dict)
        else str(author_value)
    )
    permissions = interface.get("capabilities", [])
    if not isinstance(permissions, list):
        permissions = []
    category = str(entry.get("category", interface.get("category", "Plugins")))
    local_source_path = Path(imported_root_name) / plugin_source.relative_to(checkout)

    return (
        {
            "name": name,
            "version": version,
            "display_name": str(interface.get("displayName", name)),
            "description": str(
                interface.get("shortDescription", manifest.get("description", ""))
            ),
            "category": category,
            "featured": featured,
            "author": author,
            "homepage": str(manifest.get("homepage", manifest.get("repository", ""))),
            "permissions": [str(item) for item in permissions],
            "source_path": local_source_path.as_posix(),
            "source_repository": OPENAI_PLUGIN_REPOSITORY,
            "source_revision": revision,
            "source_marketplace": OPENAI_MARKETPLACE,
            "compatibility": "codex-bundle",
        },
        plugin_source,
    )


def _read_existing_catalog(path: Path) -> list[dict[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    entries = payload.get("plugins", payload) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def _git_revision(checkout: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Existing openai/plugins checkout")
    parser.add_argument("--plugin-root", required=True, help="OSagent plugin data root")
    parser.add_argument("--revision", help="Pinned 40-character Git revision")
    args = parser.parse_args(argv)
    result = import_openai_catalog(
        args.source,
        args.plugin_root,
        revision=args.revision,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
