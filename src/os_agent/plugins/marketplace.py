"""Small, auditable plugin marketplace for OSagent extensions.

The marketplace deliberately installs plugin bundles as data. It never imports
or executes downloaded code. Declarative skills can be returned as bounded
context, while remote MCP declarations are exposed as sanitized connection
descriptors for a trusted host such as Open WebUI.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse


MAX_PLUGIN_ARCHIVE_BYTES = 100 * 1024 * 1024
MAX_PLUGIN_SKILL_FILES = 8
MAX_PLUGIN_SKILL_BYTES = 40_000
MAX_PLUGIN_MCP_CONFIG_BYTES = 64_000
MAX_PLUGIN_MCP_SERVERS = 8


@dataclass(frozen=True, slots=True)
class PluginManifest:
    name: str
    version: str
    display_name: str = ""
    description: str = ""
    category: str = ""
    featured: bool = False
    author: str = ""
    homepage: str = ""
    download_url: str = ""
    sha256: str = ""
    permissions: tuple[str, ...] = ()
    source_path: str = ""
    source_repository: str = ""
    source_revision: str = ""
    source_marketplace: str = ""
    compatibility: str = ""

    @classmethod
    def from_dict(cls, value: object) -> "PluginManifest":
        if not isinstance(value, dict):
            raise ValueError("plugin manifest must be an object")
        name = str(value.get("name", "")).strip()
        version = str(value.get("version", "")).strip()
        if not name or not version:
            raise ValueError("plugin manifest requires name and version")
        permissions = value.get("permissions", ())
        if not isinstance(permissions, (list, tuple)):
            raise ValueError("plugin permissions must be an array")
        return cls(
            name=name,
            version=version,
            display_name=str(value.get("display_name", "")),
            description=str(value.get("description", "")),
            category=str(value.get("category", "")),
            featured=bool(value.get("featured", False)),
            author=str(value.get("author", "")),
            homepage=str(value.get("homepage", "")),
            download_url=str(value.get("download_url", "")),
            sha256=str(value.get("sha256", "")).lower(),
            permissions=tuple(str(item) for item in permissions),
            source_path=str(value.get("source_path", "")),
            source_repository=str(value.get("source_repository", "")),
            source_revision=str(value.get("source_revision", "")),
            source_marketplace=str(value.get("source_marketplace", "")),
            compatibility=str(value.get("compatibility", "")),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self) | {"permissions": list(self.permissions)}


class PluginMarketplace:
    """Discover and install verified plugin bundles under a fixed directory."""

    def __init__(
        self,
        root: Path | str,
        *,
        catalog_path: Path | str | None = None,
        source_root: Path | str | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.catalog_path = Path(
            catalog_path
            or os.getenv("OS_AGENT_PLUGIN_CATALOG", str(self.root / "catalog.json"))
        ).expanduser()
        self.source_root = Path(
            source_root
            or os.getenv("OS_AGENT_PLUGIN_SOURCE_ROOT", str(self.root / "sources"))
        ).expanduser().resolve()
        self.state_path = self.root / "marketplace-state.json"

    def search(self, query: str = "") -> list[dict[str, object]]:
        needle = query.strip().casefold()
        manifests = self._load_catalog()
        return [
            item.to_dict()
            for item in manifests
            if not needle
            or needle in item.name.casefold()
            or needle in item.display_name.casefold()
            or needle in item.description.casefold()
            or needle in item.category.casefold()
            or needle in item.author.casefold()
        ]

    def list(self, query: str = "") -> list[dict[str, object]]:
        """Return catalog entries enriched with persisted lifecycle status.

        Plugins are installed as data bundles. ``enabled`` controls whether a
        trusted host may load their declarative instructions or connect to a
        sanitized remote MCP endpoint; it never imports or executes plugin code.
        """
        needle = query.strip().casefold()
        state = self._load_state()
        entries: list[dict[str, object]] = []
        for manifest in self._load_catalog():
            if needle and not (
                needle in manifest.name.casefold()
                or needle in manifest.display_name.casefold()
                or needle in manifest.description.casefold()
                or needle in manifest.category.casefold()
                or needle in manifest.author.casefold()
            ):
                continue
            key = _plugin_key(manifest)
            destination = self._destination(manifest)
            installed = destination.exists()
            saved = state.get(key, {})
            enabled = bool(saved.get("enabled", True)) if installed else False
            runtime = _inspect_plugin_runtime(destination) if installed else _empty_runtime()
            runtime_status = _runtime_status(
                installed=installed,
                enabled=enabled,
                runtime=runtime,
            )
            skill_invocation_available = bool(
                installed and enabled and runtime["skill_files"]
            )
            entry = manifest.to_dict()
            entry.update(
                {
                    "installed": installed,
                    "enabled": enabled,
                    "status": (
                        "enabled" if installed and enabled else
                        "disabled" if installed else "available"
                    ),
                    "install_path": str(destination) if installed else "",
                    "runtime_status": runtime_status,
                    "runtime": runtime,
                    "connectors": runtime["mcp_servers"] if installed and enabled else [],
                    "invocation_available": skill_invocation_available,
                    "invocation": (
                        {
                            "tool": "os_agent_plugin_marketplace",
                            "action": "invoke",
                            "plugin": manifest.name,
                            "protocol": "declarative_skill_files",
                        }
                        if skill_invocation_available
                        else None
                    ),
                }
            )
            entries.append(entry)
        return entries

    def invoke(self, name: str, *, request: str = "") -> dict[str, object]:
        """Return an enabled plugin's declarative instructions for an agent.

        Marketplace bundles remain data-only. Invocation reads bounded ``SKILL.md``
        files and returns them as context; it never imports, executes, or shells
        out to downloaded plugin code.
        """
        manifest = self._manifest(name)
        destination = self._destination(manifest)
        state = self._load_state()
        saved = state.get(_plugin_key(manifest), {})
        if not destination.exists():
            raise ValueError(f"plugin is not installed: {name}")
        if not bool(saved.get("enabled", True)):
            raise PermissionError(f"plugin is disabled: {name}")
        clean_request = str(request or "").strip()
        if len(clean_request) > 8_000:
            raise ValueError("plugin invocation request exceeds the 8,000 character limit")

        instructions: list[dict[str, str]] = []
        total_bytes = 0
        for skill_file in _find_skill_files(destination):
            if len(instructions) >= MAX_PLUGIN_SKILL_FILES:
                break
            try:
                raw = skill_file.read_bytes()
            except OSError:
                continue
            remaining = MAX_PLUGIN_SKILL_BYTES - total_bytes
            if remaining <= 0:
                break
            text = raw[:remaining].decode("utf-8", errors="replace").strip()
            if not text:
                continue
            total_bytes += len(text.encode("utf-8"))
            instructions.append(
                {
                    "path": str(skill_file.relative_to(destination)).replace("\\", "/"),
                    "content": text,
                }
            )
        if not instructions:
            runtime = _inspect_plugin_runtime(destination)
            if runtime["mcp_servers"]:
                raise ValueError(
                    "plugin exposes remote MCP tools; connect it in the host UI before use"
                )
            raise ValueError("plugin does not contain a supported declarative SKILL.md")
        return {
            "invoked": True,
            "plugin": manifest.to_dict(),
            "request": clean_request,
            "instructions": instructions,
            "execution": "declarative_only",
        }

    def install(self, name: str) -> dict[str, object]:
        if os.getenv("OS_AGENT_PLUGIN_INSTALL_ENABLED", "false").lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            raise PermissionError(
                "plugin installation is disabled; set OS_AGENT_PLUGIN_INSTALL_ENABLED=true"
            )
        manifest = self._manifest(name)
        if manifest.source_path:
            return self._install_local_source(manifest)
        parsed = urlparse(manifest.download_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("plugin download_url must use HTTPS")
        if len(manifest.sha256) != 64 or any(c not in "0123456789abcdef" for c in manifest.sha256):
            raise ValueError("plugin sha256 must be a 64-character hex digest")

        self.root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="osagent-plugin-") as temporary:
            archive = Path(temporary) / "plugin.bundle"
            request = urllib.request.Request(
                manifest.download_url,
                headers={"User-Agent": "OSagent-plugin-market/1"},
            )
            with urllib.request.urlopen(request, timeout=30) as response, archive.open("wb") as output:
                headers = getattr(response, "headers", {})
                declared_size = headers.get("Content-Length")
                try:
                    declared_bytes = int(declared_size) if declared_size else 0
                except ValueError:
                    declared_bytes = 0
                if declared_bytes > MAX_PLUGIN_ARCHIVE_BYTES:
                    raise ValueError("plugin archive exceeds the 100 MiB limit")
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_PLUGIN_ARCHIVE_BYTES:
                        raise ValueError("plugin archive exceeds the 100 MiB limit")
                    output.write(chunk)
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            if digest != manifest.sha256:
                raise ValueError(f"plugin checksum mismatch for {manifest.name}")
            destination = self._destination(manifest)
            temporary_destination = self.root / f".{destination.name}.tmp"
            if temporary_destination.exists():
                shutil.rmtree(temporary_destination)
            temporary_destination.mkdir()
            _extract_zip_safely(archive, temporary_destination)
            (temporary_destination / "marketplace-manifest.json").write_text(
                json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if destination.exists() or destination.is_symlink():
                _remove_path(destination)
            temporary_destination.replace(destination)
        state = self._load_state()
        state[_plugin_key(manifest)] = {"enabled": True}
        self._save_state(state)
        return {
            "installed": True,
            "enabled": True,
            "status": "enabled",
            "path": str(destination),
            "manifest": manifest.to_dict(),
        }

    def _install_local_source(self, manifest: PluginManifest) -> dict[str, object]:
        source = _resolve_local_source(self.source_root, manifest.source_path)
        if not (source / ".codex-plugin" / "plugin.json").is_file():
            raise ValueError("local plugin source is missing .codex-plugin/plugin.json")

        self.root.mkdir(parents=True, exist_ok=True)
        destination = self._destination(manifest)
        temporary_destination = self.root / f".{destination.name}.tmp"
        if temporary_destination.exists() or temporary_destination.is_symlink():
            _remove_path(temporary_destination)
        _copy_directory_safely(source, temporary_destination)
        (temporary_destination / "marketplace-manifest.json").write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if destination.exists() or destination.is_symlink():
            _remove_path(destination)
        temporary_destination.replace(destination)

        state = self._load_state()
        state[_plugin_key(manifest)] = {"enabled": True}
        self._save_state(state)
        return {
            "installed": True,
            "enabled": True,
            "status": "enabled",
            "path": str(destination),
            "manifest": manifest.to_dict(),
        }

    def set_enabled(self, name: str, enabled: bool) -> dict[str, object]:
        """Enable or disable an installed plugin without executing it."""
        manifest = self._manifest(name)
        destination = self._destination(manifest)
        if not destination.exists():
            raise ValueError(f"plugin is not installed: {name}")
        state = self._load_state()
        state[_plugin_key(manifest)] = {"enabled": bool(enabled)}
        self._save_state(state)
        return {
            "installed": True,
            "enabled": bool(enabled),
            "status": "enabled" if enabled else "disabled",
            "path": str(destination),
            "manifest": manifest.to_dict(),
        }

    def uninstall(self, name: str) -> dict[str, object]:
        """Remove an installed bundle and its lifecycle state."""
        manifest = self._manifest(name)
        destination = self._destination(manifest)
        if destination.exists() or destination.is_symlink():
            _remove_path(destination)
        state = self._load_state()
        state.pop(_plugin_key(manifest), None)
        self._save_state(state)
        return {
            "installed": False,
            "enabled": False,
            "status": "available",
            "path": str(destination),
            "manifest": manifest.to_dict(),
        }

    def _manifest(self, name: str) -> PluginManifest:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("plugin name must not be empty")
        manifest = next(
            (item for item in self._load_catalog() if item.name == clean_name),
            None,
        )
        if manifest is None:
            raise ValueError(f"plugin is not listed in the marketplace: {clean_name}")
        return manifest

    def _destination(self, manifest: PluginManifest) -> Path:
        return self.root / f"{_safe_name(manifest.name)}-{_safe_name(manifest.version)}"

    def _load_state(self) -> dict[str, dict[str, object]]:
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        values = payload.get("plugins", {}) if isinstance(payload, dict) else {}
        if not isinstance(values, dict):
            return {}
        return {
            str(key): value
            for key, value in values.items()
            if isinstance(value, dict)
        }

    def _save_state(self, state: dict[str, dict[str, object]]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_name(f".{self.state_path.name}.tmp")
        temporary.write_text(
            json.dumps({"version": 1, "plugins": state}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.state_path)

    def _load_catalog(self) -> list[PluginManifest]:
        try:
            payload = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        values = payload.get("plugins", payload) if isinstance(payload, dict) else payload
        if not isinstance(values, list):
            return []
        manifests: list[PluginManifest] = []
        for value in values:
            try:
                manifests.append(PluginManifest.from_dict(value))
            except ValueError:
                continue
        return manifests


def _extract_zip_safely(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        base = destination.resolve()
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if target != base and base not in target.parents:
                raise ValueError("plugin archive contains a path traversal entry")
        bundle.extractall(destination)


def _resolve_local_source(source_root: Path, source_path: str) -> Path:
    relative = Path(source_path)
    if relative.is_absolute() or not source_path.strip():
        raise ValueError("plugin source_path must be a non-empty relative path")
    base = source_root.resolve()
    candidate = (base / relative).resolve()
    if candidate == base or base not in candidate.parents:
        raise ValueError("plugin source_path escapes the configured source root")
    try:
        source = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError("local plugin source does not exist") from error
    if source == base or base not in source.parents or not source.is_dir():
        raise ValueError("plugin source_path escapes the configured source root")
    return source


def _find_skill_files(destination: Path) -> list[Path]:
    """Find bounded, regular SKILL.md files without following symlink escapes."""
    files: list[Path] = []
    try:
        candidates = sorted(destination.rglob("SKILL.md"))
    except OSError:
        return files
    base = destination.resolve()
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved != base and base not in resolved.parents:
            continue
        if resolved.is_file() and not candidate.is_symlink():
            files.append(resolved)
    return files


def _empty_runtime() -> dict[str, object]:
    return {
        "kind": "uninspected",
        "skill_files": [],
        "mcp_servers": [],
        "unsupported": [],
    }


def _runtime_status(
    *,
    installed: bool,
    enabled: bool,
    runtime: dict[str, object],
) -> str:
    if not installed:
        return "available"
    if not enabled:
        return "disabled"
    if runtime["skill_files"]:
        return "ready"
    if runtime["mcp_servers"]:
        return "connection_required"
    return "unsupported"


def _inspect_plugin_runtime(destination: Path) -> dict[str, object]:
    skill_files = [
        str(path.relative_to(destination)).replace("\\", "/")
        for path in _find_skill_files(destination)[:MAX_PLUGIN_SKILL_FILES]
    ]
    mcp_servers, unsupported = _read_mcp_servers(destination)
    kinds: list[str] = []
    if skill_files:
        kinds.append("skill")
    if mcp_servers:
        kinds.append("mcp")
    return {
        "kind": "+".join(kinds) if kinds else "unsupported",
        "skill_files": skill_files,
        "mcp_servers": mcp_servers,
        "unsupported": unsupported,
    }


def _read_mcp_servers(destination: Path) -> tuple[list[dict[str, object]], list[str]]:
    """Return safe remote MCP descriptors without exposing headers or secrets."""
    payload = _load_mcp_payload(destination)
    if payload is None:
        return [], []
    values = payload.get("mcpServers", payload)
    if not isinstance(values, dict):
        return [], ["MCP configuration must contain an mcpServers object"]

    servers: list[dict[str, object]] = []
    unsupported: list[str] = []
    plugin_manifest = _read_json_file(destination / "marketplace-manifest.json") or {}
    plugin_name = str(plugin_manifest.get("name", destination.name))
    for raw_name, raw_config in list(values.items())[:MAX_PLUGIN_MCP_SERVERS]:
        name = str(raw_name).strip()
        if not name or not isinstance(raw_config, dict):
            unsupported.append("MCP server entry has an invalid name or configuration")
            continue
        if raw_config.get("command") or raw_config.get("args"):
            unsupported.append(f"{name}: local command MCP servers are not auto-connected")
            continue
        if raw_config.get("headers") or raw_config.get("env"):
            unsupported.append(f"{name}: credential-bearing MCP configuration requires manual review")
            continue
        url = str(raw_config.get("url", "")).strip()
        parsed = urlparse(url)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            unsupported.append(f"{name}: remote MCP URL must be credential-free HTTPS")
            continue
        oauth_resource = str(raw_config.get("oauth_resource", "")).strip()
        oauth_scope = str(raw_config.get("oauth_scope", "")).strip()
        servers.append(
            {
                "id": _mcp_connection_id(plugin_name, name),
                "name": name,
                "url": url,
                "transport": str(raw_config.get("type", "http") or "http"),
                "auth_type": "oauth_2.1" if oauth_resource else "none",
                "oauth_resource": oauth_resource,
                "oauth_scope": oauth_scope,
                "connectable": True,
            }
        )
    if len(values) > MAX_PLUGIN_MCP_SERVERS:
        unsupported.append(
            f"only the first {MAX_PLUGIN_MCP_SERVERS} MCP server declarations are considered"
        )
    return servers, unsupported


def _load_mcp_payload(destination: Path) -> dict[str, object] | None:
    plugin_config = _read_json_file(destination / ".codex-plugin" / "plugin.json") or {}
    declaration = plugin_config.get("mcpServers")
    if isinstance(declaration, dict):
        return {"mcpServers": declaration}
    candidates: list[Path] = []
    if isinstance(declaration, str) and declaration.strip():
        candidates.append(destination / declaration)
    candidates.append(destination / ".mcp.json")
    for candidate in candidates:
        payload = _read_json_file(candidate, root=destination)
        if payload is not None:
            return payload
    return None


def _read_json_file(path: Path, *, root: Path | None = None) -> dict[str, object] | None:
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return None
    if root is not None:
        base = root.resolve()
        if resolved != base and base not in resolved.parents:
            return None
    if path.is_symlink() or not resolved.is_file():
        return None
    try:
        raw = resolved.read_bytes()
    except OSError:
        return None
    if len(raw) > MAX_PLUGIN_MCP_CONFIG_BYTES:
        return None
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _mcp_connection_id(plugin_name: str, server_name: str) -> str:
    raw = f"osagent-plugin-{plugin_name}-{server_name}".lower()
    return _safe_name(raw).replace(".", "-")


def _copy_directory_safely(source: Path, destination: Path) -> None:
    total = 0
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError("local plugin source contains a symbolic link")
        if path.is_file():
            total += path.stat().st_size
            if total > MAX_PLUGIN_ARCHIVE_BYTES:
                raise ValueError("plugin source exceeds the 100 MiB limit")
    shutil.copytree(source, destination)


def _safe_name(value: str) -> str:
    result = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return result[:100] or "plugin"


def _remove_path(path: Path) -> None:
    """Remove an installed entry without following a symlink."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def _plugin_key(manifest: PluginManifest) -> str:
    return f"{manifest.name}@{manifest.version}"
