"""Readiness checks for the 2026 OS Agent memory competition."""

from __future__ import annotations

import ctypes.util
import json
import os
import platform
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from os_agent.memory.embedding_provider import (
    EmbeddingConfig,
    EmbeddingError,
    create_embedding_provider,
    kylin_embedding_library_candidates,
)
from os_agent.memory.kylin_vector_engine import (
    KylinVectorEngine,
    kylin_vector_bridge_load_error,
    kylin_vector_bridge_path,
)

CheckStatus = Literal["pass", "fail", "warning"]


@dataclass(frozen=True, slots=True)
class CompetitionCheck:
    check_id: str
    label: str
    status: CheckStatus
    required: bool
    detail: str
    remediation: str = ""


@dataclass(frozen=True, slots=True)
class CompetitionPreflightReport:
    checks: tuple[CompetitionCheck, ...]

    @property
    def ready(self) -> bool:
        return all(
            check.status == "pass" for check in self.checks if check.required
        )

    def to_dict(self) -> dict[str, object]:
        required = [check for check in self.checks if check.required]
        passed = [check for check in required if check.status == "pass"]
        return {
            "ready": self.ready,
            "required_passed": len(passed),
            "required_total": len(required),
            "checks": [asdict(check) for check in self.checks],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_text(self) -> str:
        data = self.to_dict()
        state = "READY" if self.ready else "NOT READY"
        lines = [
            (
                f"Competition readiness: {state} "
                f"({data['required_passed']}/{data['required_total']} "
                "required checks passed)"
            )
        ]
        for check in self.checks:
            lines.append(f"[{check.status.upper():7}] {check.label}: {check.detail}")
            if check.status != "pass" and check.remediation:
                lines.append(f"          Fix: {check.remediation}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class CompetitionVectorSmokeReport:
    """Evidence from a real system embedding and vector database round trip."""

    backend: str
    model: str
    collection: str
    dimension: int
    documents_written: int
    top_key: str | None
    filtered_top_key: str | None
    updated_top_key: str | None
    write_verified: bool
    search_verified: bool
    filter_verified: bool
    update_verified: bool
    delete_verified: bool
    error: str = ""

    @property
    def passed(self) -> bool:
        return (
            not self.error
            and self.write_verified
            and self.search_verified
            and self.filter_verified
            and self.update_verified
            and self.delete_verified
        )

    def to_dict(self) -> dict[str, object]:
        return {"passed": self.passed, **asdict(self)}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_text(self) -> str:
        state = "PASS" if self.passed else "FAIL"
        lines = [
            f"Competition vector smoke: {state}",
            f"backend={self.backend} model={self.model or '(runtime default)'}",
            f"collection={self.collection or '(not opened)'} dimension={self.dimension}",
            (
                f"write={self.write_verified} search={self.search_verified} "
                f"filter={self.filter_verified} update={self.update_verified} "
                f"delete={self.delete_verified}"
            ),
            (
                f"top_key={self.top_key or '(none)'} "
                f"filtered_top_key={self.filtered_top_key or '(none)'} "
                f"updated_top_key={self.updated_top_key or '(none)'}"
            ),
        ]
        if self.error:
            lines.append(f"error={self.error}")
        return "\n".join(lines)


def run_competition_vector_smoke(
    *,
    embedding_config: EmbeddingConfig | None = None,
    root: Path | str | None = None,
) -> CompetitionVectorSmokeReport:
    """Exercise the required system embedding and vector SDKs end to end.

    A fixed, dimension-scoped collection makes the command repeatable. The two
    smoke rows are removed before returning, including on error when possible.
    """
    config = embedding_config or EmbeddingConfig.from_env()
    backend = config.backend.lower().replace("-", "_")
    if backend not in {"kylin", "kylin_system", "system_kylin"}:
        return CompetitionVectorSmokeReport(
            backend=config.backend,
            model=config.model or "",
            collection="",
            dimension=0,
            documents_written=0,
            top_key=None,
            filtered_top_key=None,
            updated_top_key=None,
            write_verified=False,
            search_verified=False,
            filter_verified=False,
            update_verified=False,
            delete_verified=False,
            error="embedding_backend must be kylin_system",
        )

    keys = ("competition-smoke-primary", "competition-smoke-secondary")
    texts = (
        "OSagent competition native vector smoke primary",
        "Galaxy Kylin memory bridge secondary verification",
    )
    provider = None
    engine = None
    collection = ""
    dimension = 0
    documents_written = 0
    top_key: str | None = None
    filtered_top_key: str | None = None
    updated_top_key: str | None = None
    write_verified = False
    search_verified = False
    filter_verified = False
    update_verified = False
    delete_verified = False
    error = ""
    cleanup_complete = False

    try:
        provider = create_embedding_provider(config)
        document_vectors = provider.embed_documents(list(texts))
        query_vectors = provider.embed_queries([texts[0]])
        if len(document_vectors) != len(texts) or len(query_vectors) != 1:
            raise ValueError("system embedding returned an unexpected vector count")
        dimension = len(query_vectors[0])
        if dimension <= 0 or any(len(vector) != dimension for vector in document_vectors):
            raise ValueError("system embedding returned inconsistent vector dimensions")

        memory_root = Path(
            root
            or os.getenv("OS_AGENT_MCP_MEMORY_ROOT")
            or os.getenv("OS_AGENT_MEMORY_ROOT")
            or "memory"
        )
        engine = KylinVectorEngine.from_environment(
            memory_root,
            kind="competition_smoke",
            dimension=dimension,
        )
        collection = engine.collection
        documents_written = engine.replace(zip(keys, document_vectors))
        hits = engine.search(query_vectors[0], len(keys))
        returned_keys = [key for key, _ in hits]
        top_key = returned_keys[0] if returned_keys else None
        write_verified = documents_written == len(keys) and keys[0] in returned_keys
        search_verified = top_key == keys[0]

        filtered_hits = engine.search(
            query_vectors[0],
            len(keys),
            allowed_keys={keys[1]},
        )
        filtered_top_key = filtered_hits[0][0] if filtered_hits else None
        filter_verified = (
            filtered_top_key == keys[1]
            and all(key == keys[1] for key, _ in filtered_hits)
        )

        updated = engine.replace(
            (
                (keys[0], document_vectors[1]),
                (keys[1], document_vectors[0]),
            )
        )
        updated_hits = engine.search(query_vectors[0], len(keys))
        updated_top_key = updated_hits[0][0] if updated_hits else None
        update_verified = updated == len(keys) and updated_top_key == keys[1]

        engine.replace([])
        cleanup_complete = True
        remaining = {key for key, _ in engine.search(query_vectors[0], len(keys))}
        delete_verified = not remaining.intersection(keys)
    except (EmbeddingError, OSError, RuntimeError, TypeError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if engine is not None:
            if not cleanup_complete:
                try:
                    engine.replace([])
                except (
                    EmbeddingError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as cleanup_exc:
                    detail = f"cleanup failed: {type(cleanup_exc).__name__}: {cleanup_exc}"
                    error = f"{error}; {detail}" if error else detail
            engine.close()
        close_provider = getattr(provider, "close", None)
        if callable(close_provider):
            close_provider()

    return CompetitionVectorSmokeReport(
        backend=config.backend,
        model=config.model or "",
        collection=collection,
        dimension=dimension,
        documents_written=documents_written,
        top_key=top_key,
        filtered_top_key=filtered_top_key,
        updated_top_key=updated_top_key,
        write_verified=write_verified,
        search_verified=search_verified,
        filter_verified=filter_verified,
        update_verified=update_verified,
        delete_verified=delete_verified,
        error=error,
    )


def run_competition_preflight(
    *,
    embedding_config: EmbeddingConfig | None = None,
    os_release_path: Path | str = "/etc/os-release",
) -> CompetitionPreflightReport:
    """Inspect the host without claiming unavailable SDK integrations work."""
    config = embedding_config or EmbeddingConfig.from_env()
    os_release = _read_os_release(Path(os_release_path))
    pretty_name = os_release.get("PRETTY_NAME") or os_release.get("NAME") or "unknown"
    os_blob = " ".join(os_release.values()).lower()
    is_linux = platform.system().lower() == "linux"
    is_kylin = "kylin" in os_blob or "麒麟" in os_blob
    is_desktop = "desktop" in os_blob or "桌面" in os_blob
    version_blob = " ".join(
        os_release.get(key, "") for key in ("VERSION_ID", "VERSION", "PRETTY_NAME")
    )
    is_v11 = bool(re.search(r"(?i)(?:^|\W)v?11(?:\W|$)", version_blob))

    backend = config.backend.lower().replace("-", "_")
    embedding_library = _first_loadable_library(
        kylin_embedding_library_candidates(config)
    )
    uid = _effective_user_id()
    embedding_socket_path = (
        f"/tmp/.kylin-ai-runtime-unix/{uid}/core-textembedding.sock"
    )
    embedding_socket = Path(embedding_socket_path)
    vector_library = _first_loadable_library(
        _vector_library_candidates(os.getenv("OS_AGENT_KYLIN_VECTOR_LIBRARY"))
    )
    vector_socket_path = f"/tmp/kylin-ai-vector-engine-{uid}.sock"
    vector_socket = Path(vector_socket_path)
    vector_bridge = kylin_vector_bridge_path(
        os.getenv("OS_AGENT_KYLIN_VECTOR_BRIDGE_LIBRARY")
    )
    vector_bridge_error = kylin_vector_bridge_load_error(
        os.getenv("OS_AGENT_KYLIN_VECTOR_BRIDGE_LIBRARY")
    )

    checks = (
        CompetitionCheck(
            check_id="kylin_desktop_v11",
            label="Galaxy Kylin Desktop V11",
            status=(
                "pass"
                if is_linux and is_kylin and is_desktop and is_v11
                else "fail"
            ),
            required=True,
            detail=f"detected {platform.system()} / {pretty_name}",
            remediation="Run and preserve this preflight on the required V11 target host.",
        ),
        CompetitionCheck(
            check_id="system_embedding_backend",
            label="System Embedding backend configured",
            status=(
                "pass"
                if backend in {"kylin", "kylin_system", "system_kylin"}
                else "fail"
            ),
            required=True,
            detail=f"embedding_backend={config.backend}",
            remediation="Set embedding_backend to kylin_system for the production submission.",
        ),
        CompetitionCheck(
            check_id="system_embedding_library",
            label="System Embedding SDK library",
            status="pass" if embedding_library else "fail",
            required=True,
            detail=embedding_library or "libkysdk-coreai-embedding was not loadable",
            remediation="Install libkylin-coreai-embedding or configure OS_AGENT_EMBEDDING_LIBRARY_PATH.",
        ),
        CompetitionCheck(
            check_id="system_embedding_runtime",
            label="System Embedding runtime socket",
            status="pass" if embedding_socket.exists() else "fail",
            required=True,
            detail=embedding_socket_path,
            remediation="Start the Kylin AI runtime for the current desktop user.",
        ),
        CompetitionCheck(
            check_id="system_vector_library",
            label="System vector database SDK library",
            status="pass" if vector_library else "fail",
            required=True,
            detail=vector_library or "libkysdk-vector-engine-client was not loadable",
            remediation="Install libkysdk-vector-engine-client and its runtime package.",
        ),
        CompetitionCheck(
            check_id="system_vector_runtime",
            label="System vector database runtime socket",
            status="pass" if vector_socket.exists() else "fail",
            required=True,
            detail=vector_socket_path,
            remediation="Start kylin-ai-vector-engine for the current user.",
        ),
        CompetitionCheck(
            check_id="system_vector_integration",
            label="OS Agent system vector integration",
            status="pass" if vector_bridge and not vector_bridge_error else "fail",
            required=True,
            detail=(
                str(vector_bridge)
                if vector_bridge and not vector_bridge_error
                else vector_bridge_error
                or "the C++ libkysdk-vector-engine-client bridge was not installed"
            ),
            remediation=(
                "Install matching libkysdk-vector-engine-client runtime and development "
                "packages, rebuild native/kylin_vector_bridge, and set "
                "OS_AGENT_KYLIN_VECTOR_BRIDGE_LIBRARY."
            ),
        ),
        CompetitionCheck(
            check_id="embedding_model_pinned",
            label="Embedding model pinned",
            status="pass" if config.model else "warning",
            required=False,
            detail=config.model or "the SDK default model will be selected at runtime",
            remediation="Pin a model returned by the system SDK to keep cache dimensions stable.",
        ),
    )
    return CompetitionPreflightReport(checks=checks)


def _read_os_release(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    values: dict[str, str] = {}
    for line in lines:
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _effective_user_id() -> int:
    getter = getattr(os, "geteuid", None) or getattr(os, "getuid", None)
    return int(getter()) if getter else 0


def _first_loadable_library(candidates: list[str]) -> str | None:
    for candidate in candidates:
        path = Path(candidate)
        if path.is_absolute() and path.exists():
            return str(path)
        if not path.is_absolute():
            discovered = ctypes.util.find_library(candidate)
            if discovered:
                return discovered
            if candidate.endswith((".so", ".so.1")):
                stem = candidate.removeprefix("lib").split(".so", 1)[0]
                discovered = ctypes.util.find_library(stem)
                if discovered:
                    return discovered
    return None


def _vector_library_candidates(explicit: str | None) -> list[str]:
    if explicit:
        return [explicit]
    discovered = ctypes.util.find_library("kysdk-vector-engine-client")
    candidates = [
        discovered,
        "libkysdk-vector-engine-client.so.1",
        "libkysdk-vector-engine-client.so",
        "kysdk-vector-engine-client",
    ]
    return list(dict.fromkeys(item for item in candidates if item))
