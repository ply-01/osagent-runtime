"""ctypes wrapper for the official Kylin vector-engine SDK bridge."""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Self

from os_agent.memory.embedding_provider import EmbeddingError

_BRIDGE_NAME = "libosagent_kylin_vector_bridge.so"
_BRIDGE_ABI_VERSION = 2


def kylin_vector_bridge_path(explicit: str | None = None) -> Path | None:
    """Return the installed bridge artifact without loading the SDK runtime."""
    try:
        home_bridge = str(Path.home() / ".local/lib" / _BRIDGE_NAME)
    except RuntimeError:
        # A constrained test/service environment may not expose HOME.
        home_bridge = None
    candidates = (
        explicit,
        os.getenv("OS_AGENT_KYLIN_VECTOR_BRIDGE_LIBRARY"),
        str(Path(__file__).resolve().parents[3] / "native/kylin_vector_bridge" / _BRIDGE_NAME),
        home_bridge,
        f"/usr/local/lib/{_BRIDGE_NAME}",
    )
    for candidate in candidates:
        if candidate:
            path = Path(candidate)
            if path.is_file():
                return path
    return None


def kylin_vector_bridge_available(explicit: str | None = None) -> bool:
    """Check that the C++ bridge was installed; do not dlopen during preflight."""
    return kylin_vector_bridge_path(explicit) is not None


def kylin_vector_bridge_load_error(explicit: str | None = None) -> str | None:
    """Return a diagnostic when the bridge or its SDK dependencies cannot load."""
    path = kylin_vector_bridge_path(explicit)
    if path is None:
        return "the C++ libkysdk-vector-engine-client bridge was not installed"
    try:
        library = ctypes.CDLL(_stage_library(path))
    except (EmbeddingError, OSError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return _bridge_abi_error(library)


def kylin_vector_bridge_status(explicit: str | None = None) -> dict[str, object]:
    """Return loadable bridge state without opening a vector collection.

    Presence alone is not sufficient: a bridge compiled against a different
    Kylin SDK ABI must be reported as unavailable so callers use the explicit
    fallback and operators get an actionable diagnostic.
    """
    path = kylin_vector_bridge_path(explicit)
    if path is None:
        return {
            "installed": False,
            "loadable": False,
            "abi_version": _BRIDGE_ABI_VERSION,
            "error": "bridge artifact is not installed",
        }
    error = kylin_vector_bridge_load_error(explicit)
    return {
        "installed": True,
        "loadable": error is None,
        "abi_version": _BRIDGE_ABI_VERSION,
        "path": str(path),
        "error": error,
    }


def _bridge_abi_error(library: ctypes.CDLL) -> str | None:
    try:
        version = library.osagent_kylin_vector_abi_version
    except AttributeError:
        return "bridge ABI version symbol is missing; rebuild the native bridge"
    version.argtypes = []
    version.restype = ctypes.c_uint32
    actual = int(version())
    if actual != _BRIDGE_ABI_VERSION:
        return (
            f"bridge ABI version mismatch: expected {_BRIDGE_ABI_VERSION}, "
            f"got {actual}"
        )
    return None


class KylinVectorEngine:
    """Persistent vector collection backed by libkysdk-vector-engine-client.

    The bridge exposes only a small C ABI. This class owns the collection
    handle and translates Python lists to contiguous ctypes arrays. Vector
    payloads stay in the system engine; the sidecar state file only tracks
    active logical keys so stale rows can be removed during index refresh.
    """

    def __init__(
        self,
        *,
        collection: str,
        dimension: int,
        state_path: Path,
        library_path: str | None = None,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.collection = collection
        self.dimension = dimension
        self.state_path = Path(state_path)
        self._library_path = self._resolve_library(library_path)
        self._library = ctypes.CDLL(self._library_path)
        abi_error = _bridge_abi_error(self._library)
        if abi_error:
            raise EmbeddingError(abi_error)
        self._configure_abi()
        self._handle = ctypes.c_void_p()
        error = ctypes.create_string_buffer(4096)
        code = self._library.osagent_kylin_vector_open(
            collection.encode("utf-8"),
            dimension,
            ctypes.byref(self._handle),
            error,
            len(error),
        )
        if code != 0 or not self._handle.value:
            raise EmbeddingError(self._error(code, error, "failed to open Kylin vector collection"))
        self._keys = self._read_state()

    @classmethod
    def from_environment(cls, root: Path, *, kind: str, dimension: int) -> KylinVectorEngine:
        prefix = _safe_name(os.getenv("OS_AGENT_KYLIN_VECTOR_COLLECTION_PREFIX", "osagent_memory"))
        collection = f"{prefix}_{_safe_name(kind)}_{dimension}d"
        state_dir = Path(
            os.getenv("OS_AGENT_KYLIN_VECTOR_STATE_DIR", str(Path(root) / "vector-engine"))
        )
        bridge = os.getenv("OS_AGENT_KYLIN_VECTOR_BRIDGE_LIBRARY") or None
        return cls(
            collection=collection,
            dimension=dimension,
            state_path=state_dir / f"{_safe_name(kind)}.json",
            library_path=bridge,
        )

    @staticmethod
    def _resolve_library(explicit: str | None) -> str:
        path = kylin_vector_bridge_path(explicit)
        if path is not None:
            return _stage_library(path)
        raise EmbeddingError(
            "Kylin vector bridge is unavailable; build native/kylin_vector_bridge "
            "and set OS_AGENT_KYLIN_VECTOR_BRIDGE_LIBRARY"
        )

    def _configure_abi(self) -> None:
        lib = self._library
        lib.osagent_kylin_vector_open.argtypes = [
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.osagent_kylin_vector_open.restype = ctypes.c_int
        lib.osagent_kylin_vector_close.argtypes = [ctypes.c_void_p]
        lib.osagent_kylin_vector_close.restype = None
        lib.osagent_kylin_vector_key_id.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.osagent_kylin_vector_key_id.restype = ctypes.c_int
        lib.osagent_kylin_vector_upsert.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.osagent_kylin_vector_upsert.restype = ctypes.c_int
        lib.osagent_kylin_vector_delete.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.osagent_kylin_vector_delete.restype = ctypes.c_int
        lib.osagent_kylin_vector_search.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.osagent_kylin_vector_search.restype = ctypes.c_int
        lib.osagent_kylin_vector_free.argtypes = [ctypes.c_void_p]
        lib.osagent_kylin_vector_free.restype = None

    def close(self) -> None:
        if self._handle and self._handle.value:
            self._library.osagent_kylin_vector_close(self._handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def replace(self, documents: Iterable[tuple[str, list[float]]]) -> int:
        rows = [(str(key), vector) for key, vector in documents]
        next_keys = {key for key, _ in rows}
        stale = sorted(self._keys - next_keys)
        if stale:
            self._delete(stale)
        if rows:
            self._upsert(rows)
        self._keys = next_keys
        self._write_state()
        return len(rows)

    def search(
        self,
        query: list[float],
        limit: int,
        *,
        allowed_keys: Iterable[str] | None = None,
    ) -> list[tuple[str, float]]:
        if not query or limit <= 0:
            return []
        allowed = None if allowed_keys is None else sorted(set(allowed_keys))
        if allowed is not None and not allowed:
            return []
        encoded_allowed = [key.encode("utf-8") for key in allowed or []]
        allowed_array = (
            (ctypes.c_char_p * len(encoded_allowed))(*encoded_allowed)
            if encoded_allowed
            else None
        )
        values = (ctypes.c_float * len(query))(*[float(value) for value in query])
        output = ctypes.c_void_p()
        output_size = ctypes.c_size_t()
        error = ctypes.create_string_buffer(4096)
        code = self._library.osagent_kylin_vector_search(
            self._handle,
            values,
            len(query),
            limit,
            allowed_array,
            len(encoded_allowed),
            ctypes.byref(output),
            ctypes.byref(output_size),
            error,
            len(error),
        )
        if code != 0:
            raise EmbeddingError(self._error(code, error, "Kylin vector search failed"))
        try:
            payload = ctypes.string_at(output, output_size.value).decode("utf-8") if output else ""
        finally:
            if output:
                self._library.osagent_kylin_vector_free(output)
        rows = [line.split("\t", 2) for line in payload.splitlines()]
        # ABI v2 returns the original VARCHAR key. Only older bridge output
        # needs the numeric-id compatibility map; eagerly rebuilding it makes
        # every 10k search perform 10k ctypes calls.
        needs_id_fallback = any(len(fields) < 3 for fields in rows)
        by_id = self._key_ids(self._keys) if needs_id_fallback else {}
        hits: list[tuple[str, float]] = []
        for fields in rows:
            if len(fields) < 2:
                continue
            try:
                numeric_id = int(fields[0])
                score = float(fields[1])
            except ValueError:
                continue
            key = fields[2] if len(fields) >= 3 else by_id.get(numeric_id)
            if key is not None:
                hits.append((key, score))
        return hits[:limit]

    def _upsert(self, rows: list[tuple[str, list[float]]]) -> None:
        encoded_keys = [key.encode("utf-8") for key, _ in rows]
        key_array = (ctypes.c_char_p * len(encoded_keys))(*encoded_keys)
        flat = [float(value) for _, vector in rows for value in vector]
        expected = len(rows) * self.dimension
        if len(flat) != expected:
            raise EmbeddingError(f"Kylin vector row dimension mismatch: expected {expected}, got {len(flat)}")
        vector_array = (ctypes.c_float * len(flat))(*flat)
        error = ctypes.create_string_buffer(4096)
        code = self._library.osagent_kylin_vector_upsert(
            self._handle,
            key_array,
            vector_array,
            len(rows),
            self.dimension,
            error,
            len(error),
        )
        if code != 0:
            raise EmbeddingError(self._error(code, error, "Kylin vector upsert failed"))

    def _delete(self, keys: list[str]) -> None:
        encoded = [key.encode("utf-8") for key in keys]
        key_array = (ctypes.c_char_p * len(encoded))(*encoded)
        error = ctypes.create_string_buffer(4096)
        code = self._library.osagent_kylin_vector_delete(
            self._handle,
            key_array,
            len(encoded),
            error,
            len(error),
        )
        if code != 0:
            raise EmbeddingError(self._error(code, error, "Kylin vector delete failed"))

    def _key_ids(self, keys: Iterable[str]) -> dict[int, str]:
        result: dict[int, str] = {}
        error = ctypes.create_string_buffer(256)
        for key in keys:
            numeric = ctypes.c_int64()
            code = self._library.osagent_kylin_vector_key_id(
                key.encode("utf-8"), ctypes.byref(numeric), error, len(error)
            )
            if code == 0:
                result[int(numeric.value)] = key
        return result

    def _read_state(self) -> set[str]:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return set()
        keys = data.get("keys") if isinstance(data, dict) else None
        return {str(key) for key in keys} if isinstance(keys, list) else set()

    def _write_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"collection": self.collection, "dimension": self.dimension, "keys": sorted(self._keys)}
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.state_path)

    @staticmethod
    def _error(code: int, buffer: ctypes.Array[ctypes.c_char], fallback: str) -> str:
        detail = bytes(buffer).split(b"\0", 1)[0].decode("utf-8", errors="replace")
        return f"{fallback} (code={code}): {detail or 'unknown error'}"


def _safe_name(value: str) -> str:
    result = "".join(character if character.isalnum() or character == "_" else "_" for character in value)
    return result[:180] or "osagent"


def _stage_library(source: Path) -> str:
    """Return the installed bridge or an explicitly requested staged copy.

    Kylin execution control associates trust with the installed path. Copying
    a verified bridge to /tmp creates an unverified object and can make dlopen
    fail with ``failed to map segment``. Therefore staging is opt-in; an
    operator who sets a stage directory is responsible for permitting that
    path under the target OS policy. A source signature keeps explicit staged
    copies from reusing an older artifact.
    """
    try:
        source = source.resolve()
        stage_value = os.getenv("OS_AGENT_KYLIN_VECTOR_STAGE_DIR", "").strip()
        if not stage_value:
            return str(source)
        stage_dir = Path(stage_value).expanduser().resolve()
        if source.parent == stage_dir:
            return str(source)
        stat = source.stat()
        stage_dir.mkdir(parents=True, exist_ok=True)
        staged = stage_dir / (
            f"osagent-kylin-vector-{stat.st_size:x}-{stat.st_mtime_ns:x}.so"
        )
        if not staged.is_file() or staged.stat().st_size != stat.st_size:
            temporary = staged.with_suffix(staged.suffix + ".tmp")
            shutil.copy2(source, temporary)
            temporary.chmod(0o755)
            temporary.replace(staged)
        return str(staged)
    except OSError as exc:
        raise EmbeddingError(f"unable to stage Kylin vector bridge: {exc}") from exc
