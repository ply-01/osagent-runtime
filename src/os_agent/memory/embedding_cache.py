"""Small JSON cache for semantic memory embeddings."""

from __future__ import annotations

import json
import sqlite3
from array import array
from hashlib import blake2b
from pathlib import Path


class EmbeddingCache:
    """Persist embedding vectors by provider namespace and text fingerprint."""

    def __init__(self, path: Path | str = "data/vector-cache/memory-embeddings.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._values: dict[str, list[float]] | None = None
        self._transient: dict[str, list[float]] = {}

    def get(self, namespace: str, text: str) -> list[float] | None:
        return self._load().get(self._key(namespace, text))

    def get_many(self, namespace: str, texts: list[str]) -> list[list[float] | None]:
        values = self._load()
        return [values.get(self._key(namespace, text)) for text in texts]

    def set(self, namespace: str, text: str, vector: list[float]) -> None:
        values = self._load()
        values[self._key(namespace, text)] = vector
        self._save(values)

    def set_many(self, namespace: str, rows: dict[str, list[float]]) -> None:
        """Persist a batch with one file rewrite."""
        if not rows:
            return
        values = self._load()
        for text, vector in rows.items():
            values[self._key(namespace, text)] = vector
        self._save(values)

    def get_transient(self, namespace: str, text: str) -> list[float] | None:
        return self._transient.get(self._key(namespace, text))

    def set_many_transient(
        self, namespace: str, rows: dict[str, list[float]]
    ) -> None:
        for text, vector in rows.items():
            self._transient[self._key(namespace, text)] = vector

    def clear(self) -> None:
        """Remove all cached vectors and reset the in-process view."""
        self._values = {}
        self._transient = {}
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def _load(self) -> dict[str, list[float]]:
        if self._values is not None:
            return self._values
        if not self.path.exists():
            self._values = {}
            return self._values
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._values = {}
            return self._values
        if not isinstance(raw, dict):
            self._values = {}
            return self._values
        values: dict[str, list[float]] = {}
        for key, vector in raw.items():
            if isinstance(key, str) and isinstance(vector, list):
                try:
                    values[key] = [float(item) for item in vector]
                except (TypeError, ValueError):
                    continue
        self._values = values
        return self._values

    def _save(self, values: dict[str, list[float]]) -> None:
        self.path.write_text(
            json.dumps(values, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
            newline="\n",
        )

    def _key(self, namespace: str, text: str) -> str:
        stable = f"{namespace}\n{text}".encode()
        return blake2b(stable, digest_size=16).hexdigest()


class SqliteEmbeddingCache:
    """Incremental vector cache without JSON rewrite amplification."""

    def __init__(
        self, path: Path | str = "data/vector-cache/memory-embeddings.sqlite3"
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._transient: dict[str, list[float]] = {}

    def get(self, namespace: str, text: str) -> list[float] | None:
        return self.get_many(namespace, [text])[0]

    def get_many(self, namespace: str, texts: list[str]) -> list[list[float] | None]:
        if not texts:
            return []
        keys = [self._key(namespace, text) for text in texts]
        connection = self._connect()
        try:
            rows: dict[str, bytes] = {}
            for start in range(0, len(keys), 900):
                batch = keys[start : start + 900]
                placeholders = ",".join("?" for _ in batch)
                rows.update(
                    connection.execute(
                        f"SELECT cache_key, vector FROM embeddings "
                        f"WHERE cache_key IN ({placeholders})",
                        batch,
                    ).fetchall()
                )
        finally:
            connection.close()
        results: list[list[float] | None] = []
        for key in keys:
            payload = rows.get(key)
            if payload is None:
                results.append(None)
                continue
            values = array("f")
            values.frombytes(payload)
            results.append(list(values))
        return results

    def set(self, namespace: str, text: str, vector: list[float]) -> None:
        self.set_many(namespace, {text: vector})

    def set_many(self, namespace: str, rows: dict[str, list[float]]) -> None:
        if not rows:
            return
        connection = self._connect()
        try:
            connection.executemany(
                "INSERT OR REPLACE INTO embeddings(cache_key, vector) VALUES (?, ?)",
                (
                    (self._key(namespace, text), array("f", vector).tobytes())
                    for text, vector in rows.items()
                ),
            )
            connection.commit()
        finally:
            connection.close()

    def get_transient(self, namespace: str, text: str) -> list[float] | None:
        return self._transient.get(self._key(namespace, text))

    def set_many_transient(
        self, namespace: str, rows: dict[str, list[float]]
    ) -> None:
        for text, vector in rows.items():
            self._transient[self._key(namespace, text)] = vector

    def clear(self) -> None:
        self._transient = {}
        for path in (
            self.path,
            Path(str(self.path) + "-wal"),
            Path(str(self.path) + "-shm"),
        ):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def close(self) -> None:
        """Kept for callers that share the JSON and SQLite cache interfaces."""

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "cache_key TEXT PRIMARY KEY, vector BLOB NOT NULL)"
        )
        return connection

    @staticmethod
    def _key(namespace: str, text: str) -> str:
        stable = f"{namespace}\n{text}".encode()
        return blake2b(stable, digest_size=16).hexdigest()
