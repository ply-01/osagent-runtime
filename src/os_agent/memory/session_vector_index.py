"""Semantic retrieval over deterministic session summaries."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from os_agent.memory.embedding_cache import EmbeddingCache, SqliteEmbeddingCache
from os_agent.memory.embedding_provider import (
    EmbeddingError,
    EmbeddingProvider,
    create_embedding_provider,
)
from os_agent.memory.mem0_schemas import SessionDocument


@dataclass(frozen=True, slots=True)
class SessionSemanticHit:
    document: SessionDocument
    similarity: float


class SessionSemanticIndex:
    """Search session summaries with the same provider used for Core facts."""

    def __init__(
        self,
        *,
        provider: EmbeddingProvider | None = None,
        cache: EmbeddingCache | SqliteEmbeddingCache | None = None,
        batch_size: int = 128,
        vector_store: object | None = None,
        vector_store_root: Path | None = None,
    ) -> None:
        self.provider = provider or create_embedding_provider()
        self.cache = cache or EmbeddingCache()
        self.batch_size = max(batch_size, 1)
        self.vector_store = vector_store
        self.vector_store_root = Path(vector_store_root) if vector_store_root else None
        self._resident_texts: tuple[str, ...] = ()
        self._resident_keys: tuple[str, ...] = ()
        self._resident_vectors: list[list[float]] = []

    @property
    def enabled(self) -> bool:
        return self.provider.enabled

    def refresh(self, documents: list[SessionDocument]) -> int:
        """Precompute missing session vectors outside the query path."""
        if not self.enabled:
            self._clear_resident()
            return 0
        if not documents:
            self._clear_resident()
            if self.vector_store is not None:
                self.vector_store.replace([])
            return 0
        texts = [document.embedding_text or document.text for document in documents]
        try:
            vectors = self._embed_with_cache(texts, kind="document")
        except EmbeddingError:
            if getattr(self.provider, "required", False):
                raise
            self._clear_resident()
            return 0
        self._resident_texts = tuple(texts)
        self._resident_keys = tuple(self._key_text(document) for document in documents)
        self._resident_vectors = vectors
        self._ensure_kylin_store(vectors)
        if self.vector_store is not None:
            self.vector_store.replace(
                [(self._key_text(document), vector) for document, vector in zip(documents, vectors)]
            )
        return len(documents)

    def search(
        self,
        query: str,
        documents: list[SessionDocument],
        limit: int,
        query_embedding: list[float] | None = None,
    ) -> list[SessionSemanticHit]:
        if not self.enabled or not query.strip() or limit <= 0 or not documents:
            return []
        try:
            # Use caller-provided embedding if available, otherwise compute locally
            if query_embedding is not None and len(query_embedding) > 0:
                query_vector = query_embedding
            else:
                query_vector = self._embed_with_cache([query], kind="query")[0]

            document_texts = [
                document.embedding_text or document.text for document in documents
            ]
            document_keys = tuple(self._key_text(document) for document in documents)
            full_resident_index = (
                tuple(document_texts) == self._resident_texts
                and document_keys == self._resident_keys
            )
            if full_resident_index:
                document_vectors = self._resident_vectors
            else:
                document_vectors = self._embed_with_cache(
                    document_texts, kind="document"
                )
                self._resident_texts = tuple(document_texts)
                self._resident_keys = document_keys
                self._resident_vectors = document_vectors
        except (EmbeddingError, IndexError) as exc:
            if isinstance(exc, EmbeddingError) and getattr(
                self.provider, "required", False
            ):
                raise
            return []
        self._ensure_kylin_store(document_vectors)
        if self.vector_store is not None:
            by_key = {self._key_text(document): document for document in documents}
            return [
                SessionSemanticHit(document=by_key[key], similarity=score)
                for key, score in self.vector_store.search(
                    query_vector,
                    limit,
                    allowed_keys=None if full_resident_index else by_key,
                )
                if key in by_key
            ]
        similarities = self._similarities(query_vector, document_vectors)
        hits = [
            SessionSemanticHit(document=document, similarity=similarity)
            for document, similarity in zip(documents, similarities)
        ]
        hits = [hit for hit in hits if hit.similarity > 0]
        hits.sort(key=lambda hit: (-hit.similarity, hit.document.document_id))
        return hits[:limit]

    def clear_cache(self) -> None:
        self._clear_resident()
        self.cache.clear()

    def _clear_resident(self) -> None:
        self._resident_texts = ()
        self._resident_keys = ()
        self._resident_vectors = []

    @staticmethod
    def _key_text(document: SessionDocument) -> str:
        return f"session:{document.document_id}"

    def _ensure_kylin_store(self, vectors: list[list[float]]) -> None:
        if self.vector_store is not None or self.vector_store_root is None:
            return
        if not getattr(self.provider, "is_kylin_system", False) or not vectors:
            return
        from os_agent.memory.kylin_vector_engine import KylinVectorEngine

        self.vector_store = KylinVectorEngine.from_environment(
            self.vector_store_root, kind="sessions", dimension=len(vectors[0])
        )

    def _embed_with_cache(self, texts: list[str], *, kind: str) -> list[list[float]]:
        namespace = f"{self.provider.cache_namespace}|{kind}"
        if kind == "query":
            vectors = [self.cache.get_transient(namespace, text) for text in texts]
        else:
            vectors = self.cache.get_many(namespace, texts)
        missing_indices = [index for index, vector in enumerate(vectors) if vector is None]
        for start in range(0, len(missing_indices), self.batch_size):
            indices = missing_indices[start : start + self.batch_size]
            missing = [texts[index] for index in indices]
            embed = (
                self.provider.embed_queries
                if kind == "query"
                else self.provider.embed_documents
            )
            embedded = embed(missing)
            if len(embedded) != len(indices):
                raise EmbeddingError("Embedding provider returned an unexpected vector count")
            rows = {texts[index]: vector for index, vector in zip(indices, embedded)}
            if kind == "query":
                self.cache.set_many_transient(namespace, rows)
            else:
                self.cache.set_many(namespace, rows)
            for index, vector in zip(indices, embedded):
                vectors[index] = vector
        return [vector or [] for vector in vectors]

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        numerator = sum(a * b for a, b in zip(left, right))
        denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
            sum(value * value for value in right)
        )
        return numerator / denominator if denominator else 0.0

    @classmethod
    def _similarities(
        cls, query: list[float], documents: list[list[float]]
    ) -> list[float]:
        if not query or not documents:
            return []
        try:
            import numpy as np

            matrix = np.asarray(documents, dtype=np.float32)
            query_array = np.asarray(query, dtype=np.float32)
            if matrix.ndim != 2 or matrix.shape[1] != query_array.shape[0]:
                return [cls._cosine(query, vector) for vector in documents]
            matrix_norms = np.linalg.norm(matrix, axis=1)
            query_norm = float(np.linalg.norm(query_array))
            denominator = np.maximum(matrix_norms * query_norm, 1e-12)
            return ((matrix @ query_array) / denominator).astype(float).tolist()
        except (ImportError, TypeError, ValueError):
            return [cls._cosine(query, vector) for vector in documents]
