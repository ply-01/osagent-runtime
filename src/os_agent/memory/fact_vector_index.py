"""Optional semantic retrieval over structured Core facts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from os_agent.memory.embedding_cache import EmbeddingCache, SqliteEmbeddingCache
from os_agent.memory.embedding_provider import (
    EmbeddingError,
    EmbeddingProvider,
    create_embedding_provider,
)
from os_agent.memory.mem0_schemas import FactCard


@dataclass(frozen=True, slots=True)
class FactSemanticHit:
    fact: FactCard
    similarity: float


class FactSemanticIndex:
    """Search structured facts with a configured embedding provider."""

    def __init__(
        self,
        *,
        provider: EmbeddingProvider | None = None,
        cache: EmbeddingCache | SqliteEmbeddingCache | None = None,
        batch_size: int = 128,
        vector_store: Any | None = None,
    ) -> None:
        self.provider = provider or create_embedding_provider()
        self.cache = cache or EmbeddingCache()
        self.batch_size = max(batch_size, 1)
        self.vector_store = vector_store
        self._resident_texts: tuple[str, ...] = ()
        self._resident_keys: tuple[tuple[str, str], ...] = ()
        self._resident_key_to_row: dict[tuple[str, str], int] = {}
        self._resident_vectors: list[list[float]] = []
        self._resident_matrix: Any = None

    @property
    def enabled(self) -> bool:
        return self.provider.enabled

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        """Embed multiple retrieval routes in one provider call."""
        if not self.enabled or not queries:
            return []
        try:
            return self._embed_with_cache(queries, kind="query")
        except EmbeddingError:
            if getattr(self.provider, "required", False):
                raise
            return []

    def refresh(self, facts: list[FactCard]) -> int:
        """Precompute missing document vectors outside the query path."""
        if not self.enabled:
            self._clear_resident()
            return 0
        if not facts:
            self._clear_resident()
            if self.vector_store is not None:
                self.vector_store.replace([])
            return 0
        texts = [self._fact_text(fact) for fact in facts]
        try:
            vectors = self._embed_with_cache(texts, kind="document")
        except EmbeddingError:
            if getattr(self.provider, "required", False):
                raise
            self._clear_resident()
            return 0
        self._set_resident(facts, texts, vectors)
        if self.vector_store is not None:
            self.vector_store.replace(
                [(self._key_text(fact), vector) for fact, vector in zip(facts, vectors)]
            )
        return len(facts)

    def search(self, query: str, facts: list[FactCard], limit: int, query_embedding: list[float] | None = None) -> list[FactSemanticHit]:
        if not self.enabled or not query.strip() or limit <= 0 or not facts:
            return []
        fact_keys = tuple(self._fact_key(fact) for fact in facts)
        full_resident_index = fact_keys == self._resident_keys
        for fact in facts:
            fact.semantic_score = 0.0
        try:
            # Use caller-provided embedding if available, otherwise compute locally
            if query_embedding is not None and len(query_embedding) > 0:
                query_vectors = [query_embedding]
            else:
                query_vectors = self._embed_with_cache([query], kind="query")

            fact_matrix = None
            if fact_keys == self._resident_keys:
                fact_vectors = self._resident_vectors
                fact_matrix = self._resident_matrix
            elif self._resident_matrix is not None and all(
                key in self._resident_key_to_row for key in fact_keys
            ):
                import numpy as np

                rows = np.fromiter(
                    (self._resident_key_to_row[key] for key in fact_keys),
                    dtype=np.int64,
                    count=len(fact_keys),
                )
                fact_vectors = []
                fact_matrix = self._resident_matrix[rows]
            else:
                fact_texts = [self._fact_text(fact) for fact in facts]
                fact_vectors = self._embed_with_cache(fact_texts, kind="document")
                fact_matrix = self._normalized_matrix(fact_vectors)
                if not self._resident_keys:
                    self._set_resident(facts, fact_texts, fact_vectors)
                    fact_matrix = self._resident_matrix
        except EmbeddingError:
            if getattr(self.provider, "required", False):
                raise
            return []
        if not query_vectors or not query_vectors[0]:
            return []
        if self.vector_store is not None:
            by_key = {self._key_text(fact): fact for fact in facts}
            hits: list[FactSemanticHit] = []
            for key, score in self.vector_store.search(
                query_vectors[0],
                limit,
                # The vector collection already contains exactly the resident
                # snapshot. Avoid serializing a 10k-key SQL expression when no
                # metadata filter narrowed the candidate set.
                allowed_keys=None if full_resident_index else by_key,
            ):
                fact = by_key.get(key)
                if fact is None:
                    continue
                fact.semantic_score = score
                hits.append(FactSemanticHit(fact=fact, similarity=score))
            return hits
        ranked = self._top_similarities(
            query_vectors[0],
            fact_vectors,
            fact_matrix,
            limit,
        )
        hits = [FactSemanticHit(fact=facts[index], similarity=score) for index, score in ranked]
        hits.sort(key=lambda hit: (-hit.similarity, hit.fact.entity, hit.fact.attribute))
        for hit in hits:
            hit.fact.semantic_score = hit.similarity
        return hits

    def search_many(
        self,
        queries: list[str],
        facts: list[FactCard],
        limit: int,
        *,
        query_embeddings: list[list[float] | None] | None = None,
    ) -> list[list[FactSemanticHit]]:
        """Search several independent routes with one matrix multiplication."""
        if not queries:
            return []
        if (
            not self.enabled
            or limit <= 0
            or not facts
            or any(not query.strip() for query in queries)
        ):
            return [[] for _ in queries]
        supplied = list(query_embeddings or [None] * len(queries))
        if len(supplied) != len(queries):
            raise ValueError("query_embeddings must match queries")
        missing = [index for index, vector in enumerate(supplied) if not vector]
        try:
            if missing:
                embedded = self._embed_with_cache(
                    [queries[index] for index in missing], kind="query"
                )
                for index, vector in zip(missing, embedded):
                    supplied[index] = vector
        except EmbeddingError:
            if getattr(self.provider, "required", False):
                raise
            return [[] for _ in queries]
        if any(not vector for vector in supplied):
            return [[] for _ in queries]

        # The native engine currently exposes one-query ABI calls. It still
        # benefits from batched embedding, while the local matrix path below
        # avoids N separate 10k x dimension BLAS invocations.
        if self.vector_store is not None:
            merged_scores: dict[str, float] = {}
            results: list[list[FactSemanticHit]] = []
            for query, vector in zip(queries, supplied):
                hits = self.search(query, facts, limit, query_embedding=vector)
                results.append(hits)
                for hit in hits:
                    merged_scores[hit.fact.fact_id] = max(
                        merged_scores.get(hit.fact.fact_id, 0.0), hit.similarity
                    )
            for fact in facts:
                fact.semantic_score = merged_scores.get(fact.fact_id, 0.0)
            return results

        fact_keys = tuple(self._fact_key(fact) for fact in facts)
        try:
            import numpy as np

            if fact_keys == self._resident_keys:
                matrix = self._resident_matrix
            elif self._resident_matrix is not None and all(
                key in self._resident_key_to_row for key in fact_keys
            ):
                rows = np.fromiter(
                    (self._resident_key_to_row[key] for key in fact_keys),
                    dtype=np.int64,
                    count=len(fact_keys),
                )
                matrix = self._resident_matrix[rows]
            else:
                texts = [self._fact_text(fact) for fact in facts]
                vectors = self._embed_with_cache(texts, kind="document")
                matrix = self._normalized_matrix(vectors)
            vectors_array = np.asarray(supplied, dtype=np.float32)
            norms = np.linalg.norm(vectors_array, axis=1, keepdims=True)
            vectors_array = vectors_array / np.maximum(norms, 1e-12)
            if matrix is None or matrix.shape[1] != vectors_array.shape[1]:
                raise ValueError("incompatible embedding dimensions")
            score_columns = matrix @ vectors_array.T
        except (ImportError, TypeError, ValueError, AttributeError, EmbeddingError):
            return [
                self.search(query, facts, limit, query_embedding=vector)
                for query, vector in zip(queries, supplied)
            ]

        for fact in facts:
            fact.semantic_score = 0.0
        results = []
        for column in range(len(queries)):
            ranked = self._top_scores(score_columns[:, column], limit)
            hits = [
                FactSemanticHit(fact=facts[index], similarity=score)
                for index, score in ranked
            ]
            for hit in hits:
                hit.fact.semantic_score = max(hit.fact.semantic_score, hit.similarity)
            results.append(hits)
        return results

    def clear_cache(self) -> None:
        self._clear_resident()
        self.cache.clear()

    def _clear_resident(self) -> None:
        self._resident_texts = ()
        self._resident_keys = ()
        self._resident_key_to_row = {}
        self._resident_vectors = []
        self._resident_matrix = None

    def _set_resident(
        self,
        facts: list[FactCard],
        texts: list[str],
        vectors: list[list[float]],
    ) -> None:
        self._resident_texts = tuple(texts)
        self._resident_keys = tuple(self._fact_key(fact) for fact in facts)
        self._resident_key_to_row = {
            key: index for index, key in enumerate(self._resident_keys)
        }
        self._resident_vectors = vectors
        self._resident_matrix = self._normalized_matrix(vectors)

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
            if kind == "query":
                embed = getattr(self.provider, "embed_queries", None)
            else:
                embed = getattr(self.provider, "embed_documents", None)
            if embed is None:
                embed = self.provider.embed_texts
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
    def _fact_text(fact: FactCard) -> str:
        # Evidence is the closest representation to the source language. Keep
        # the established weighting so performance work does not silently
        # change retrieval quality or the embedding-cache identity.
        if fact.evidence and fact.evidence.strip():
            evidence = fact.evidence.strip()
            if len(evidence) < 50:
                return f"{fact.entity} {fact.attribute}: {evidence}"
            return f"{evidence} {evidence[:100]}"
        if fact.value:
            value_text = str(fact.value).strip()
            return f"{fact.entity} {fact.attribute}: {value_text}"
        return f"{fact.entity} {fact.attribute} {fact.value}"

    @staticmethod
    def _fact_key(fact: FactCard) -> tuple[str, str]:
        return fact.fact_id, fact.updated_at

    @classmethod
    def _key_text(cls, fact: FactCard) -> str:
        fact_id, updated_at = cls._fact_key(fact)
        return f"fact:{fact_id}:{updated_at}"


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

    @staticmethod
    def _normalized_matrix(vectors: list[list[float]]) -> Any:
        if not vectors:
            return None
        try:
            import numpy as np

            matrix = np.asarray(vectors, dtype=np.float32)
            if matrix.ndim != 2:
                return None
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            return np.ascontiguousarray(matrix / np.maximum(norms, 1e-12))
        except (ImportError, TypeError, ValueError):
            return None

    @classmethod
    def _top_similarities(
        cls,
        query: list[float],
        documents: list[list[float]],
        matrix: Any,
        limit: int,
    ) -> list[tuple[int, float]]:
        if limit <= 0:
            return []
        try:
            import numpy as np

            resolved = matrix if matrix is not None else cls._normalized_matrix(documents)
            query_array = np.asarray(query, dtype=np.float32)
            if (
                resolved is None
                or resolved.ndim != 2
                or resolved.shape[1] != query_array.shape[0]
            ):
                raise ValueError("incompatible embedding dimensions")
            query_norm = float(np.linalg.norm(query_array))
            if query_norm <= 1e-12:
                return []
            scores = resolved @ (query_array / query_norm)
            positive = np.flatnonzero(scores > 0)
            if positive.size == 0:
                return []
            count = min(limit, int(positive.size))
            if positive.size > count:
                local = np.argpartition(scores[positive], -count)[-count:]
                selected = positive[local]
            else:
                selected = positive
            return sorted(
                ((int(index), float(scores[index])) for index in selected),
                key=lambda item: (-item[1], item[0]),
            )
        except (ImportError, TypeError, ValueError, AttributeError):
            similarities = cls._similarities(query, documents)
            ranked = [
                (index, score)
                for index, score in enumerate(similarities)
                if score > 0
            ]
            ranked.sort(key=lambda item: (-item[1], item[0]))
            return ranked[:limit]

    @staticmethod
    def _top_scores(scores: Any, limit: int) -> list[tuple[int, float]]:
        """Return deterministic top positive scores from one NumPy column."""
        import numpy as np

        positive = np.flatnonzero(scores > 0)
        if positive.size == 0 or limit <= 0:
            return []
        count = min(limit, int(positive.size))
        if positive.size > count:
            local = np.argpartition(scores[positive], -count)[-count:]
            selected = positive[local]
        else:
            selected = positive
        return sorted(
            ((int(index), float(scores[index])) for index in selected),
            key=lambda item: (-item[1], item[0]),
        )
