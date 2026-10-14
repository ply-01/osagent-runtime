"""Local vector retrieval for Markdown memory sections."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from hashlib import blake2b

from os_agent.memory.embedding_cache import EmbeddingCache
from os_agent.memory.embedding_provider import EmbeddingError, OpenAICompatibleEmbeddingProvider
from os_agent.memory.schemas import MemoryPageItem


SparseVector = dict[int, float]
DenseVector = list[float]


@dataclass(slots=True)
class VectorSearchHit:
    """A memory item selected by vector similarity."""

    item: MemoryPageItem
    similarity: float
    method: str = "local_vector"


class HashingVectorizer:
    """Deterministic bag-of-token hashing vectorizer.

    This is intentionally dependency-free. It gives the memory layer a stable
    vector-search interface now, while keeping the door open for a later
    embedding provider.
    """

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def embed(self, text: str) -> SparseVector:
        vector: SparseVector = {}
        for token in self.tokenize(text):
            index = self._hash_token(token)
            vector[index] = vector.get(index, 0.0) + 1.0
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm == 0:
            return {}
        return {index: value / norm for index, value in vector.items()}

    def tokenize(self, text: str) -> list[str]:
        lower = text.lower()
        tokens = re.findall(r"[a-z0-9_./\\-]{2,}", lower)
        for sequence in re.findall(r"[\u4e00-\u9fff]+", text):
            if len(sequence) >= 2:
                tokens.extend(
                    sequence[index : index + 2]
                    for index in range(0, len(sequence) - 1)
                )
            if 2 <= len(sequence) <= 12:
                tokens.append(sequence)
        return tokens

    def cosine(self, left: SparseVector, right: SparseVector) -> float:
        if not left or not right:
            return 0.0
        if len(left) > len(right):
            left, right = right, left
        return sum(value * right.get(index, 0.0) for index, value in left.items())

    def _hash_token(self, token: str) -> int:
        digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dimensions


class MemoryVectorIndex:
    """Search Markdown memory sections by semantic or local vector similarity."""

    def __init__(
        self,
        vectorizer: HashingVectorizer | None = None,
        embedding_provider: OpenAICompatibleEmbeddingProvider | None = None,
        embedding_cache: EmbeddingCache | None = None,
        allow_local_fallback: bool = True,
    ) -> None:
        self.vectorizer = vectorizer or HashingVectorizer()
        self.embedding_provider = embedding_provider or OpenAICompatibleEmbeddingProvider()
        self.embedding_cache = embedding_cache or EmbeddingCache()
        self.allow_local_fallback = allow_local_fallback

    def search(
        self,
        query: str,
        items: list[MemoryPageItem],
        limit: int = 5,
    ) -> list[VectorSearchHit]:
        if self.embedding_provider.enabled:
            try:
                return self._search_semantic(query, items, limit)
            except EmbeddingError:
                if not self.allow_local_fallback:
                    raise
        return self._search_local(query, items, limit)

    def _search_semantic(
        self,
        query: str,
        items: list[MemoryPageItem],
        limit: int,
    ) -> list[VectorSearchHit]:
        texts = [query] + [f"{item.title}\n{item.content}" for item in items]
        vectors = self._embed_with_cache(texts)
        if not vectors:
            return []
        query_vector = vectors[0]
        hits: list[VectorSearchHit] = []
        for item, item_vector in zip(items, vectors[1:]):
            similarity = self._dense_cosine(query_vector, item_vector)
            if similarity <= 0:
                continue
            hits.append(VectorSearchHit(item=item, similarity=similarity, method="semantic_embedding"))
        hits.sort(key=lambda hit: (-hit.similarity, hit.item.source, hit.item.title))
        return hits[: max(limit, 0)]

    def _search_local(
        self,
        query: str,
        items: list[MemoryPageItem],
        limit: int,
    ) -> list[VectorSearchHit]:
        query_vector = self.vectorizer.embed(query)
        if not query_vector:
            return []
        hits: list[VectorSearchHit] = []
        for item in items:
            item_vector = self.vectorizer.embed(f"{item.title}\n{item.content}")
            similarity = self.vectorizer.cosine(query_vector, item_vector)
            if similarity <= 0:
                continue
            hits.append(VectorSearchHit(item=item, similarity=similarity, method="local_vector"))
        hits.sort(key=lambda hit: (-hit.similarity, hit.item.source, hit.item.title))
        return hits[: max(limit, 0)]

    def _embed_with_cache(self, texts: list[str]) -> list[DenseVector]:
        namespace = self.embedding_provider.cache_namespace
        cached: list[DenseVector | None] = [
            self.embedding_cache.get(namespace, text) for text in texts
        ]
        missing_texts = [
            text for text, vector in zip(texts, cached) if vector is None
        ]
        if missing_texts:
            missing_vectors = self.embedding_provider.embed_texts(missing_texts)
            if len(missing_vectors) != len(missing_texts):
                raise EmbeddingError("Embedding provider returned an unexpected vector count")
            vector_iter = iter(missing_vectors)
            for index, text in enumerate(texts):
                if cached[index] is not None:
                    continue
                vector = next(vector_iter)
                self.embedding_cache.set(namespace, text, vector)
                cached[index] = vector
        return [vector or [] for vector in cached]

    def _dense_cosine(self, left: DenseVector, right: DenseVector) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)
