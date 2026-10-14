"""Deterministic context compaction before an LLM call."""

from __future__ import annotations

import re
from dataclasses import dataclass

from os_agent.memory.retrieval_text import tokenize_search_text


@dataclass(frozen=True, slots=True)
class _Chunk:
    index: int
    text: str
    score: float


class ContextCompressor:
    """Keep the most useful context under a character budget.

    This is intentionally extractive: source memory and event logs remain
    unchanged, and no model call is needed just to fit a prompt budget.
    """

    _HEADING = re.compile(r"^(?:#{1,6}\s|<[^/!][^>]*>$)")
    _STRUCTURAL = re.compile(
        r"^(?:#|Available tools:|Available skills:|<mem0_recall|"
        r"<markdown_memory_page|<item\b|Session:|Task:)",
        re.IGNORECASE,
    )

    def compress(self, text: str, *, query: str = "", max_chars: int = 12000) -> str:
        clean = text.strip()
        budget = max(int(max_chars), 256)
        if len(clean) <= budget:
            return clean

        query_terms = set(tokenize_search_text(query))
        chunks = self._chunks(clean, query_terms)
        if not chunks:
            return clean[:budget]

        # Always retain the opening structural context, then spend the rest on
        # the highest-scoring chunks. Original order is restored for coherence.
        selected: list[_Chunk] = []
        used = 0
        for chunk in chunks:
            if chunk.index < 2 and self._fits(chunk.text, used, budget):
                selected.append(chunk)
                used += len(chunk.text) + 1

        for chunk in sorted(chunks, key=lambda item: (-item.score, item.index)):
            if chunk in selected:
                continue
            if self._fits(chunk.text, used, budget):
                selected.append(chunk)
                used += len(chunk.text) + 1

        selected.sort(key=lambda item: item.index)
        body = "\n".join(chunk.text for chunk in selected)
        marker = "\n...[context compressed; source memory preserved]"
        if len(body) + len(marker) <= budget:
            return body + marker
        return (body[: max(budget - len(marker), 0)] + marker).strip()

    @classmethod
    def _chunks(cls, text: str, query_terms: set[str]) -> list[_Chunk]:
        raw_blocks = re.split(r"\n\s*\n", text)
        blocks: list[str] = []
        for block in raw_blocks:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            blocks.extend(lines or [block.strip()])

        chunks: list[_Chunk] = []
        for index, block in enumerate(blocks):
            terms = set(tokenize_search_text(block))
            overlap = len(terms & query_terms) / max(1, len(query_terms))
            structural = 0.35 if cls._STRUCTURAL.search(block) else 0.0
            heading = 0.2 if cls._HEADING.search(block) else 0.0
            chunks.append(_Chunk(index, block, overlap + structural + heading))
        return chunks

    @staticmethod
    def _fits(text: str, used: int, budget: int) -> bool:
        return used + len(text) + 1 <= budget
