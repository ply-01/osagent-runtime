"""Markdown-backed memory store for stage one."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING

from os_agent.events.schemas import MemoryCandidate, utc_now_iso
from os_agent.memory.schemas import MemoryContext, MemoryPage, MemoryPageItem, RetrievedMemory
from os_agent.memory.policy import MemoryDecision
from os_agent.memory.vector_index import MemoryVectorIndex

if TYPE_CHECKING:
    from os_agent.memory.mem0_schemas import FactCard


class MarkdownMemoryStore:
    """Store auditable first-stage memories as Markdown files."""

    # tier thresholds based on time-decayed weighted score
    DECAY_RATE = 0.85          # per day: score *= 0.85 each day since last access
    L1_HOT_THRESHOLD = 3.0     # weighted_score >= 3.0 → hot (≈3 recent accesses)
    L2_WARM_THRESHOLD = 0.5    # weighted_score >= 0.5 → warm (≈1 recent access)
                                # weighted_score < 0.5  → cold

    CORE_FILES = {
        "profile": "profile.md",
        "project_knowledge": "project-knowledge.md",
        "workflows": "workflows.md",
        "tool_history": "tool-history.md",
        "safety": "safety.md",
    }
    MEMORY_LAYERS = ("hot", "warm", "cold")
    PAGE_STRATEGIES = ("layered", "priority")
    HOT_ACCESS_THRESHOLD = 5
    WARM_ACCESS_THRESHOLD = 2
    LAYER_PAGE_QUOTAS = {"hot": 0.50, "warm": 0.30, "cold": 0.20}

    def __init__(self, root: Path | str = "memory") -> None:
        self.root = Path(root)
        self.pending = self.root / "pending-confirmation"
        self.archive = self.root / "archive"
        self.paging = self.root / ".paging"
        self.paging_index = self.paging / "index.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.pending.mkdir(parents=True, exist_ok=True)
        self.archive.mkdir(parents=True, exist_ok=True)
        self.paging.mkdir(parents=True, exist_ok=True)
        self.vector_index = MemoryVectorIndex()
        self._ensure_core_files()

    # ------------------------------------------------------------------
    # tier index for access-frequency-based layered retrieval
    # ------------------------------------------------------------------
    @property
    def _tier_index_path(self) -> Path:
        return self.root / ".tier_index.json"

    def _load_tier_index(self) -> dict:
        if not self._tier_index_path.exists():
            return {}
        try:
            return json.loads(self._tier_index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_tier_index(self, index: dict) -> None:
        self._tier_index_path.write_text(
            json.dumps(index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _section_key(source: str, title: str) -> str:
        return f"{source}::{title}"

    def _tier_for(self, key: str, index: dict) -> int:
        """Return tier level: 0=L1(hot), 1=L2(warm), 2=L3(cold)."""
        score = round(self._effective_score(index.get(key, {})), 2)
        if score >= self.L1_HOT_THRESHOLD:
            return 0
        if score >= self.L2_WARM_THRESHOLD:
            return 1
        return 2

    def _bump_access(self, key: str, index: dict) -> None:
        """Decay old score, then add 1.0 for this access."""
        entry = index.setdefault(key, {"weighted_score": 0.0, "access_count": 0, "last_accessed": ""})
        decayed = self._decayed_score(entry)
        entry["weighted_score"] = round(decayed + 1.0, 4)
        entry["access_count"] = entry.get("access_count", 0) + 1
        entry["last_accessed"] = utc_now_iso()

    def _days_since(self, iso_timestamp: str) -> float:
        """Return days elapsed since the given ISO timestamp."""
        if not iso_timestamp:
            return 0.0
        try:
            last_dt = datetime.fromisoformat(iso_timestamp)
            now = datetime.now(timezone.utc)
            return max((now - last_dt).total_seconds() / 86400.0, 0.0)
        except (ValueError, TypeError):
            return 0.0

    def _decayed_score(self, entry: dict) -> float:
        """Apply time decay to an entry's weighted_score."""
        raw = entry.get("weighted_score", 0.0)
        days = self._days_since(entry.get("last_accessed", ""))
        if days <= 0:
            return raw
        return raw * (self.DECAY_RATE ** days)

    def _effective_score(self, entry: dict) -> float:
        """Current effective score after time decay."""
        return self._decayed_score(entry)

    def _prune_tier_index(self, index: dict) -> None:
        """Remove entries for sections that no longer exist in any file."""
        valid_keys: set[str] = set()
        for filename in self.CORE_FILES.values():
            path = self.root / filename
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for title, _ in self._iter_sections(text, filename):
                if any(line.strip().startswith("## ") for line in text.splitlines()):
                    valid_keys.add(self._section_key(filename, title))
        if self.pending.exists():
            for pf in self.pending.glob("*.md"):
                text = pf.read_text(encoding="utf-8", errors="replace")
                source = f"pending-confirmation/{pf.name}"
                for title, _ in self._iter_sections(text, source):
                    if any(line.strip().startswith("## ") for line in text.splitlines()):
                        valid_keys.add(self._section_key(source, title))
        stale = [k for k in index if k not in valid_keys]
        for k in stale:
            del index[k]

    def tier_info(self, source: str, title: str) -> dict:
        """Return {tier, weighted_score, access_count, last_accessed} with decay."""
        index = self._load_tier_index()
        key = self._section_key(source, title)
        entry = index.get(key, {})
        effective = round(self._effective_score(entry), 2)
        count = entry.get("access_count", 0)
        if effective >= self.L1_HOT_THRESHOLD:
            tier = "L1-hot"
        elif effective >= self.L2_WARM_THRESHOLD:
            tier = "L2-warm"
        else:
            tier = "L3-cold"
        return {
            "tier": tier,
            "weighted_score": round(effective, 2),
            "access_count": count,
            "last_accessed": entry.get("last_accessed", ""),
        }

    def load_context(self) -> MemoryContext:
        values = {}
        for field_name, filename in self.CORE_FILES.items():
            values[field_name] = (self.root / filename).read_text(
                encoding="utf-8",
                errors="replace",
            )
        return MemoryContext(**values)

    def page_memory(
        self,
        page: int = 1,
        page_size: int = 10,
        source: str | None = None,
        layer: str | None = None,
        query: str | None = None,
        strategy: str = "layered",
    ) -> MemoryPage:
        """Return a stable page of Markdown memory sections."""
        normalized_page = max(int(page), 1)
        normalized_page_size = max(int(page_size), 1)
        normalized_source = self._normalize_source(source)
        normalized_layer = self._normalize_layer(layer)
        normalized_strategy = self._normalize_page_strategy(strategy)
        items = self._memory_page_items(normalized_source)
        self._score_page_items(items, query)
        if normalized_layer:
            items = [item for item in items if item.layer == normalized_layer]
        items = self._sort_memory_page_items(
            items,
            strategy=normalized_strategy,
            page_size=normalized_page_size,
        )
        total_items = len(items)
        total_pages = ceil(total_items / normalized_page_size) if total_items else 0
        start = (normalized_page - 1) * normalized_page_size
        end = start + normalized_page_size
        return MemoryPage(
            items=items[start:end],
            page=normalized_page,
            page_size=normalized_page_size,
            total_items=total_items,
            total_pages=total_pages,
            source=normalized_source,
            layer=normalized_layer,
            query=query.strip() if query else None,
            strategy=normalized_strategy,
        )

    def memory_layer_summary(self, source: str | None = None) -> dict[str, int]:
        """Return stable memory section counts grouped by access layer."""
        normalized_source = self._normalize_source(source)
        summary = {layer: 0 for layer in self.MEMORY_LAYERS}
        for item in self._memory_page_items(normalized_source):
            summary[item.layer] = summary.get(item.layer, 0) + 1
        summary["total"] = sum(summary.values())
        return summary

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        source: str | None = None,
        record_access: bool = True,
    ) -> list[RetrievedMemory]:
        """Return tiered keyword-ranked memory excerpts."""
        if limit <= 0:
            return []
        terms = self._tokenize(query)
        if not terms:
            return []
        matches = self._keyword_matches(terms, source=source)
        results = self._rank_tiered_results(matches, limit)
        if record_access:
            self._record_retrieved_access(results)
        return results

    def _keyword_matches(
        self,
        terms: list[str],
        source: str | None = None,
    ) -> list[RetrievedMemory]:
        matches: list[RetrievedMemory] = []
        normalized_source = self._normalize_source(source)
        filenames = [normalized_source] if normalized_source else list(self.CORE_FILES.values())
        for filename in filenames:
            if filename is None:
                continue
            path = self.root / filename
            text = path.read_text(encoding="utf-8", errors="replace")
            for title, content in self._iter_sections(text, filename):
                haystack = f"{title}\n{content}".lower()
                score = sum(haystack.count(term) for term in terms)
                if score <= 0:
                    continue
                matches.append(
                    RetrievedMemory(
                        source=filename,
                        title=title,
                        excerpt=self._excerpt(content, terms),
                        score=score,
                        retrieval_method="keyword",
                    )
                )
        if normalized_source is None and self.pending.exists():
            for pending_file in sorted(self.pending.glob("*.md")):
                text = pending_file.read_text(encoding="utf-8", errors="replace")
                pending_source = f"pending-confirmation/{pending_file.name}"
                for title, content in self._iter_sections(text, pending_source):
                    haystack = f"{title}\n{content}".lower()
                    score = sum(haystack.count(term) for term in terms)
                    if score <= 0:
                        continue
                    matches.append(
                        RetrievedMemory(
                            source=pending_source,
                            title=title,
                            excerpt=self._excerpt(content, terms),
                            score=score,
                            retrieval_method="keyword",
                        )
                    )
        return matches

    def _rank_tiered_results(
        self,
        matches: list[RetrievedMemory],
        limit: int,
    ) -> list[RetrievedMemory]:
        if limit <= 0 or not matches:
            return []
        tier_index = self._load_tier_index()
        buckets: dict[int, list[RetrievedMemory]] = {0: [], 1: [], 2: []}
        for item in matches:
            key = self._section_key(item.source, item.title)
            buckets[self._tier_for(key, tier_index)].append(item)
        for tier_list in buckets.values():
            tier_list.sort(key=lambda item: (-item.score, item.source, item.title))

        results: list[RetrievedMemory] = []
        for tier in (0, 1, 2):
            remaining = limit - len(results)
            if remaining <= 0:
                break
            results.extend(buckets[tier][:remaining])
        return results

    def retrieve_vector(
        self,
        query: str,
        limit: int = 5,
        source: str | None = None,
        record_access: bool = True,
    ) -> list[RetrievedMemory]:
        """Return memory excerpts ranked by local vector similarity."""
        normalized_source = self._normalize_source(source)
        items = self._memory_page_items(normalized_source)
        hits = self.vector_index.search(query, items, limit=limit)
        terms = self._tokenize(query)
        results: list[RetrievedMemory] = []
        for hit in hits:
            results.append(
                RetrievedMemory(
                    source=hit.item.source,
                    title=hit.item.title,
                    excerpt=self._excerpt(hit.item.content, terms),
                    score=max(1, int(round(hit.similarity * 1000))),
                    retrieval_method=hit.method,
                )
            )
        if record_access:
            self._record_retrieved_access(results)
        return results

    def retrieve_hybrid(
        self,
        query: str,
        limit: int = 5,
        source: str | None = None,
        record_access: bool = True,
    ) -> list[RetrievedMemory]:
        """Return memory excerpts using combined keyword and vector scores."""
        keyword_matches = self.retrieve(
            query,
            limit=max(limit * 2, 10),
            source=source,
            record_access=False,
        )
        vector_matches = self.retrieve_vector(
            query,
            limit=max(limit * 2, 10),
            source=source,
            record_access=False,
        )
        if not keyword_matches:
            results = vector_matches[:limit]
            if record_access:
                self._record_retrieved_access(results)
            return results
        if not vector_matches:
            results = keyword_matches[:limit]
            if record_access:
                self._record_retrieved_access(results)
            return results

        max_keyword_score = max(match.score for match in keyword_matches) or 1
        merged: dict[tuple[str, str], dict] = {}
        for match in keyword_matches:
            key = (match.source, match.title)
            merged[key] = {
                "source": match.source,
                "title": match.title,
                "excerpt": match.excerpt,
                "score": match.score / max_keyword_score,
                "methods": {"keyword"},
            }
        for match in vector_matches:
            key = (match.source, match.title)
            normalized_score = match.score / 1000.0
            if key not in merged:
                merged[key] = {
                    "source": match.source,
                    "title": match.title,
                    "excerpt": match.excerpt,
                    "score": 0.0,
                    "methods": set(),
                }
            merged[key]["score"] += normalized_score
            merged[key]["methods"].add(match.retrieval_method)
            if len(match.excerpt) > len(str(merged[key]["excerpt"])):
                merged[key]["excerpt"] = match.excerpt

        ranked = sorted(
            merged.values(),
            key=lambda item: (-float(item["score"]), str(item["source"]), str(item["title"])),
        )
        results = [
            RetrievedMemory(
                source=str(item["source"]),
                title=str(item["title"]),
                excerpt=str(item["excerpt"]),
                score=max(1, int(round(float(item["score"]) * 1000))),
                retrieval_method="+".join(sorted(item["methods"])),
            )
            for item in ranked[:limit]
        ]
        if record_access:
            self._record_retrieved_access(results)
        return results

    def append_stable(
        self,
        candidate: MemoryCandidate,
        decision: MemoryDecision,
    ) -> Path:
        target_name = decision.target or "project-knowledge.md"
        target = self.root / target_name
        if self.has_candidate(candidate):
            return target
        with target.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(self._render_stable(candidate, decision))
        return target

    def upsert_core_fact(self, fact: "FactCard") -> Path:
        """Synchronize one active Core fact into the readable Markdown mirror.

        Core owns the fact and its version.  Markdown only owns this derived
        representation, so an update replaces the previous mirror section
        instead of appending a second value that could become stale.
        """
        memory_type = self._core_memory_type(fact)
        target_name = self._target_for_type(memory_type)
        title = self._core_mirror_title(fact)
        core_key = f"{fact.entity}|{fact.attribute}"
        body = "\n".join(
            [
                "source: core",
                f"core_fact_id: {fact.fact_id}",
                f"core_key: {core_key}",
                f"version: {fact.version}",
                f"confidence: {fact.confidence:.2f}",
                f"importance: {fact.importance:.2f}",
                f"source_session_id: {fact.source_session_id}",
                f"source_event_id: {fact.source_event_id}",
                f"auto_written_at: {fact.updated_at}",
                "",
                f"Evidence: {fact.evidence.strip()}",
                f"Value: {fact.value}",
            ]
        )
        replacement = (title, body)
        target = self.root / target_name
        for filename in self.CORE_FILES.values():
            path = self.root / filename
            text = path.read_text(encoding="utf-8", errors="replace")
            document_title, sections = self._split_document(text)
            updated_sections: list[tuple[str, str]] = []
            inserted = False
            changed = False
            for section_title, section_body in sections:
                metadata = self._section_metadata(section_body)
                same_key = metadata.get("core_key") == core_key
                same_id = metadata.get("core_fact_id") == fact.fact_id
                if same_key or same_id:
                    changed = True
                    if filename == target_name and not inserted:
                        updated_sections.append(replacement)
                        inserted = True
                    continue
                updated_sections.append((section_title, section_body))
            if filename == target_name and not inserted:
                updated_sections.append(replacement)
                changed = True
            if changed:
                path.write_text(
                    self._render_document(document_title, updated_sections),
                    encoding="utf-8",
                    newline="\n",
                )
        return target

    def remove_core_fact(self, fact: "FactCard") -> Path | None:
        """Remove a Core fact's current Markdown mirror after Core deletion."""
        core_key = f"{fact.entity}|{fact.attribute}"
        removed_path: Path | None = None
        for target_name in self.CORE_FILES.values():
            target = self.root / target_name
            text = target.read_text(encoding="utf-8", errors="replace")
            document_title, sections = self._split_document(text)
            kept = []
            removed = False
            for section_title, section_body in sections:
                metadata = self._section_metadata(section_body)
                if (
                    metadata.get("core_key") == core_key
                    or metadata.get("core_fact_id") == fact.fact_id
                ):
                    removed = True
                    continue
                kept.append((section_title, section_body))
            if removed:
                target.write_text(
                    self._render_document(document_title, kept),
                    encoding="utf-8",
                    newline="\n",
                )
                removed_path = removed_path or target
        return removed_path

    def append_pending_confirmation(
        self,
        candidate: MemoryCandidate,
        decision: MemoryDecision,
        session_id: str,
    ) -> Path:
        if self.has_candidate(candidate):
            return self.pending / f"{candidate.created_at[:10]}-{session_id}.md"
        filename = f"{candidate.created_at[:10]}-{session_id}.md"
        path = self.pending / filename
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(self._render_pending(candidate, decision))
        return path

    def list_pending_confirmations(self) -> list[Path]:
        return sorted(self.pending.glob("*.md"))

    def compact_stable_files(self, max_sections_per_file: int = 80) -> dict[str, int]:
        removed_duplicates = 0
        archived_sections = 0
        compacted_files = 0
        for filename in self.CORE_FILES.values():
            path = self.root / filename
            text = path.read_text(encoding="utf-8", errors="replace")
            title, sections = self._split_document(text)
            seen: set[str] = set()
            kept: list[tuple[str, str]] = []
            file_removed_duplicates = 0
            for section_title, body in sections:
                fingerprint = self._section_fingerprint(section_title, body)
                if fingerprint in seen:
                    file_removed_duplicates += 1
                    continue
                seen.add(fingerprint)
                kept.append((section_title, body))
            removed_duplicates += file_removed_duplicates
            overflow: list[tuple[str, str]] = []
            if len(kept) > max_sections_per_file:
                overflow = kept[: len(kept) - max_sections_per_file]
                kept = kept[len(kept) - max_sections_per_file :]
                archived_sections += len(overflow)
                self._archive_sections(filename, overflow)
            if file_removed_duplicates or overflow:
                path.write_text(
                    self._render_document(title, kept),
                    encoding="utf-8",
                    newline="\n",
                )
                compacted_files += 1
        # prune tier index of archived/removed sections
        index = self._load_tier_index()
        self._prune_tier_index(index)
        self._save_tier_index(index)

        return {
            "compacted_files": compacted_files,
            "removed_duplicates": removed_duplicates,
            "archived_sections": archived_sections,
        }

    def has_candidate(self, candidate: MemoryCandidate) -> bool:
        fingerprint = self._candidate_fingerprint(candidate)
        for filename in self.CORE_FILES.values():
            text = (self.root / filename).read_text(encoding="utf-8", errors="replace")
            if candidate.title in text and candidate.content.strip() in text:
                return True
        for path in self.pending.glob("*.md"):
            text = path.read_text(encoding="utf-8", errors="replace")
            for block in self._candidate_blocks(text):
                if fingerprint == self._candidate_block_fingerprint(block):
                    return True
        return False

    def has_conflict(self, candidate: MemoryCandidate) -> bool:
        target = self.root / self._target_for_type(candidate.memory_type)
        text = target.read_text(encoding="utf-8", errors="replace")
        sections = self._iter_sections(text, candidate.memory_type)
        # skip fallback section when file has no real ## entries
        if not any(line.strip().startswith("## ") for line in text.splitlines()):
            return False
        for title, body in sections:
            # same title, different content → conflict
            if title == candidate.title and candidate.content.strip() not in body:
                return True
            # different title, same category (preference/safety) → potential conflict
            if candidate.memory_type in ("preference", "safety"):
                if title != candidate.title and candidate.content.strip() not in body:
                    return True
        return False

    def _ensure_core_files(self) -> None:
        readme = self.root / "README.md"
        if not readme.exists():
            readme.write_text(
                dedent(
                    """\
                    # Markdown Memory

                    Stage-one memory is stored as human-auditable Markdown.

                    - `profile.md`: stable user preferences.
                    - `project-knowledge.md`: stable project facts.
                    - `workflows.md`: reusable operating procedures.
                    - `tool-history.md`: tool usage lessons.
                    - `safety.md`: safety and privacy rules.
                    - `pending-confirmation/`: high-risk or conflicting memories.
                    - `archive/`: compacted memory sections.
                    """
                ),
                encoding="utf-8",
                newline="\n",
            )
        for filename in self.CORE_FILES.values():
            path = self.root / filename
            if not path.exists():
                title = filename.removesuffix(".md").replace("-", " ").title()
                path.write_text(f"# {title}\n\n", encoding="utf-8", newline="\n")

    def _render_stable(
        self,
        candidate: MemoryCandidate,
        decision: MemoryDecision,
    ) -> str:
        tags = ", ".join(candidate.tags)
        return dedent(
            f"""\

            ## {candidate.title}

            source_event_id: {candidate.source_event_id}
            confidence: {candidate.confidence:.2f}
            auto_written_at: {candidate.created_at}
            policy_reason: {decision.reason}
            tags: {tags}

            {candidate.content.strip()}
            """
        )

    def _render_pending(
        self,
        candidate: MemoryCandidate,
        decision: MemoryDecision,
    ) -> str:
        tags = ", ".join(candidate.tags)
        return dedent(
            f"""\

            ## {candidate.title}

            type: {candidate.memory_type}
            source_event_id: {candidate.source_event_id}
            confidence: {candidate.confidence:.2f}
            status: pending_confirmation
            policy_reason: {decision.reason}
            tags: {tags}
            created_at: {candidate.created_at}

            {candidate.content.strip()}
            """
        )

    def _candidate_fingerprint(self, candidate: MemoryCandidate) -> str:
        stable = "\n".join(
            [
                candidate.title,
                candidate.memory_type,
                ",".join(sorted(candidate.tags)),
                candidate.content.strip(),
            ]
        )
        return self._text_fingerprint(stable)

    def _candidate_block_fingerprint(self, block: str) -> str:
        lines = block.splitlines()
        title = lines[0].removeprefix("##").strip() if lines else ""
        metadata: dict[str, str] = {}
        content_start = len(lines)
        metadata_started = False
        for index, line in enumerate(lines[1:], start=1):
            clean = line.strip()
            if not clean and not metadata_started:
                continue
            if not clean and metadata_started:
                content_start = index + 1
                break
            if ":" in clean:
                metadata_started = True
                key, value = clean.split(":", 1)
                metadata[key.strip()] = value.strip()
                continue
            if metadata_started:
                content_start = index
                break
        tags = ",".join(
            sorted(item.strip() for item in metadata.get("tags", "").split(",") if item.strip())
        )
        stable = "\n".join(
            [
                title,
                metadata.get("type", "knowledge"),
                tags,
                "\n".join(lines[content_start:]).strip(),
            ]
        )
        return self._text_fingerprint(stable)

    def _text_fingerprint(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()

    def _candidate_blocks(self, text: str) -> list[str]:
        matches = list(re.finditer(r"^##\s+.+$", text, flags=re.MULTILINE))
        blocks = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            blocks.append(text[match.start() : end])
        return blocks

    def _iter_sections(self, text: str, fallback_title: str) -> list[tuple[str, str]]:
        matches = list(re.finditer(r"^##\s+(.+)$", text, flags=re.MULTILINE))
        if not matches:
            return [(fallback_title, text)]
        sections = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            sections.append((match.group(1).strip(), text[match.end() : end].strip()))
        return sections

    def _split_document(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        lines = text.splitlines()
        document_title = lines[0].strip() if lines and lines[0].startswith("# ") else "# Memory"
        if not re.search(r"^##\s+.+$", text, flags=re.MULTILINE):
            return document_title, []
        return document_title, self._iter_sections(text, document_title.removeprefix("# ").strip())

    def _render_document(self, title: str, sections: list[tuple[str, str]]) -> str:
        rendered = [title.strip(), ""]
        for section_title, body in sections:
            rendered.extend([f"## {section_title}", "", body.strip(), ""])
        return "\n".join(rendered).rstrip() + "\n"

    def _archive_sections(self, filename: str, sections: list[tuple[str, str]]) -> None:
        if not sections:
            return
        archive_path = self.archive / f"{filename}.archive.md"
        with archive_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(self._render_document(f"# Archived {filename}", sections))

    def _section_fingerprint(self, title: str, body: str) -> str:
        volatile_prefixes = (
            "source_event_id:",
            "confidence:",
            "auto_written_at:",
            "created_at:",
            "policy_reason:",
        )
        stable_lines = [
            line
            for line in body.splitlines()
            if not line.strip().startswith(volatile_prefixes)
        ]
        return self._text_fingerprint(title + "\n" + "\n".join(stable_lines))

    def _target_for_type(self, memory_type: str) -> str:
        return {
            "preference": "profile.md",
            "knowledge": "project-knowledge.md",
            "workflow": "workflows.md",
            "tool_history": "tool-history.md",
            "safety": "safety.md",
        }.get(memory_type, "project-knowledge.md")

    @staticmethod
    def _core_memory_type(fact: "FactCard") -> str:
        tags = set(fact.tags)
        if "preference" in tags:
            return "preference"
        if "safety" in tags:
            return "safety"
        if "workflow" in tags:
            return "workflow"
        if "shell" in tags or "failure" in tags:
            return "tool_history"
        return "knowledge"

    @staticmethod
    def _core_mirror_title(fact: "FactCard") -> str:
        evidence = fact.evidence.strip()
        prefixes = {
            "读取文件 ": "Knowledge from ",
            "列出目录 ": "Directory listing: ",
            "执行命令 ": "Shell: ",
        }
        for prefix, title_prefix in prefixes.items():
            if evidence.startswith(prefix):
                return f"{title_prefix}{evidence[len(prefix):]}"
        return f"[{fact.entity}] {fact.attribute}"

    def _normalize_source(self, source: str | None) -> str | None:
        if not source:
            return None
        value = source.strip()
        if value in self.CORE_FILES:
            return self.CORE_FILES[value]
        if value in self.CORE_FILES.values():
            return value
        raise ValueError(
            "unknown memory source: "
            f"{source}. Expected one of: {', '.join(self.CORE_FILES.values())}"
        )

    def _normalize_layer(self, layer: str | None) -> str | None:
        if not layer:
            return None
        value = layer.strip().lower()
        aliases = {
            "l1": "hot",
            "layer1": "hot",
            "first": "hot",
            "l2": "warm",
            "layer2": "warm",
            "second": "warm",
            "l3": "cold",
            "layer3": "cold",
            "third": "cold",
        }
        value = aliases.get(value, value)
        if value in self.MEMORY_LAYERS:
            return value
        raise ValueError(
            "unknown memory layer: "
            f"{layer}. Expected one of: {', '.join(self.MEMORY_LAYERS)}"
        )

    def _normalize_page_strategy(self, strategy: str | None) -> str:
        value = (strategy or "layered").strip().lower()
        aliases = {
            "default": "layered",
            "stable": "layered",
            "working_set": "priority",
            "working-set": "priority",
            "query": "priority",
        }
        value = aliases.get(value, value)
        if value in self.PAGE_STRATEGIES:
            return value
        raise ValueError(
            "unknown memory page strategy: "
            f"{strategy}. Expected one of: {', '.join(self.PAGE_STRATEGIES)}"
        )

    def _memory_page_items(self, source: str | None) -> list[MemoryPageItem]:
        filenames = [source] if source else list(self.CORE_FILES.values())
        items: list[MemoryPageItem] = []
        paging_index = self._load_paging_index()
        tier_index = self._load_tier_index()
        for filename in filenames:
            if filename is None:
                continue
            path = self.root / filename
            text = path.read_text(encoding="utf-8", errors="replace")
            for title, content in self._iter_sections(text, filename):
                clean = self._clean_page_content(content)
                if not clean:
                    continue
                stats = paging_index.get(self._memory_item_key(filename, title), {})
                tier_stats = tier_index.get(self._section_key(filename, title), {})
                recency = self._recency_score(tier_stats, stats)
                importance = self._importance_score(content, filename)
                access_count = max(
                    self._int_value(stats.get("access_count", 0)),
                    self._int_value(tier_stats.get("access_count", 0)),
                )
                tier = self._layer_for_tier(self._tier_for(self._section_key(filename, title), tier_index))
                items.append(
                    MemoryPageItem(
                        source=filename,
                        title=title,
                        content=clean,
                        index=len(items) + 1,
                        layer=tier,
                        access_count=access_count,
                        last_accessed=str(
                            tier_stats.get("last_accessed")
                            or stats.get("last_accessed", "")
                        ),
                        recency_score=recency,
                        importance_score=importance,
                    )
                )
        return items

    def _sort_memory_page_items(
        self,
        items: list[MemoryPageItem],
        strategy: str = "layered",
        page_size: int = 10,
    ) -> list[MemoryPageItem]:
        if strategy == "priority":
            return self._sort_priority_page_items(items, page_size=page_size)
        layer_order = {"hot": 0, "warm": 1, "cold": 2}
        return sorted(
            items,
            key=lambda item: (
                layer_order.get(item.layer, 99),
                -item.access_count,
                item.source,
                item.index,
            ),
        )

    def _sort_priority_page_items(
        self,
        items: list[MemoryPageItem],
        page_size: int,
    ) -> list[MemoryPageItem]:
        buckets = {
            layer: sorted(
                [item for item in items if item.layer == layer],
                key=self._priority_sort_key,
            )
            for layer in self.MEMORY_LAYERS
        }
        ordered: list[MemoryPageItem] = []
        remaining_total = sum(len(bucket) for bucket in buckets.values())
        while remaining_total:
            slots = max(page_size, 1)
            page_items: list[MemoryPageItem] = []
            quotas = self._layer_page_quotas(slots)
            for layer in self.MEMORY_LAYERS:
                while buckets[layer] and quotas[layer] > 0 and len(page_items) < slots:
                    page_items.append(buckets[layer].pop(0))
                    quotas[layer] -= 1
            while len(page_items) < slots:
                candidates = [bucket[0] for bucket in buckets.values() if bucket]
                if not candidates:
                    break
                best = sorted(candidates, key=self._priority_sort_key)[0]
                buckets[best.layer].pop(0)
                page_items.append(best)
            ordered.extend(page_items)
            remaining_total = sum(len(bucket) for bucket in buckets.values())
        return ordered

    def _priority_sort_key(self, item: MemoryPageItem) -> tuple[float, float, float, float, str, int]:
        return (
            -item.priority_score,
            -item.relevance_score,
            -item.recency_score,
            -item.importance_score,
            item.source,
            item.index,
        )

    def _layer_page_quotas(self, page_size: int) -> dict[str, int]:
        quotas = {
            layer: int(page_size * self.LAYER_PAGE_QUOTAS[layer])
            for layer in self.MEMORY_LAYERS
        }
        for layer in self.MEMORY_LAYERS:
            if page_size >= len(self.MEMORY_LAYERS) and quotas[layer] == 0:
                quotas[layer] = 1
        while sum(quotas.values()) > page_size:
            for layer in reversed(self.MEMORY_LAYERS):
                if quotas[layer] > 0 and sum(quotas.values()) > page_size:
                    quotas[layer] -= 1
        return quotas

    def _score_page_items(self, items: list[MemoryPageItem], query: str | None) -> None:
        relevance_by_key: dict[str, float] = {}
        clean_query = query.strip() if query else ""
        if clean_query:
            terms = self._tokenize(clean_query)
            relevance_by_key.update(self._keyword_relevance_scores(items, terms))
            for hit in self.vector_index.search(clean_query, items, limit=len(items)):
                key = self._memory_item_key(hit.item.source, hit.item.title)
                relevance_by_key[key] = max(
                    relevance_by_key.get(key, 0.0),
                    self._clamp_score(hit.similarity),
                )
        for item in items:
            key = self._memory_item_key(item.source, item.title)
            item.relevance_score = round(relevance_by_key.get(key, 0.0), 4)
            item.priority_score = round(self._page_priority_score(item), 4)

    def _keyword_relevance_scores(
        self,
        items: list[MemoryPageItem],
        terms: list[str],
    ) -> dict[str, float]:
        if not terms:
            return {}
        raw_scores: dict[str, int] = {}
        max_score = 0
        for item in items:
            haystack = f"{item.title}\n{item.content}".lower()
            score = sum(haystack.count(term) for term in terms)
            if score <= 0:
                continue
            key = self._memory_item_key(item.source, item.title)
            raw_scores[key] = score
            max_score = max(max_score, score)
        if max_score <= 0:
            return {}
        return {
            key: self._clamp_score(score / max_score)
            for key, score in raw_scores.items()
        }

    def _page_priority_score(self, item: MemoryPageItem) -> float:
        layer_score = {"hot": 1.0, "warm": 0.55, "cold": 0.0}.get(item.layer, 0.0)
        return (
            0.45 * item.relevance_score
            + 0.25 * item.recency_score
            + 0.20 * item.importance_score
            + 0.10 * layer_score
        )

    def _recency_score(self, tier_stats: dict, paging_stats: dict) -> float:
        timestamp = str(
            tier_stats.get("last_accessed")
            or paging_stats.get("last_accessed")
            or ""
        )
        if not timestamp:
            return 0.0
        return self._clamp_score(self.DECAY_RATE ** self._days_since(timestamp))

    def _importance_score(self, content: str, source: str) -> float:
        metadata = self._section_metadata(content)
        explicit = metadata.get("importance") or metadata.get("importance_score")
        if explicit is not None:
            parsed = self._float_value(explicit)
            if parsed is not None:
                return self._clamp_score(parsed)
        confidence = metadata.get("confidence")
        source_default = {
            "safety.md": 0.90,
            "profile.md": 0.80,
            "workflows.md": 0.70,
            "project-knowledge.md": 0.60,
            "tool-history.md": 0.50,
        }.get(source, 0.50)
        parsed_confidence = self._float_value(confidence)
        if parsed_confidence is None:
            return source_default
        return self._clamp_score(max(parsed_confidence, source_default))

    def _record_retrieved_access(self, results: list[RetrievedMemory]) -> None:
        if not results:
            return
        paging_index = self._load_paging_index()
        tier_index = self._load_tier_index()
        now = utc_now_iso()
        for result in results:
            key = self._memory_item_key(result.source, result.title)
            stats = paging_index.get(key, {})
            access_count = self._int_value(stats.get("access_count", 0)) + 1
            paging_index[key] = {
                "source": result.source,
                "title": result.title,
                "access_count": access_count,
                "last_accessed": now,
            }
            self._bump_access(self._section_key(result.source, result.title), tier_index)
        self._save_paging_index(paging_index)
        self._save_tier_index(tier_index)

    def _load_paging_index(self) -> dict[str, dict]:
        if not self.paging_index.exists():
            return {}
        try:
            data = json.loads(self.paging_index.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        values: dict[str, dict] = {}
        for key, value in data.items():
            if isinstance(key, str) and isinstance(value, dict):
                values[key] = value
        return values

    def _save_paging_index(self, index: dict[str, dict]) -> None:
        self.paging_index.write_text(
            json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
            newline="\n",
        )

    def _memory_item_key(self, source: str, title: str) -> str:
        return self._text_fingerprint(f"{source}\n{title}")

    def _layer_for_access_count(self, access_count: int) -> str:
        if access_count >= self.HOT_ACCESS_THRESHOLD:
            return "hot"
        if access_count >= self.WARM_ACCESS_THRESHOLD:
            return "warm"
        return "cold"

    @staticmethod
    def _layer_for_tier(tier: int) -> str:
        return {0: "hot", 1: "warm"}.get(tier, "cold")

    def _int_value(self, value: object) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _float_value(self, value: object) -> float | None:
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _clamp_score(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _section_metadata(self, content: str) -> dict[str, str]:
        metadata: dict[str, str] = {}
        for line in content.splitlines():
            clean = line.strip()
            if not clean:
                if metadata:
                    break
                continue
            if ":" not in clean:
                if metadata:
                    break
                continue
            key, value = clean.split(":", 1)
            normalized_key = key.strip().lower()
            if re.fullmatch(r"[a-z][a-z0-9_]*", normalized_key):
                metadata[normalized_key] = value.strip()
        return metadata

    def _clean_page_content(self, content: str) -> str:
        lines = [line.rstrip() for line in content.splitlines()]
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines:
            return ""
        if len(lines) == 1 and lines[0].startswith("# "):
            return ""
        return "\n".join(lines).strip()

    def _tokenize(self, text: str) -> list[str]:
        ascii_terms = re.findall(r"[a-z0-9_./\\-]{2,}", text.lower())
        cjk_terms = re.findall(r"[\u4e00-\u9fff]{2,}", text)
        return sorted(set(ascii_terms + cjk_terms))

    def _excerpt(self, text: str, terms: list[str], max_chars: int = 500) -> str:
        clean = " ".join(line.strip() for line in text.splitlines() if line.strip())
        lower = clean.lower()
        first_hit = min(
            [lower.find(term) for term in terms if lower.find(term) != -1],
            default=0,
        )
        start = max(first_hit - 120, 0)
        return clean[start : start + max_chars]
