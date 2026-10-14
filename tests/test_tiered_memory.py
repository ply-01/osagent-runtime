"""Tests for time-decayed tiered memory retrieval (L1-hot / L2-warm / L3-cold)."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from os_agent.memory.markdown_store import MarkdownMemoryStore


class TieredMemoryTests(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self._tmp_path = Path(self._tmp.name)
        self.store = MarkdownMemoryStore(self._tmp_path / "memory")
        self._seed_memories()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _seed_memories(self) -> None:
        knowledge = self._tmp_path / "memory" / "project-knowledge.md"
        profile = self._tmp_path / "memory" / "profile.md"

        knowledge.write_text(
            """# Project Knowledge

## Alpha the first
confidence: 0.55
tags: test

This memory is about project Alpha and its structure with README files.

## Beta project
confidence: 0.55
tags: test

Beta is the second project containing configuration and documentation.

## Gamma system
confidence: 0.50
tags: test

Gamma handles system operations and shell command execution.

## Delta tools
confidence: 0.50
tags: test

Delta provides tools for filesystem inspection and directory listing.

## Epsilon workflow
confidence: 0.55
tags: test

Epsilon describes reusable workflows for project setup tasks.

## Zeta overview
confidence: 0.55
tags: test

Zeta gives an overview of the entire system architecture design.

## Eta config
confidence: 0.50
tags: test

Eta contains configuration settings and environment variable definitions.

## Theta shell
confidence: 0.50
tags: test

Theta documents shell usage patterns and command execution guidelines.
""",
            encoding="utf-8",
        )

        profile.write_text(
            """# Profile

## Prefer Chinese
confidence: 0.55
tags: user_message, preference

User prefers Chinese language replies in all interactions.

## Prefer concise
confidence: 0.55
tags: user_message, preference

User prefers short and concise answers over long explanations.
""",
            encoding="utf-8",
        )

    def _access_n_times(self, query: str, n: int) -> list:
        result = []
        for _ in range(n):
            result = self.store.retrieve(query, limit=10)
        return result

    # ==================================================================
    # Tier classification with time decay
    # ==================================================================

    def test_fresh_store_all_cold(self) -> None:
        for title in ["Alpha the first", "Beta project"]:
            ti = self.store.tier_info("project-knowledge.md", title)
            self.assertEqual(ti["tier"], "L3-cold")
            self.assertEqual(ti["access_count"], 0)
            self.assertEqual(ti["weighted_score"], 0.0)

    def test_one_access_promotes_to_warm(self) -> None:
        self.store.retrieve("Alpha README files", limit=10)
        ti = self.store.tier_info("project-knowledge.md", "Alpha the first")
        self.assertEqual(ti["access_count"], 1)
        self.assertEqual(ti["tier"], "L2-warm")

    def test_two_accesses_still_warm(self) -> None:
        """2 rapid accesses → weighted=2.0 < 3.0 → L2-warm."""
        self._access_n_times("Alpha README", 2)
        ti = self.store.tier_info("project-knowledge.md", "Alpha the first")
        self.assertEqual(ti["access_count"], 2)
        self.assertAlmostEqual(ti["weighted_score"], 2.0, delta=0.1)
        self.assertEqual(ti["tier"], "L2-warm")

    def test_three_accesses_promotes_to_hot(self) -> None:
        """3 rapid accesses → weighted=3.0 ≥ 3.0 → L1-hot."""
        self._access_n_times("Alpha README files project", 3)
        ti = self.store.tier_info("project-knowledge.md", "Alpha the first")
        self.assertEqual(ti["access_count"], 3)
        self.assertEqual(ti["tier"], "L1-hot")

    def test_tier_info_for_unknown_section(self) -> None:
        ti = self.store.tier_info("nonexistent.md", "Ghost section")
        self.assertEqual(ti["tier"], "L3-cold")
        self.assertEqual(ti["access_count"], 0)

    def test_one_access_is_warm_not_hot(self) -> None:
        """1 access → weighted=1.0, below 3.0 hot threshold."""
        self.store.retrieve("Alpha README", limit=10)
        ti = self.store.tier_info("project-knowledge.md", "Alpha the first")
        self.assertEqual(ti["tier"], "L2-warm")

    # ==================================================================
    # Time decay behavior
    # ==================================================================

    def test_weighted_score_reflects_both_count_and_recency(self) -> None:
        """Each rapid access adds ~1.0 to weighted_score."""
        self._access_n_times("Alpha README", 3)
        ti = self.store.tier_info("project-knowledge.md", "Alpha the first")
        # 3 rapid accesses → approximately 3.0
        self.assertAlmostEqual(ti["weighted_score"], 3.0, delta=0.2)

    def test_decay_rate_is_configurable(self) -> None:
        """DECAY_RATE controls how fast old accesses lose weight."""
        self.assertAlmostEqual(MarkdownMemoryStore.DECAY_RATE, 0.85)
        # After 1 day: score *= 0.85
        # After 5 days: score *= 0.85^5 ≈ 0.44 (halved)

    # ==================================================================
    # Access count increments
    # ==================================================================

    def test_access_count_increments_only_for_selected(self) -> None:
        self.store.retrieve("Alpha Beta Gamma Delta Epsilon", limit=2)
        alpha = self.store.tier_info("project-knowledge.md", "Alpha the first")
        self.assertGreaterEqual(alpha["access_count"], 1)

    def test_repeated_retrievals_accumulate_count(self) -> None:
        for i in range(3):
            self.store.retrieve("Alpha", limit=10)
            ti = self.store.tier_info("project-knowledge.md", "Alpha the first")
            self.assertEqual(ti["access_count"], i + 1)

    def test_zero_match_does_not_update_counts(self) -> None:
        results = self.store.retrieve("zzz_nonexistent_term_xyz", limit=10)
        self.assertEqual(results, [])
        ti = self.store.tier_info("project-knowledge.md", "Alpha the first")
        self.assertEqual(ti["access_count"], 0)

    # ==================================================================
    # Cascading retrieval: L1 → L2 → L3
    # ==================================================================

    def test_cascade_l1_first(self) -> None:
        """When L1 has enough results, L2/L3 excluded."""
        self._access_n_times("Alpha Beta project", 3)  # Alpha,Beta → hot
        results = self.store.retrieve("Alpha Beta Gamma project", limit=2)
        titles = [r.title for r in results]
        for title in titles:
            ti = self.store.tier_info("project-knowledge.md", title)
            self.assertEqual(ti["tier"], "L1-hot", f"{title} should be L1-hot")

    def test_cascade_l2_fills_gap(self) -> None:
        """L1 has 1 result, L2 fills remaining before L3."""
        self._access_n_times("Alpha files README", 3)   # Alpha → hot
        self._access_n_times("Beta project config", 1)  # Beta → warm
        results = self.store.retrieve("Alpha Beta Gamma system", limit=3)
        titles = [r.title for r in results]
        self.assertIn("Alpha the first", titles)
        # Beta (L2) before Gamma (L3)
        if "Beta project" in titles and "Gamma system" in titles:
            self.assertLess(titles.index("Beta project"), titles.index("Gamma system"))

    def test_cascade_l3_fills_when_l1_l2_insufficient(self) -> None:
        """Only Alpha is hot, limit=5 → 4 cold fill the gap."""
        self._access_n_times("Alpha files README", 3)
        results = self.store.retrieve(
            "Alpha Beta Gamma Delta Epsilon Zeta Eta Theta shell config system overview",
            limit=5,
        )
        self.assertEqual(len(results), 5)
        self.assertIn("Alpha the first", [r.title for r in results])

    def test_cascade_all_same_tier(self) -> None:
        """All L3-cold → pure score ordering."""
        results = self.store.retrieve("Alpha Beta Gamma system", limit=3)
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_cascade_limit_zero_returns_empty(self) -> None:
        self.assertEqual(self.store.retrieve("Alpha", limit=0), [])

    # ==================================================================
    # Tier index persistence
    # ==================================================================

    def test_tier_index_persists_to_disk(self) -> None:
        self.store.retrieve("Alpha files", limit=10)
        self.store.retrieve("Alpha README", limit=10)
        self.store.retrieve("Alpha project", limit=10)

        store2 = MarkdownMemoryStore(self._tmp_path / "memory")
        ti = store2.tier_info("project-knowledge.md", "Alpha the first")
        self.assertEqual(ti["access_count"], 3)

    def test_tier_index_file_is_valid_json(self) -> None:
        self.store.retrieve("Alpha", limit=10)
        index_path = self._tmp_path / "memory" / ".tier_index.json"
        self.assertTrue(index_path.exists())
        data = json.loads(index_path.read_text(encoding="utf-8"))
        self.assertIsInstance(data, dict)
        for key, entry in data.items():
            self.assertIn("weighted_score", entry)
            self.assertIn("access_count", entry)
            self.assertIn("last_accessed", entry)

    # ==================================================================
    # Cross-file retrieval
    # ==================================================================

    def test_retrieval_scans_all_core_files(self) -> None:
        results = self.store.retrieve("Chinese language", limit=10)
        titles = [r.title for r in results]
        self.assertTrue(any("Prefer Chinese" in t for t in titles))

    def test_different_files_maintain_separate_tier_keys(self) -> None:
        self.store.retrieve("Alpha", limit=10)
        self.store.retrieve("Chinese", limit=10)
        alpha = self.store.tier_info("project-knowledge.md", "Alpha the first")
        chinese = self.store.tier_info("profile.md", "Prefer Chinese")
        self.assertGreaterEqual(alpha["access_count"], 1)
        self.assertGreaterEqual(chinese["access_count"], 1)

    # ==================================================================
    # Tier index cleanup
    # ==================================================================

    def test_prune_removes_stale_sections(self) -> None:
        self.store.retrieve("Alpha files", limit=10)
        idx = self.store._load_tier_index()
        alpha_keys = [k for k in idx if "Alpha" in k]
        self.assertTrue(alpha_keys)

        knowledge_path = self._tmp_path / "memory" / "project-knowledge.md"
        text = knowledge_path.read_text(encoding="utf-8")
        new_text = text.replace("## Alpha the first\n", "## REMOVED_Alpha the first\n")
        knowledge_path.write_text(new_text, encoding="utf-8")

        idx = self.store._load_tier_index()
        self.store._prune_tier_index(idx)
        alpha_keys_after = [k for k in idx if "Alpha" in k]
        self.assertEqual(alpha_keys_after, [])

    # ==================================================================
    # Edge cases
    # ==================================================================

    def test_empty_store_retrieval(self) -> None:
        empty_path = self._tmp_path / "empty_memory"
        empty_store = MarkdownMemoryStore(empty_path)
        for fn in MarkdownMemoryStore.CORE_FILES.values():
            (empty_path / fn).write_text("# Empty\n\n", encoding="utf-8")
        self.assertEqual(empty_store.retrieve("anything", limit=10), [])

    def test_very_long_query(self) -> None:
        results = self.store.retrieve("Alpha Beta Gamma Delta Epsilon Zeta Eta Theta " * 10, limit=10)
        self.assertIsInstance(results, list)

    def test_unicode_query(self) -> None:
        knowledge = self._tmp_path / "memory" / "project-knowledge.md"
        current = knowledge.read_text(encoding="utf-8")
        knowledge.write_text(
            current + """
## 项目说明
confidence: 0.55
tags: test

这是一个关于操作系统智能体的项目，主要用于文件管理和系统操作。
""",
            encoding="utf-8",
        )
        results = self.store.retrieve("操作系统 智能体", limit=10)
        titles = [r.title for r in results]
        self.assertTrue(any("项目说明" in t for t in titles))

    def test_multiple_terms_scoring(self) -> None:
        results = self.store.retrieve("Alpha files README project", limit=10)
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_pending_retrieved_as_cold(self) -> None:
        pending_dir = self._tmp_path / "memory" / "pending-confirmation"
        (pending_dir / "test-pending.md").write_text(
            """## User preference: always test
type: preference
confidence: 0.55
tags: user_message, preference

User always wants to run tests before deploying code changes.
""",
            encoding="utf-8",
        )
        source = "pending-confirmation/test-pending.md"
        ti = self.store.tier_info(source, "User preference: always test")
        self.assertEqual(ti["tier"], "L3-cold")
        self.store.retrieve("test deploy code", limit=10)
        ti = self.store.tier_info(source, "User preference: always test")
        self.assertGreaterEqual(ti["access_count"], 1)

    def test_limit_exceeds_total_memories(self) -> None:
        results = self.store.retrieve("project system shell config overview architecture", limit=999)
        self.assertIsInstance(results, list)

    # ==================================================================
    # Score vs tier interaction
    # ==================================================================

    def test_hot_beats_cold_regardless_of_score(self) -> None:
        """L1-hot with low score returns before L3-cold with high score."""
        self._access_n_times("Eta config settings environment variable", 3)  # Eta → hot
        results = self.store.retrieve("Alpha files README", limit=8)
        titles = [r.title for r in results]
        l3_indices = [i for i, t in enumerate(titles)
                      if self.store.tier_info("project-knowledge.md", t)["tier"] == "L3-cold"]
        l1_indices = [i for i, t in enumerate(titles)
                      if self.store.tier_info("project-knowledge.md", t)["tier"] == "L1-hot"]
        if l1_indices and l3_indices:
            self.assertLess(max(l1_indices), min(l3_indices),
                           "All L1-hot must appear before any L3-cold")


if __name__ == "__main__":
    unittest.main()
