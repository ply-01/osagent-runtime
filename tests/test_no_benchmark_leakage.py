"""Guard against copying evaluation cases into production memory code."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from typing import ClassVar

from benchmarks.competition_memory_eval import (
    PREFERENCE_CASES,
    RETRIEVAL_CASES,
    sanitize_evaluation_output,
)


class BenchmarkLeakageTests(unittest.TestCase):
    _PUBLIC_LABEL_KEYS: ClassVar[set[str]] = {
        "expected_preference",
        "expected_state",
        "relevant_fact_ids",
        "gold_label",
        "memory_evidence",
        "forgetting_evidence",
    }

    def test_production_code_does_not_import_benchmark_modules(self) -> None:
        source_root = Path("src/os_agent")
        for path in source_root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.append(node.module)
            self.assertFalse(
                any(name == "benchmarks" or name.startswith("benchmarks.") for name in imported),
                f"production module imports benchmark code: {path}",
            )

    def test_full_evaluation_prompts_are_absent_from_production_code(self) -> None:
        production = "\n".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in Path("src/os_agent").rglob("*.py")
        )
        prompts = [case.text for case in PREFERENCE_CASES]
        prompts.extend(case.query for case in RETRIEVAL_CASES)
        for prompt in prompts:
            self.assertNotIn(prompt, production, f"evaluation prompt leaked into production: {prompt}")

    def test_public_benchmark_gold_fields_are_absent_from_production_code(self) -> None:
        production = "\n".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in Path("src/os_agent").rglob("*.py")
        )
        for key in self._PUBLIC_LABEL_KEYS:
            self.assertNotIn(key, production, f"public benchmark label leaked into production: {key}")

    def test_published_evaluation_omits_failure_examples(self) -> None:
        public = sanitize_evaluation_output(
            {
                "summary": {"accuracy": 0.5},
                "failures": [{"text": "private case"}],
                "nested": {"failures": [{"query": "private query"}]},
            }
        )

        self.assertEqual(public["summary"], {"accuracy": 0.5})
        self.assertNotIn("failures", public)
        self.assertNotIn("failures", public["nested"])


if __name__ == "__main__":
    unittest.main()
