"""Use LLM to generate query variants for better recall (no embedding needed)."""
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

# export OPENAI_API_KEY=sk-...
# export OPENAI_BASE_URL=https://api.5566676.xyz/v1

def test_with_llm_query_expansion():
    """Use GPT to generate query variants instead of using embeddings."""
    try:
        import openai
    except ImportError:
        print("ERROR: openai package not installed")
        return

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from os_agent.memory.module import AgentMemoryModule
    from os_agent.memory.memory_config import MemoryConfig
    from os_agent.memory.interface import RememberRequest, RecallRequest

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")

    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
        print(f"Using API: {base_url}")

    client = openai.Client(**client_kwargs)

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        memory = AgentMemoryModule(
            root / "memory",
            config=MemoryConfig(root / "config.json")
        )

        # Test data
        print("\n=== Preparing test data ===")
        test_facts = [
            ("user", "document_format", "PDF", "User prefers PDF format documents"),
            ("user", "save_location", "D:/Work", "User saves work files in D:/Work"),
            ("project", "test_framework", "pytest", "Project uses pytest framework"),
            ("user", "theme", "dark", "User prefers dark theme interface"),
            ("user", "hotel", "quiet", "User likes quiet hotels near transit"),
        ]

        for entity, attr, value, evidence in test_facts:
            memory.remember(RememberRequest(
                content=evidence,
                entity=entity,
                attribute=attr,
                value=value,
                evidence=evidence
            ))
        print(f"Added {len(test_facts)} facts")

        # Test queries
        test_queries = [
            ("what format does user like", "document_format", "PDF"),
            ("where to save files", "save_location", "D:/Work"),
            ("what testing tool is used", "test_framework", "pytest"),
            ("user interface color", "theme", "dark"),
            ("quiet accommodation", "hotel", "quiet"),
        ]

        print("\n=== Baseline: simple query (no expansion) ===")
        baseline_results = []
        for query, expected_attr, expected_value in test_queries:
            result = memory.recall(RecallRequest(query=query, limit=3))
            if result.records:
                top1 = result.records[0]
                correct = expected_value in str(top1.value)
                baseline_results.append(correct)
                status = "PASS" if correct else "FAIL"
                print(f"{status} | {query[:40]:40} -> {top1.attribute}={top1.value}")
            else:
                baseline_results.append(False)
                print(f"FAIL | {query[:40]:40} -> (no results)")

        baseline_acc = sum(baseline_results) / len(baseline_results)
        print(f"\nBaseline: {sum(baseline_results)}/{len(baseline_results)} = {baseline_acc:.1%}")

        # Now test with LLM-generated query variants
        print("\n=== With LLM query expansion (GPT-5.5) ===")
        expanded_results = []
        for query, expected_attr, expected_value in test_queries:
            try:
                # Use LLM to generate query variants
                response = client.chat.completions.create(
                    model="gpt-5.5",
                    messages=[{
                        "role": "user",
                        "content": f"Generate 3 different ways to ask this question (synonyms, paraphrases): '{query}'\nReturn only the 3 variants, one per line, no numbering."
                    }],
                    temperature=0.7,
                    max_tokens=100
                )

                variants_text = response.choices[0].message.content.strip()
                variants = [v.strip() for v in variants_text.split('\n') if v.strip()]

                print(f"\nQuery: {query}")
                print(f"  Variants: {variants[:2]}")  # Show first 2

                # Call OSagent with query_variants
                result = memory.recall(RecallRequest(
                    query=query,
                    limit=3,
                    query_variants=variants
                ))

                if result.records:
                    top1 = result.records[0]
                    correct = expected_value in str(top1.value)
                    expanded_results.append(correct)
                    status = "PASS" if correct else "FAIL"
                    print(f"  {status} -> {top1.attribute}={top1.value}")
                else:
                    expanded_results.append(False)
                    print(f"  FAIL -> (no results)")

            except Exception as e:
                print(f"  ERROR: {str(e)[:60]}")
                expanded_results.append(False)

        expanded_acc = sum(expanded_results) / len(expanded_results)
        print(f"\nExpanded: {sum(expanded_results)}/{len(expanded_results)} = {expanded_acc:.1%}")

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Baseline (simple query):     {baseline_acc:.1%}")
        print(f"LLM expansion (query_variants): {expanded_acc:.1%}")
        improvement = expanded_acc - baseline_acc
        print(f"Improvement:                 {improvement:+.1%}")

        if improvement > 0:
            print("\nOK - LLM query expansion improves recall!")
            print("This is what Codex can do without needing embeddings.")
        else:
            print("\nNo improvement (test set may be too easy)")


if __name__ == "__main__":
    test_with_llm_query_expansion()
