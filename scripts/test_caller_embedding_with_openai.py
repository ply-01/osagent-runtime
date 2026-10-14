"""Test OSagent with OpenAI embedding (simulating Codex's stronger embedding)."""
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

# 使用中转 API 示例：
# export OPENAI_API_KEY=sk-...
# export OPENAI_BASE_URL=https://your-relay.com/v1  # 可选，中转 API 地址

def test_with_external_embedding():
    """Use OpenAI embedding as a proxy for Codex's stronger embedding."""
    try:
        import openai
    except ImportError:
        print("ERROR: openai package not installed")
        print("Install with: pip install openai")
        return

    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from os_agent.memory.module import AgentMemoryModule
    from os_agent.memory.memory_config import MemoryConfig
    from os_agent.memory.interface import RememberRequest, RecallRequest

    # Initialize OpenAI (supports relay API via base_url)
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")  # 中转 API 地址

    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY=sk-...")
        return

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
        print(f"Using relay API: {base_url}")
    else:
        print("Using official OpenAI API")

    client = openai.Client(**client_kwargs)

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        memory = AgentMemoryModule(
            root / "memory",
            config=MemoryConfig(root / "config.json")
        )

        # Test data - chinese and english mixed
        print("\n=== Preparing test data ===")
        test_facts = [
            ("user", "document_format", "PDF", "用户明确表示喜欢 PDF 格式的文档"),
            ("user", "save_location", "D:/Work", "用户习惯把工作文件保存在 D:/Work 目录"),
            ("project", "test_framework", "pytest", "这个项目使用 pytest 作为测试框架"),
            ("user", "theme", "dark", "用户偏好使用深色主题界面"),
            ("user", "hotel_preference", "quiet", "User prefers quiet hotels near transit"),
        ]

        for entity, attr, value, evidence in test_facts:
            memory.remember(RememberRequest(
                content=evidence,
                entity=entity,
                attribute=attr,
                value=value,
                evidence=evidence
            ))
        print(f"Added {len(test_facts)} facts to memory")

        # Test queries (paraphrased/cross-language)
        test_queries = [
            ("用户喜欢什么格式的文档", "document_format", "PDF"),
            ("what file format does user prefer", "document_format", "PDF"),  # cross-language
            ("文件应该存放在哪里", "save_location", "D:/Work"),
            ("where to save work files", "save_location", "D:/Work"),  # cross-language
            ("使用的是什么测试工具", "test_framework", "pytest"),
            ("用户界面颜色偏好", "theme", "dark"),
            ("安静的酒店", "hotel_preference", "quiet"),  # cross-language
        ]

        print("\n=== Baseline: OSagent local embedding (e5-small) ===")
        baseline_results = []
        for query, expected_attr, expected_value in test_queries:
            result = memory.recall(RecallRequest(query=query, limit=3))
            if result.records:
                top1 = result.records[0]
                correct = expected_value in str(top1.value)
                baseline_results.append(correct)
                status = "PASS" if correct else "FAIL"
                print(f"{status} | {query[:45]:45} -> {top1.attribute}={top1.value} | sem={top1.metadata.get('semantic_score', 0):.3f}")
            else:
                baseline_results.append(False)
                print(f"FAIL | {query[:45]:45} -> (no results)")

        baseline_acc = sum(baseline_results) / len(baseline_results)
        print(f"\nBaseline Accuracy: {sum(baseline_results)}/{len(baseline_results)} = {baseline_acc:.1%}")

        # Now test with OpenAI embedding (simulating Codex)
        print("\n=== With OpenAI embedding (simulating Codex) ===")
        openai_results = []
        for query, expected_attr, expected_value in test_queries:
            try:
                # Get OpenAI embedding
                response = client.embeddings.create(
                    input=query,
                    model="text-embedding-3-small"
                )
                openai_embedding = response.data[0].embedding

                # Call OSagent with external embedding
                result = memory.recall(RecallRequest(
                    query=query,
                    limit=3,
                    query_embedding=openai_embedding
                ))

                if result.records:
                    top1 = result.records[0]
                    correct = expected_value in str(top1.value)
                    openai_results.append(correct)
                    status = "PASS" if correct else "FAIL"
                    print(f"{status} | {query[:45]:45} -> {top1.attribute}={top1.value} | sem={top1.metadata.get('semantic_score', 0):.3f}")
                else:
                    openai_results.append(False)
                    print(f"FAIL | {query[:45]:45} -> (no results)")

            except Exception as e:
                print(f"ERROR | {query[:45]:45} -> {str(e)}")
                openai_results.append(False)

        openai_acc = sum(openai_results) / len(openai_results)
        print(f"\nOpenAI Embedding Accuracy: {sum(openai_results)}/{len(openai_results)} = {openai_acc:.1%}")

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Baseline (e5-small):          {baseline_acc:.1%}")
        print(f"OpenAI (text-embedding-3-small): {openai_acc:.1%}")
        improvement = openai_acc - baseline_acc
        print(f"Improvement:                  {improvement:+.1%}")

        if improvement > 0:
            print("\n✓ OpenAI embedding improves recall quality!")
            print("  This suggests using Codex's embedding would help.")
        elif improvement < 0:
            print("\n✗ OpenAI embedding performed worse (unexpected)")
            print("  Check if embedding dimensions match.")
        else:
            print("\n= No difference detected")
            print("  Test set may be too easy or sample size too small.")


if __name__ == "__main__":
    test_with_external_embedding()
