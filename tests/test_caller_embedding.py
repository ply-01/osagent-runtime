"""Test caller-provided query embeddings for multi-agent interop."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from os_agent.memory.interface import RecallRequest, RememberRequest
from os_agent.memory.memory_config import MemoryConfig
from os_agent.memory.module import AgentMemoryModule


class DeterministicEmbeddingProvider:
    enabled = True
    cache_namespace = "test-caller-embedding-v1"

    def __init__(self):
        self.query_calls = 0

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_queries(self, texts):
        self.query_calls += 1
        return [[0.8, 0.6] for _ in texts]


class CallerEmbeddingTests(unittest.TestCase):
    def test_caller_can_provide_query_embedding(self) -> None:
        """Verify that callers can pass pre-computed query embeddings."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider = DeterministicEmbeddingProvider()
            memory = AgentMemoryModule(
                root / "memory", config=MemoryConfig(root / "config.json"),
                embedding_provider=provider,
            )

            # Store a fact
            memory.remember(
                RememberRequest(
                    content="User prefers PDF format",
                    entity="user",
                    attribute="document_format",
                    value="PDF",
                )
            )

            # Recall with local embedding (baseline)
            r1 = memory.recall(RecallRequest(query="retrieval probe", limit=5))
            self.assertEqual(len(r1.records), 1)
            baseline_score = r1.records[0].metadata.get("semantic_score", 0.0)

            # Same dimension and vector space as the document provider.
            before = provider.query_calls
            fake_embedding = [0.9, 0.4]
            r2 = memory.recall(
                RecallRequest(
                    query="retrieval probe",
                    limit=5,
                    query_embedding=fake_embedding,
                )
            )
            self.assertEqual(len(r2.records), 1)
            caller_score = r2.records[0].metadata.get("semantic_score", 0.0)

            # Both should return results (scores will differ since fake embedding is random)
            self.assertGreater(baseline_score, 0.0)
            self.assertGreater(caller_score, 0.0)
            self.assertEqual(provider.query_calls, before)
            self.assertNotEqual(baseline_score, caller_score)

    def test_mcp_recall_accepts_query_embedding(self) -> None:
        """Verify MCP service accepts query_embedding parameter."""
        from os_agent.mcp_server import OSAgentMCPConfig, OSAgentMCPService

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = OSAgentMCPConfig(
                workspace=root / "workspace",
                memory_root=root / "memory",
                allow_memory_mutations=True,
            )
            service = OSAgentMCPService(config)

            # Remember via MCP
            service.remember(
                content="User likes quiet hotels",
                entity="user",
                attribute="hotel_preference",
                value="quiet",
            )

            # Recall with caller embedding
            fake_embedding = [0.3] * 128
            result = service.recall(
                query="hotel preference", limit=5, query_embedding=fake_embedding
            )

            records = result.get("records", [])
            self.assertGreaterEqual(len(records), 1)
            # Verify semantic_score is present (indicating semantic search ran)
            self.assertIn("semantic_score", records[0].get("metadata", {}))


if __name__ == "__main__":
    unittest.main()
