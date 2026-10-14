import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from os_agent.memory.module import AgentMemoryModule
from os_agent.subagents.native_memory import (
    LocalMemorySubagent,
    LocalMemorySubagentConfig,
    OSagentMemorySubagentClient,
    OSagentMemorySubagentConfig,
)
from os_agent.subagents.server import create_server


class DisabledEmbeddingProvider:
    enabled = False
    cache_namespace = "test-disabled"

    def embed_queries(self, texts):
        return []

    def embed_documents(self, texts):
        return []


def build_subagent(root: Path) -> LocalMemorySubagent:
    config = LocalMemorySubagentConfig(root=root, top_k=8)
    module = AgentMemoryModule(
        root,
        embedding_provider=DisabledEmbeddingProvider(),  # type: ignore[arg-type]
    )
    return LocalMemorySubagent(config, memory_module=module)


class LocalMemorySubagentTests(unittest.TestCase):
    def test_observe_recall_and_restart_are_persistent(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "subagent-memory"
            subagent = build_subagent(root)

            stored = subagent.observe_turn(
                session_id="session-one",
                user_input="I prefer dark mode for every project.",
                assistant_output="I will remember that preference.",
            )
            hits = subagent.recall("dark mode preference", session_id="session-two")

            self.assertEqual(stored["status"], "stored")
            self.assertTrue(any("dark mode" in hit.content.lower() for hit in hits))

            restarted = build_subagent(root)
            restarted_hits = restarted.recall(
                "dark mode preference",
                session_id="session-three",
            )
            self.assertTrue(
                any("dark mode" in hit.content.lower() for hit in restarted_hits)
            )
            self.assertGreaterEqual(restarted.health()["observed_turns"], 1)

            restarted.observe_turn(
                session_id="session-delete-source",
                user_input="The transient smoke marker is Atlas-Z9.",
                assistant_output="Acknowledged.",
            )
            deletion_hits = restarted.recall(
                "Atlas-Z9",
                session_id="session-delete-query",
            )
            raw_hit = next(
                hit for hit in deletion_hits if hit.memory_id.startswith("event:")
            )
            delete_ready = build_subagent(root)
            reloaded_hit = next(
                hit
                for hit in delete_ready.recall(
                    "Atlas-Z9",
                    session_id="session-delete-after-restart",
                )
                if hit.memory_id.startswith("event:")
            )
            self.assertEqual(reloaded_hit.memory_id, raw_hit.memory_id)

            deleted = delete_ready.manage(
                "forget",
                {"scope": "ltm", "memory_id": reloaded_hit.memory_id},
                session_id="session-delete",
            )
            self.assertTrue(deleted["ok"])
            self.assertFalse(
                any(
                    "atlas-z9" in hit.content.lower()
                    for hit in delete_ready.recall(
                        "Atlas-Z9",
                        session_id="session-after-delete",
                    )
                )
            )

            after_delete_restart = build_subagent(root)
            self.assertFalse(
                any(
                    "atlas-z9" in hit.content.lower()
                    for hit in after_delete_restart.recall(
                        "Atlas-Z9",
                        session_id="session-after-delete-restart",
                    )
                )
            )

    def test_observe_turn_is_idempotent(self):
        with TemporaryDirectory() as tmp:
            subagent = build_subagent(Path(tmp) / "subagent-memory")
            arguments = {
                "session_id": "same-session",
                "user_input": "Remember the release is Friday.",
                "assistant_output": "Recorded.",
            }

            first = subagent.observe_turn(**arguments)
            second = subagent.observe_turn(**arguments)

            self.assertTrue(first["stored"])
            self.assertFalse(second["stored"])
            self.assertEqual(second["status"], "duplicate")

    def test_http_client_reaches_real_subagent(self):
        with TemporaryDirectory() as tmp:
            subagent = build_subagent(Path(tmp) / "subagent-memory")
            server = create_server("127.0.0.1", 0, subagent)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                port = int(server.server_address[1])
                client = OSagentMemorySubagentClient(
                    OSagentMemorySubagentConfig(
                        base_url=f"http://127.0.0.1:{port}",
                        top_k=8,
                    )
                )

                stored = client.observe_turn(
                    session_id="http-write",
                    user_input="My deployment window is Friday evening.",
                    assistant_output="The deployment window was recorded.",
                )
                hits = client.recall(
                    "Friday deployment window",
                    session_id="http-read",
                )
                health = client.health()

                self.assertTrue(stored["stored"])
                self.assertTrue(any("Friday" in hit.content for hit in hits))
                self.assertEqual(health["status"], "ok")
                self.assertGreaterEqual(health["sessions"], 1)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_http_client_manages_durable_memory(self):
        with TemporaryDirectory() as tmp:
            subagent = build_subagent(Path(tmp) / "subagent-memory")
            server = create_server("127.0.0.1", 0, subagent)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                port = int(server.server_address[1])
                client = OSagentMemorySubagentClient(
                    OSagentMemorySubagentConfig(
                        base_url=f"http://127.0.0.1:{port}",
                        top_k=8,
                    )
                )

                remembered = client.manage(
                    "remember",
                    {
                        "content": "The release channel is beta.",
                        "entity": "project",
                        "attribute": "release_channel",
                        "value": "beta",
                    },
                    session_id="manage-one",
                )
                updated = client.manage(
                    "update",
                    {
                        "entity": "project",
                        "attribute": "release_channel",
                        "value": "stable",
                    },
                    session_id="manage-two",
                )
                hits = client.recall("release channel", session_id="manage-read")
                forgotten = client.manage(
                    "forget",
                    {
                        "entity": "project",
                        "attribute": "release_channel",
                    },
                    session_id="manage-three",
                )
                after_forget = client.recall(
                    "release channel stable",
                    session_id="manage-read-two",
                )

                self.assertTrue(remembered["ok"])
                self.assertTrue(updated["ok"])
                self.assertTrue(any("stable" in hit.content for hit in hits))
                self.assertTrue(forgotten["ok"])
                self.assertFalse(
                    any(
                        hit.metadata.get("temporal_status") == "stable"
                        for hit in after_forget
                    )
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
