import asyncio
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


FILTER_PATH = (
    Path(__file__).resolve().parent.parent
    / "integrations"
    / "open_webui"
    / "osagent_memory_filter.py"
)
SPEC = importlib.util.spec_from_file_location("osagent_memory_filter", FILTER_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
Filter = MODULE.Filter


class OpenWebUIMemoryFilterTests(unittest.TestCase):
    def test_background_tasks_do_not_recall_or_observe(self) -> None:
        memory_filter = Filter()
        memory_filter._request = Mock(side_effect=AssertionError("must not call"))
        body = {
            "messages": [
                {"role": "user", "content": "Generate a title"},
                {"role": "assistant", "content": "A title"},
            ]
        }
        metadata = {"task": "title_generation", "chat_id": "chat-1"}

        inlet_result = asyncio.run(
            memory_filter.inlet(body, __chat_id__="chat-1", __metadata__=metadata)
        )
        outlet_result = asyncio.run(
            memory_filter.outlet(body, __chat_id__="chat-1", __metadata__=metadata)
        )

        self.assertIs(inlet_result, body)
        self.assertIs(outlet_result, body)
        memory_filter._request.assert_not_called()

    def test_inlet_recalls_from_latest_user_message_and_injects_evidence(self) -> None:
        memory_filter = Filter()
        body = {
            "messages": [
                {"role": "user", "content": "Which editor theme do I prefer?"}
            ]
        }
        recalled = {
            "records": [
                {
                    "memory_id": "fact-1",
                    "content": "The user prefers a dark editor theme.",
                }
            ]
        }
        with patch.object(memory_filter, "_request", return_value=recalled) as call:
            result = asyncio.run(memory_filter.inlet(body, __chat_id__="chat-1"))

        self.assertEqual(call.call_args.args[0], "/v1/memory/recall")
        self.assertEqual(
            call.call_args.args[1]["query"],
            "Which editor theme do I prefer?",
        )
        self.assertEqual(result["messages"][0]["role"], "system")
        self.assertIn("untrusted remembered evidence", result["messages"][0]["content"])
        self.assertIn("dark editor theme", result["messages"][0]["content"])

    def test_outlet_observes_completed_turn_without_exposing_an_llm_tool(self) -> None:
        memory_filter = Filter()
        body = {
            "messages": [
                {"role": "user", "content": "Use concise answers."},
                {"role": "assistant", "content": "Understood."},
            ]
        }
        with patch.object(memory_filter, "_request", return_value={}) as call:
            asyncio.run(memory_filter.outlet(body, __chat_id__="chat-2"))

        self.assertEqual(call.call_args.args[0], "/v1/memory/observe")
        self.assertEqual(call.call_args.args[1]["session_id"], "openwebui-chat-2")
        self.assertEqual(call.call_args.args[1]["user_input"], "Use concise answers.")
        self.assertEqual(call.call_args.args[1]["assistant_output"], "Understood.")


if __name__ == "__main__":
    unittest.main()
