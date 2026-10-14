from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from os_agent.browser.qianxin import (
    QianxinBrowserController,
    _click_page_target,
    _editable_locator,
    _kaiming_command,
    _looks_like_qianxin,
    _workspace_path,
)


class QianxinBrowserTests(unittest.TestCase):
    def test_executable_filter_is_specific(self) -> None:
        self.assertTrue(_looks_like_qianxin("/usr/bin/qianxin-browser"))
        self.assertFalse(_looks_like_qianxin("/usr/bin/firefox"))

    def test_kare_launcher_uses_desktop_entry_shape(self) -> None:
        self.assertEqual(
            _kaiming_command("/opt/kaiming/bin/top.openkylin.qaxbrowser-safe-stable"),
            [
                "/opt/kaiming-tools/bin/kaiming",
                "run",
                "--command=/usr/bin/qaxbrowser-safe-stable",
                "top.openkylin.qaxbrowser-safe-stable",
                "--",
            ],
        )

    def test_computer_use_is_disabled_by_default(self) -> None:
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            controller = QianxinBrowserController(Path(tmp))
            status = controller.execute("status")
            self.assertFalse(status["enabled"])
            with self.assertRaises(PermissionError):
                controller.execute("navigate", url="https://example.com")

    def test_runtime_policy_is_persisted_without_changing_environment(self) -> None:
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            root = Path(tmp)
            policy_path = root / "data" / "computer-use.json"
            controller = QianxinBrowserController(root, policy_path=policy_path)

            enabled = controller.configure(enabled=True, mode="allowlist")
            restarted = QianxinBrowserController(root, policy_path=policy_path)

            self.assertTrue(enabled["enabled"])
            self.assertTrue(restarted.enabled)
            self.assertEqual(restarted.mode, "allowlist")
            self.assertTrue(policy_path.exists())

    def test_kaiming_launcher_exit_does_not_abort_cdp_wait(self) -> None:
        with TemporaryDirectory() as tmp:
            controller = QianxinBrowserController(Path(tmp))
            controller._process = Mock()
            controller._process.poll.return_value = 0
            expected = {"connected": True, "port": controller.port}
            with patch.object(
                controller,
                "status",
                side_effect=[{"connected": False}, expected],
            ), patch("os_agent.browser.qianxin.time.sleep"):
                actual = controller._wait_until_connected(launcher_may_detach=True)

            self.assertEqual(actual, expected)
            self.assertEqual(controller._process.poll.call_count, 0)

    def test_press_action_is_forwarded_to_the_active_page(self) -> None:
        with TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"OS_AGENT_COMPUTER_USE_ENABLED": "true"},
            clear=True,
        ):
            controller = QianxinBrowserController(Path(tmp))
            expected = {"action": "press", "url": "https://example.com"}
            with patch.object(controller, "_ensure_connected"), patch.object(
                controller,
                "_with_page",
                return_value=expected,
            ) as with_page:
                actual = controller.execute(
                    "press",
                    selector="textarea:visible",
                    text="Enter",
                )

            self.assertEqual(actual, expected)
            with_page.assert_called_once_with(
                "press",
                url="",
                selector="textarea:visible",
                text="Enter",
                key="",
                x=None,
                y=None,
                button="left",
                repeat=1,
                screenshot_path="browser-screenshot.png",
            )

    def test_browser_click_uses_screenshot_coordinates(self) -> None:
        class Mouse:
            def __init__(self) -> None:
                self.calls = []

            def click(self, x, y, **kwargs):
                self.calls.append((x, y, kwargs))

        class Page:
            def __init__(self) -> None:
                self.mouse = Mouse()

        page = Page()
        _click_page_target(
            page,
            selector="",
            x=120,
            y=240,
            button="left",
            timeout=1000,
        )
        self.assertEqual(page.mouse.calls, [(120.0, 240.0, {"button": "left", "delay": 30})])

    def test_empty_selector_resolves_visible_editable_control(self) -> None:
        class CandidateList:
            def count(self) -> int:
                return 2

            @property
            def last(self):
                return "last-visible-editor"

        class Page:
            def locator(self, selector):
                self.selector = selector
                return CandidateList()

        page = Page()
        self.assertEqual(_editable_locator(page, ""), "last-visible-editor")
        self.assertIn("[contenteditable='true']:visible", page.selector)

    def test_screenshot_path_cannot_escape_workspace(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "inside"):
                _workspace_path(root, "../outside.png")

    def test_runtime_policy_cannot_unlock_public_mode(self) -> None:
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            root = Path(tmp)
            controller = QianxinBrowserController(
                root,
                policy_path=root / "computer-use.json",
            )
            with self.assertRaises(PermissionError):
                controller.configure(enabled=True, mode="public")

    def test_origin_allowlist_is_required(self) -> None:
        with TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"OS_AGENT_COMPUTER_USE_ENABLED": "true"},
            clear=True,
        ):
            controller = QianxinBrowserController(Path(tmp))
            with self.assertRaises(PermissionError):
                controller._check_origin("https://example.com")

    def test_origin_allowlist_does_not_match_host_prefix(self) -> None:
        with TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "OS_AGENT_COMPUTER_USE_ENABLED": "true",
                "OS_AGENT_BROWSER_ALLOWED_ORIGINS": "https://intranet.example",
            },
            clear=True,
        ):
            controller = QianxinBrowserController(Path(tmp))
            controller._check_origin("https://intranet.example/path")
            with self.assertRaises(PermissionError):
                controller._check_origin("https://intranet.example.com/path")

    def test_public_mode_allows_public_hosts_after_confirmation(self) -> None:
        with TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "OS_AGENT_COMPUTER_USE_ENABLED": "true",
                "OS_AGENT_COMPUTER_USE_MODE": "public",
                "OS_AGENT_COMPUTER_USE_PUBLIC_CONFIRM": "I_UNDERSTAND_PUBLIC_WEB",
            },
            clear=True,
        ), patch(
            "socket.getaddrinfo",
            return_value=[(2, 1, 6, "", ("93.184.216.34", 443))],
        ):
            controller = QianxinBrowserController(Path(tmp))
            controller._check_origin("https://example.com/path")

    def test_public_mode_blocks_private_hosts(self) -> None:
        with TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "OS_AGENT_COMPUTER_USE_ENABLED": "true",
                "OS_AGENT_COMPUTER_USE_MODE": "public",
                "OS_AGENT_COMPUTER_USE_PUBLIC_CONFIRM": "I_UNDERSTAND_PUBLIC_WEB",
            },
            clear=True,
        ), patch(
            "socket.getaddrinfo",
            return_value=[(2, 1, 6, "", ("127.0.0.1", 80))],
        ):
            controller = QianxinBrowserController(Path(tmp))
            with self.assertRaises(PermissionError):
                controller._check_origin("http://private.example")

    def test_unrestricted_mode_requires_confirmation(self) -> None:
        with TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "OS_AGENT_COMPUTER_USE_ENABLED": "true",
                "OS_AGENT_COMPUTER_USE_MODE": "unrestricted",
            },
            clear=True,
        ):
            controller = QianxinBrowserController(Path(tmp))
            with self.assertRaises(PermissionError):
                controller._check_origin("http://127.0.0.1:9000")


if __name__ == "__main__":
    unittest.main()
