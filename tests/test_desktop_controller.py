import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from os_agent.desktop.controller import DesktopController


class DesktopControllerTests(unittest.TestCase):
    def test_disabled_by_default_and_settings_are_explicit(self) -> None:
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            root = Path(tmp)
            controller = DesktopController(root, policy_path=root / "policy.json")
            settings = controller.settings()
            self.assertFalse(settings["enabled"])
            self.assertEqual(settings["scope"], "full_desktop")
            with self.assertRaises(PermissionError):
                controller.execute("click", x=10, y=20)

    def test_policy_enables_input_without_shell_interpolation(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = root / "policy.json"
            policy.write_text(json.dumps({"enabled": True}), encoding="utf-8")
            controller = DesktopController(root, policy_path=policy)
            with patch.object(controller, "_input_backend", return_value="xdotool"), patch.object(
                controller, "_run"
            ) as run:
                result = controller.execute("click", x=12, y=24)
            self.assertEqual(result["backend"], "xdotool")
            run.assert_called_once_with(
                ["xdotool", "mousemove", "--", "12", "24", "click", "1"]
            )

    def test_coordinates_and_text_are_bounded(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = root / "policy.json"
            policy.write_text(json.dumps({"enabled": True}), encoding="utf-8")
            controller = DesktopController(root, policy_path=policy)
            with patch.object(controller, "_input_backend", return_value="xdotool"):
                with self.assertRaisesRegex(ValueError, "between 0 and 32767"):
                    controller.execute("click", x=-1, y=2)
                with self.assertRaisesRegex(ValueError, "20,000"):
                    controller.execute("type", text="x" * 20_001)

    def test_ydotool_uses_kylin_v11_mouse_syntax(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = root / "policy.json"
            policy.write_text(json.dumps({"enabled": True}), encoding="utf-8")
            controller = DesktopController(root, policy_path=policy)
            with patch.object(controller, "_input_backend", return_value="ydotool"), patch.object(
                controller, "_run"
            ) as run:
                controller.execute("double_click", x=12, y=24)
            self.assertEqual(
                run.call_args_list,
                [
                    unittest.mock.call(["ydotool", "mousemove", "12", "24"]),
                    unittest.mock.call(["ydotool", "click", "1"]),
                    unittest.mock.call(["ydotool", "click", "1"]),
                ],
            )

    def test_move_does_not_emit_an_extra_click(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = root / "policy.json"
            policy.write_text(json.dumps({"enabled": True}), encoding="utf-8")
            controller = DesktopController(root, policy_path=policy)
            with patch.object(controller, "_input_backend", return_value="ydotool"), patch.object(
                controller, "_run"
            ) as run:
                controller.execute("move", x=12, y=24)
            run.assert_called_once_with(["ydotool", "mousemove", "12", "24"])

    def test_screenshot_path_cannot_escape_workspace(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = root / "policy.json"
            policy.write_text(json.dumps({"enabled": True}), encoding="utf-8")
            controller = DesktopController(root, policy_path=policy)
            with patch.object(controller, "_screenshot_backend", return_value="scrot"):
                with self.assertRaisesRegex(ValueError, "inside"):
                    controller.execute("screenshot", screenshot_path="../outside.png")
