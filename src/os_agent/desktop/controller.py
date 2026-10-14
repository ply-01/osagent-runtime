"""Controlled desktop input for a logged-in Linux graphical session.

The browser adapter operates through CDP and is intentionally page-scoped. This
module is the separate desktop layer used for browser chrome, native dialogs,
and other applications. It uses existing session tools (xdotool or ydotool)
and never runs a shell command constructed from model input.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import struct
import subprocess
from pathlib import Path
from threading import RLock
from typing import Any


DESKTOP_ACTIONS = {
    "screenshot",
    "move",
    "click",
    "double_click",
    "right_click",
    "middle_click",
    "drag",
    "scroll",
    "keypress",
    "type",
}
_KEY_RE = re.compile(r"^[A-Za-z0-9_+\-:.]+$")
_BUTTONS = {"left": "1", "middle": "2", "right": "3"}
# Kylin V11 ships ydotool 0.1.x, whose click command uses these button ids.
_YDOTOOL_BUTTONS = {"left": "1", "middle": "3", "right": "2"}
_YDOTOOL_KEYS = {
    "ESC": 1,
    "ESCAPE": 1,
    "TAB": 15,
    "ENTER": 28,
    "RETURN": 28,
    "BACKSPACE": 14,
    "SPACE": 57,
    "DELETE": 111,
    "INSERT": 110,
    "HOME": 102,
    "END": 107,
    "PAGEUP": 104,
    "PAGEDOWN": 109,
    "UP": 103,
    "ARROWUP": 103,
    "DOWN": 108,
    "ARROWDOWN": 108,
    "LEFT": 105,
    "ARROWLEFT": 105,
    "RIGHT": 106,
    "ARROWRIGHT": 106,
    "CTRL": 29,
    "CONTROL": 29,
    "ALT": 56,
    "SHIFT": 42,
    "SUPER": 125,
    "META": 125,
}
_YDOTOOL_KEYS.update({f"F{index}": 58 + index for index in range(1, 13)})


class DesktopController:
    """Execute bounded desktop actions in the current user's GUI session."""

    def __init__(
        self,
        workspace: Path | str,
        *,
        policy_path: Path | str | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.policy_path = Path(policy_path).resolve() if policy_path else None
        self._policy_lock = RLock()

    @property
    def enabled(self) -> bool:
        policy = self._read_policy()
        value = policy.get("enabled")
        if isinstance(value, bool):
            return value
        return _env_bool("OS_AGENT_COMPUTER_USE_ENABLED", False)

    def settings(self) -> dict[str, Any]:
        input_backend = self._input_backend()
        screenshot_backend = self._screenshot_backend()
        display = os.getenv("DISPLAY", "").strip()
        wayland_display = os.getenv("WAYLAND_DISPLAY", "").strip()
        return {
            "enabled": self.enabled,
            "available": bool(input_backend or screenshot_backend),
            "connected": bool(input_backend and (display or wayland_display or _uinput_available())),
            "input_backend": input_backend or "none",
            "screenshot_backend": screenshot_backend or "none",
            "display": display,
            "wayland_display": wayland_display,
            "actions": sorted(DESKTOP_ACTIONS),
            "scope": "full_desktop",
        }

    def execute(
        self,
        action: str,
        *,
        x: float | int | None = None,
        y: float | int | None = None,
        text: str = "",
        key: str = "",
        button: str = "left",
        start_x: float | int | None = None,
        start_y: float | int | None = None,
        end_x: float | int | None = None,
        end_y: float | int | None = None,
        path: list[list[float | int]] | None = None,
        delta_x: float | int = 0,
        delta_y: float | int = 0,
        repeat: int = 1,
        screenshot_path: str = "desktop-screenshot.png",
    ) -> dict[str, Any]:
        normalized = action.strip().lower()
        if normalized == "status":
            return self.settings()
        if normalized not in DESKTOP_ACTIONS:
            raise ValueError(
                "desktop action must be screenshot, move, click, double_click, "
                "right_click, middle_click, drag, scroll, keypress, or type"
            )
        if not self.enabled:
            raise PermissionError(
                "desktop computer use is disabled; enable it in the OSagent control dialog"
            )
        if normalized == "screenshot":
            return self._screenshot(screenshot_path)

        backend = self._input_backend()
        if not backend:
            raise RuntimeError(
                "no desktop input backend is available; the graphical session must expose "
                "DISPLAY or Wayland and xdotool/ydotool"
            )
        if normalized in {"move", "click", "double_click", "right_click", "middle_click"}:
            point = _point(x, y, name="x/y")
            self._mouse_action(normalized, point[0], point[1], backend)
            return {"action": normalized, "x": point[0], "y": point[1], "backend": backend}
        if normalized == "drag":
            points = _drag_points(start_x, start_y, end_x, end_y, path)
            self._drag(points, backend)
            return {"action": normalized, "path": points, "backend": backend}
        if normalized == "scroll":
            dx = _finite_number(delta_x, "delta_x")
            dy = _finite_number(delta_y, "delta_y")
            if dx == 0 and dy == 0:
                raise ValueError("scroll requires a non-zero delta_x or delta_y")
            point = _point(x, y, name="x/y", allow_missing=True)
            self._scroll(dx, dy, point, backend)
            return {
                "action": normalized,
                "delta_x": dx,
                "delta_y": dy,
                "x": point[0] if point else None,
                "y": point[1] if point else None,
                "backend": backend,
            }
        if normalized == "type":
            if not isinstance(text, str) or not text:
                raise ValueError("type requires non-empty text")
            if len(text) > 20_000:
                raise ValueError("type text exceeds the 20,000 character limit")
            self._type(text, backend)
            return {"action": normalized, "characters": len(text), "backend": backend}

        value = key or text
        if not isinstance(value, str) or not value.strip():
            raise ValueError("keypress requires key or text")
        _validate_key(value)
        if repeat < 1 or repeat > 100:
            raise ValueError("repeat must be between 1 and 100")
        self._keypress(value, repeat, backend)
        return {"action": normalized, "key": value, "repeat": repeat, "backend": backend}

    def _read_policy(self) -> dict[str, Any]:
        if self.policy_path is None:
            return {}
        with self._policy_lock:
            try:
                payload = json.loads(self.policy_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {}
        return payload if isinstance(payload, dict) else {}

    def _input_backend(self) -> str:
        configured = os.getenv("OS_AGENT_DESKTOP_INPUT_BACKEND", "").strip().lower()
        if configured in {"none", "disabled"}:
            return ""
        if configured in {"xdotool", "ydotool"}:
            return configured if shutil.which(configured) else ""
        if os.getenv("DISPLAY", "").strip() and shutil.which("xdotool"):
            return "xdotool"
        if shutil.which("ydotool") and _uinput_available():
            return "ydotool"
        return ""

    def _screenshot_backend(self) -> str:
        if os.getenv("DISPLAY", "").strip():
            for command in ("gnome-screenshot", "import", "scrot"):
                if shutil.which(command):
                    return command
        if os.getenv("WAYLAND_DISPLAY", "").strip() or _wayland_socket_exists():
            for command in ("grim", "gnome-screenshot", "spectacle"):
                if shutil.which(command):
                    return command
        return ""

    def _screenshot(self, screenshot_path: str) -> dict[str, Any]:
        backend = self._screenshot_backend()
        if not backend:
            raise RuntimeError("no desktop screenshot backend is available")
        target = _workspace_path(self.workspace, screenshot_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if backend == "gnome-screenshot":
            command = [backend, "-f", str(target)]
        elif backend == "import":
            command = [backend, "-window", "root", str(target)]
        else:
            command = [backend, str(target)]
        self._run(command)
        width, height = _png_dimensions(target)
        return {
            "action": "screenshot",
            "screenshot": str(target),
            "width": width,
            "height": height,
            "backend": backend,
        }

    def _mouse_action(self, action: str, x: int, y: int, backend: str) -> None:
        if backend == "xdotool":
            if action == "move":
                self._run(["xdotool", "mousemove", "--", str(x), str(y)])
                return
            button = {"click": "left", "double_click": "left", "right_click": "right", "middle_click": "middle"}[action]
            command = ["xdotool", "mousemove", "--", str(x), str(y), "click"]
            if action == "double_click":
                command.extend(["--repeat", "2"])
            command.append(_BUTTONS[button])
            self._run(command)
            return
        self._run(["ydotool", "mousemove", str(x), str(y)])
        if action == "move":
            return
        button = "left" if action in {"click", "double_click"} else action.removesuffix("_click")
        command = ["ydotool", "click", _YDOTOOL_BUTTONS[button]]
        if action == "double_click":
            self._run(command)
            self._run(command)
            return
        self._run(command)

    def _drag(self, points: list[list[int]], backend: str) -> None:
        if backend == "xdotool":
            first = points[0]
            self._run(["xdotool", "mousemove", "--", str(first[0]), str(first[1]), "mousedown", "1"])
            for point in points[1:]:
                self._run(["xdotool", "mousemove", "--", str(point[0]), str(point[1])])
            self._run(["xdotool", "mouseup", "1"])
            return
        first = points[0]
        self._run(["ydotool", "mousemove", str(first[0]), str(first[1])])
        self._run(["ydotool", "click", "0x40"])
        for point in points[1:]:
            self._run(["ydotool", "mousemove", str(point[0]), str(point[1])])
        self._run(["ydotool", "click", "0x80"])

    def _scroll(self, delta_x: float, delta_y: float, point: list[int] | None, backend: str) -> None:
        if backend != "xdotool":
            raise RuntimeError("scroll requires the xdotool desktop backend")
        if point:
            self._run(["xdotool", "mousemove", "--", str(point[0]), str(point[1])])
        if delta_y:
            button = "4" if delta_y < 0 else "5"
            count = min(max(int(abs(delta_y) / 100), 1), 100)
            self._run(["xdotool", "click", "--repeat", str(count), button])
        if delta_x:
            button = "6" if delta_x < 0 else "7"
            count = min(max(int(abs(delta_x) / 100), 1), 100)
            self._run(["xdotool", "click", "--repeat", str(count), button])

    def _type(self, text: str, backend: str) -> None:
        if backend == "xdotool":
            self._run(["xdotool", "type", "--clearmodifiers", "--delay", "0", "--", text])
        else:
            self._run(["ydotool", "type", "--delay", "0", "--", text])

    def _keypress(self, key: str, repeat: int, backend: str) -> None:
        for _ in range(repeat):
            if backend == "xdotool":
                self._run(["xdotool", "key", "--clearmodifiers", "--", key])
            else:
                self._run(["ydotool", "key", *_ydotool_key_events(key)])

    def _run(self, command: list[str]) -> None:
        try:
            completed = subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=_command_timeout_seconds(),
                env=os.environ.copy(),
            )
            # Some distro builds of ydotool return zero even when their parser
            # prints an error. Treat that output as a failed input event rather
            # than reporting a false-positive tool result.
            detail = (completed.stderr or "").strip()
            if re.search(r"(?:^|\n).*\b(?:error|failed|unrecognised)\b", detail, re.IGNORECASE):
                raise RuntimeError(f"desktop action failed ({command[0]}): {detail[-500:]}")
        except FileNotFoundError as exc:
            raise RuntimeError(f"desktop backend command is unavailable: {command[0]}") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or "").strip()[-500:]
            raise RuntimeError(f"desktop action failed ({command[0]}): {detail or 'unknown error'}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"desktop action timed out ({command[0]})") from exc


def _point(x: float | int | None, y: float | int | None, *, name: str, allow_missing: bool = False) -> list[int] | None:
    if x is None and y is None and allow_missing:
        return None
    if x is None or y is None:
        raise ValueError(f"{name} requires both coordinates")
    return [_coordinate(x, "x"), _coordinate(y, "y")]


def _drag_points(
    start_x: float | int | None,
    start_y: float | int | None,
    end_x: float | int | None,
    end_y: float | int | None,
    path: list[list[float | int]] | None,
) -> list[list[int]]:
    if path is not None:
        if not isinstance(path, list) or len(path) < 2:
            raise ValueError("drag path must contain at least two points")
        points = []
        for item in path:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError("each drag path point must contain x and y")
            points.append([_coordinate(item[0], "x"), _coordinate(item[1], "y")])
        return points
    first = _point(start_x, start_y, name="start_x/start_y")
    last = _point(end_x, end_y, name="end_x/end_y")
    assert first is not None and last is not None
    return [first, last]


def _coordinate(value: Any, name: str) -> int:
    number = _finite_number(value, name)
    if number < 0 or number > 32767:
        raise ValueError(f"{name} must be between 0 and 32767")
    return int(round(number))


def _finite_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _validate_key(value: str) -> None:
    clean = value.strip()
    if len(clean) > 120 or not _KEY_RE.fullmatch(clean) or clean.startswith("-"):
        raise ValueError("key contains unsupported characters")


def _ydotool_key_events(value: str) -> list[str]:
    parts = [part for part in re.split(r"[+ ]+", value.strip()) if part]
    if not parts:
        raise ValueError("keypress key must not be empty")
    codes = []
    for part in parts:
        upper = part.upper()
        code = _YDOTOOL_KEYS.get(upper)
        if code is None and len(part) == 1 and part.isprintable():
            code = ord(part.upper()) - ord("A") + 30 if part.isalpha() else None
        if code is None:
            raise ValueError(f"ydotool key is not mapped: {part}")
        codes.append(code)
    modifiers = codes[:-1]
    main = codes[-1]
    events = [f"{code}:1" for code in modifiers]
    events.extend([f"{main}:1", f"{main}:0"])
    events.extend(f"{code}:0" for code in reversed(modifiers))
    return events


def _workspace_path(workspace: Path, value: str) -> Path:
    clean = str(value or "").strip()
    if not clean:
        raise ValueError("screenshot_path must not be empty")
    target = (workspace / clean).resolve()
    if target != workspace and workspace not in target.parents:
        raise ValueError("screenshot_path must stay inside the OSagent workspace")
    return target


def _png_dimensions(path: Path) -> tuple[int | None, int | None]:
    try:
        with path.open("rb") as handle:
            if handle.read(8) != b"\x89PNG\r\n\x1a\n":
                return None, None
            if handle.read(4) != b"\x00\x00\x00\r" or handle.read(4) != b"IHDR":
                return None, None
            width, height = struct.unpack(">II", handle.read(8))
            return width, height
    except (OSError, struct.error):
        return None, None


def _uinput_available() -> bool:
    try:
        return os.access("/dev/uinput", os.W_OK)
    except OSError:
        return False


def _wayland_socket_exists() -> bool:
    runtime = os.getenv("XDG_RUNTIME_DIR", "").strip()
    display = os.getenv("WAYLAND_DISPLAY", "wayland-0").strip()
    return bool(runtime and Path(runtime, display).exists())


def _command_timeout_seconds() -> float:
    try:
        value = float(os.getenv("OS_AGENT_DESKTOP_ACTION_TIMEOUT_SECONDS", "10"))
    except ValueError:
        value = 10.0
    return max(1.0, min(value, 60.0))


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
