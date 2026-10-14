"""Constrained Computer Use adapter for the Qianxin trusted browser.

The adapter uses Chromium DevTools Protocol through Playwright when the
browser exposes a debugging endpoint. It never sends arbitrary desktop input
and rejects executables that do not look like the configured Qianxin browser.
"""

from __future__ import annotations

import json
import ipaddress
import os
import shlex
import signal
import shutil
import socket
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path
from threading import RLock
from typing import Any


COMPUTER_USE_MODES = {"allowlist", "public", "unrestricted"}


class QianxinBrowserController:
    def __init__(
        self,
        workspace: Path | str,
        *,
        policy_path: Path | str | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.port = int(os.getenv("OS_AGENT_QIANXIN_CDP_PORT", "9222"))
        self.policy_path = Path(policy_path).resolve() if policy_path else None
        self._policy_lock = RLock()
        self._process: subprocess.Popen[bytes] | None = None
        self._process_group_id: int | None = None
        self._profile_dir: Path | None = None

    @property
    def enabled(self) -> bool:
        policy = self._read_policy()
        value = policy.get("enabled")
        if isinstance(value, bool):
            return value
        return _env_bool("OS_AGENT_COMPUTER_USE_ENABLED", False)

    @property
    def mode(self) -> str:
        policy = self._read_policy()
        value = str(
            policy.get("mode", os.getenv("OS_AGENT_COMPUTER_USE_MODE", "allowlist"))
        ).strip().lower()
        return _normalize_mode(value)

    def settings(self) -> dict[str, Any]:
        status = self.status()
        status.update(
            {
                "modes": [
                    {"id": "allowlist", "available": True},
                    {
                        "id": "public",
                        "available": _public_mode_confirmed(),
                    },
                    {
                        "id": "unrestricted",
                        "available": _unrestricted_mode_confirmed(),
                    },
                ],
                "policy_persistent": self.policy_path is not None,
            }
        )
        return status

    def configure(
        self,
        *,
        enabled: bool | None = None,
        mode: str | None = None,
    ) -> dict[str, Any]:
        if enabled is not None and not isinstance(enabled, bool):
            raise TypeError("enabled must be a boolean")
        target_enabled = self.enabled if enabled is None else enabled
        target_mode = self.mode if mode is None else _normalize_mode(mode)
        if (
            (target_enabled or mode is not None)
            and target_mode == "public"
            and not _public_mode_confirmed()
        ):
            raise PermissionError(
                "public mode is locked by the host policy"
            )
        if (
            (target_enabled or mode is not None)
            and target_mode == "unrestricted"
            and not _unrestricted_mode_confirmed()
        ):
            raise PermissionError(
                "unrestricted mode is locked by the host policy"
            )
        self._write_policy({"enabled": target_enabled, "mode": target_mode})
        if not target_enabled:
            self.close()
        return self.settings()

    def _read_policy(self) -> dict[str, Any]:
        if self.policy_path is None:
            return {}
        with self._policy_lock:
            try:
                payload = json.loads(self.policy_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {}
        return payload if isinstance(payload, dict) else {}

    def _write_policy(self, policy: dict[str, Any]) -> None:
        if self.policy_path is None:
            raise RuntimeError("computer-use policy persistence is not configured")
        with self._policy_lock:
            self.policy_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.policy_path.with_suffix(self.policy_path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(policy, ensure_ascii=True, indent=2) + "\n",
                encoding="utf-8",
            )
            try:
                temporary.chmod(0o600)
            except OSError:
                pass
            os.replace(temporary, self.policy_path)

    @staticmethod
    def _status_timeout() -> float:
        try:
            milliseconds = int(os.getenv("OS_AGENT_QIANXIN_STATUS_TIMEOUT_MS", "350"))
        except ValueError:
            milliseconds = 350
        return max(50, min(milliseconds, 2_000)) / 1_000

    @staticmethod
    def _launch_timeout() -> float:
        try:
            seconds = float(os.getenv("OS_AGENT_QIANXIN_LAUNCH_TIMEOUT_SECONDS", "10"))
        except ValueError:
            seconds = 10.0
        return max(1.0, min(seconds, 30.0))

    def _wait_until_connected(
        self,
        *,
        launcher_may_detach: bool = False,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + self._launch_timeout()
        while time.monotonic() < deadline:
            status = self.status()
            if status["connected"]:
                return status
            if (
                not launcher_may_detach
                and self._process is not None
                and self._process.poll() is not None
            ):
                break
            time.sleep(0.15)
        self.close()
        raise RuntimeError(
            f"Qianxin browser did not expose its CDP endpoint on port {self.port}"
        )

    def _ensure_connected(self, *, initial_url: str = "") -> None:
        if not self.status()["connected"]:
            self.launch(url=initial_url)

    def execute(
        self,
        action: str,
        *,
        url: str = "",
        selector: str = "",
        text: str = "",
        key: str = "",
        x: float | int | None = None,
        y: float | int | None = None,
        button: str = "left",
        repeat: int = 1,
        screenshot_path: str = "browser-screenshot.png",
    ) -> dict[str, Any]:
        normalized = action.strip().lower()
        if normalized == "status":
            return self.settings()
        if normalized == "close":
            return self.close()
        if not self.enabled:
            raise PermissionError(
                "computer use is disabled; enable it in OSagent settings"
            )
        if normalized == "launch":
            return self.launch(url=url)
        if normalized not in {
            "navigate", "click", "fill", "press", "type", "keypress", "screenshot"
        }:
            raise ValueError(
                "action must be status, launch, navigate, click, fill, press, "
                "type, keypress, screenshot, or close"
            )
        self._ensure_connected(initial_url=url if normalized == "navigate" else "")
        return self._with_page(
            normalized,
            url=url,
            selector=selector,
            text=text,
            key=key,
            x=x,
            y=y,
            button=button,
            repeat=repeat,
            screenshot_path=screenshot_path,
        )

    def status(self) -> dict[str, Any]:
        endpoint = f"http://127.0.0.1:{self.port}/json/version"
        try:
            with urllib.request.urlopen(endpoint, timeout=self._status_timeout()) as response:
                version = json.loads(response.read().decode("utf-8"))
            return {
                "enabled": self.enabled,
                "mode": self.mode,
                "connected": True,
                "browser": version.get("Browser", ""),
                "webSocketDebuggerUrl": bool(version.get("webSocketDebuggerUrl")),
                "port": self.port,
            }
        except (OSError, json.JSONDecodeError):
            return {
                "enabled": self.enabled,
                "mode": self.mode,
                "connected": False,
                "port": self.port,
            }

    def launch(self, *, url: str = "") -> dict[str, Any]:
        current = self.status()
        if current["connected"]:
            if url:
                return self._with_page("navigate", url=url)
            return {**current, "launched": False, "reason": "already connected"}
        executable = self._resolve_executable()
        profile = self.workspace / ".qianxin-browser-profile"
        profile.mkdir(parents=True, exist_ok=True)
        self._profile_dir = profile
        uses_kaiming_launcher = _is_kaiming_launcher(executable)
        command = _kaiming_command(executable) if uses_kaiming_launcher else [executable]
        browser_args = [
            f"--remote-debugging-port={self.port}",
            f"--user-data-dir={profile}",
            "--no-first-run",
            "--no-default-browser-check",
        ]
        command.extend(browser_args)
        if url:
            self._check_origin(url)
            command.append(url)
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=(os.name == "posix"),
        )
        if os.name == "posix":
            try:
                self._process_group_id = os.getpgid(self._process.pid)
            except ProcessLookupError:
                self._process_group_id = None
        # Kaiming launches the sandboxed application and then exits. Its short-lived
        # wrapper is not the browser lifecycle signal, so wait for CDP instead.
        status = self._wait_until_connected(
            launcher_may_detach=uses_kaiming_launcher,
        )
        return {
            **status,
            "launched": True,
            "pid": self._process.pid,
            "executable": executable,
        }

    def close(self) -> dict[str, Any]:
        if self._process is not None:
            if os.name == "posix" and self._process_group_id is not None:
                try:
                    os.killpg(self._process_group_id, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            elif self._process.poll() is None:
                self._process.terminate()
            self._process = None
            self._process_group_id = None
            self._terminate_profile_processes()
            self._profile_dir = None
            return {"closed": True}
        return {"closed": False, "reason": "browser process was not launched by OSagent"}

    def _with_page(self, action: str, **kwargs: Any) -> dict[str, Any]:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError("install the optional computer-use dependency: pip install 'os-agent-memory[computer-use]'") from exc
        with sync_playwright() as playwright:
            browser = playwright.chromium.connect_over_cdp(f"http://127.0.0.1:{self.port}")
            try:
                pages = [page for context in browser.contexts for page in context.pages]
                if not pages:
                    raise RuntimeError("Qianxin browser has no controllable page")
                page = _active_page(pages)
                action_timeout = _action_timeout_ms()
                operation_timeout = max(1_000, action_timeout // 3)
                if action == "navigate":
                    self._check_origin(kwargs["url"])
                    page.goto(
                        kwargs["url"],
                        wait_until="domcontentloaded",
                        timeout=action_timeout,
                    )
                elif action == "click":
                    _click_page_target(
                        page,
                        selector=kwargs["selector"],
                        x=kwargs["x"],
                        y=kwargs["y"],
                        button=kwargs["button"],
                        timeout=operation_timeout,
                    )
                elif action == "fill":
                    locator = _editable_locator(page, kwargs["selector"])
                    _retry_locator_action(
                        locator,
                        lambda target: target.fill(
                            kwargs["text"],
                            timeout=operation_timeout,
                        ),
                        timeout=action_timeout,
                    )
                elif action == "press":
                    value = kwargs["key"] or kwargs["text"]
                    if not value.strip():
                        raise ValueError("press requires key or text")
                    if kwargs["selector"].strip():
                        locator = page.locator(kwargs["selector"])
                        _retry_locator_action(
                            locator,
                            lambda target: target.press(
                                value,
                                timeout=operation_timeout,
                            ),
                            timeout=action_timeout,
                        )
                    else:
                        page.keyboard.press(value)
                elif action == "type":
                    if not kwargs["text"]:
                        raise ValueError("type requires non-empty text")
                    if kwargs["selector"].strip():
                        locator = _editable_locator(page, kwargs["selector"])
                        _retry_locator_action(
                            locator,
                            lambda target: target.press_sequentially(
                                kwargs["text"],
                                timeout=operation_timeout,
                            ),
                            timeout=action_timeout,
                        )
                    else:
                        page.keyboard.insert_text(kwargs["text"])
                elif action == "keypress":
                    value = kwargs["key"] or kwargs["text"]
                    if not value.strip():
                        raise ValueError("keypress requires key or text")
                    repeat = int(kwargs["repeat"])
                    if repeat < 1 or repeat > 100:
                        raise ValueError("repeat must be between 1 and 100")
                    for _ in range(repeat):
                        page.keyboard.press(value)
                else:
                    target = _workspace_path(self.workspace, kwargs["screenshot_path"])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    # Dynamic pages can continuously change their document height.
                    # A bounded viewport capture is reliable on Kylin's Chromium 102.
                    page.screenshot(
                        path=str(target),
                        full_page=False,
                        timeout=action_timeout,
                    )
                    return {"screenshot": str(target)}
                return {"action": action, "url": page.url, "title": page.title()}
            finally:
                browser.close()


    def _resolve_executable(self) -> str:
        explicit = os.getenv("OS_AGENT_QIANXIN_BROWSER_EXECUTABLE", "").strip()
        candidates = [explicit] if explicit else []
        for directory in (
            Path.home() / ".local/share/applications",
            Path("/usr/share/applications"),
            Path("/opt/kaiming/share/applications"),
        ):
            if directory.is_dir():
                for desktop in directory.glob("*.desktop"):
                    try:
                        content = desktop.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        continue
                    if any(token in content.casefold() for token in ("qianxin", "qax", "可信浏览器")):
                        for line in content.splitlines():
                            if line.startswith("X-Kaiming="):
                                app_id = line.split("=", 1)[1].strip()
                                if app_id:
                                    candidates.append(f"/opt/kaiming/bin/{app_id}")
                        for line in content.splitlines():
                            if line.startswith("Exec="):
                                try:
                                    command = shlex.split(line[5:])
                                except ValueError:
                                    continue
                                if command:
                                    candidates.append(command[0])
        for candidate in candidates:
            resolved = shutil.which(candidate) or (candidate if Path(candidate).is_file() else "")
            if resolved and _looks_like_qianxin(resolved):
                return resolved
        raise FileNotFoundError("Qianxin trusted browser executable was not found; set OS_AGENT_QIANXIN_BROWSER_EXECUTABLE")

    def _check_origin(self, url: str) -> None:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("browser URL must use http or https")
        allowed = [item.strip().rstrip("/") for item in os.getenv("OS_AGENT_BROWSER_ALLOWED_ORIGINS", "").split(",") if item.strip()]
        origin = f"{parsed.scheme}://{parsed.netloc}"
        mode = self.mode
        if mode == "unrestricted" and not _unrestricted_mode_confirmed():
            raise PermissionError(
                "unrestricted mode requires OS_AGENT_COMPUTER_USE_UNRESTRICTED_CONFIRM=I_UNDERSTAND_UNRESTRICTED_WEB"
            )
        # Match exact origins only; prefix matching would allow a different host.
        if origin in allowed:
            return
        if mode == "public":
            if not _public_mode_confirmed():
                raise PermissionError(
                    "public mode requires OS_AGENT_COMPUTER_USE_PUBLIC_CONFIRM=I_UNDERSTAND_PUBLIC_WEB"
                )
            if _is_public_host(parsed.hostname or ""):
                return
            raise PermissionError(
                f"public mode blocks private or unresolved browser host: {parsed.hostname}"
            )
        if mode == "unrestricted":
            return
        if not allowed:
            raise PermissionError(f"browser origin is not in OS_AGENT_BROWSER_ALLOWED_ORIGINS: {origin}")
        raise PermissionError(f"browser origin is not allowed in allowlist mode: {origin}")

    def _terminate_profile_processes(self) -> None:
        if os.name != "posix" or self._profile_dir is None:
            return
        marker = f"--user-data-dir={self._profile_dir}"
        proc_root = Path("/proc")
        for entry in proc_root.glob("[0-9]*"):
            try:
                pid = int(entry.name)
                if pid == os.getpid():
                    continue
                command_line = (entry / "cmdline").read_bytes().decode("utf-8", errors="ignore")
            except (OSError, ValueError):
                continue
            if marker in command_line:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass


def _active_page(pages: list[Any]) -> Any:
    """Choose the most recently usable page without assuming one context."""
    usable = [
        page
        for page in pages
        if str(getattr(page, "url", "")) not in {"", "about:blank"}
    ]
    return (usable or pages)[-1]


def _editable_locator(page: Any, selector: str) -> Any:
    """Resolve an explicit selector or the currently visible editable control."""
    clean = selector.strip()
    if clean:
        return page.locator(clean)
    candidates = page.locator(
        "textarea:visible, input:not([type=hidden]):visible, "
        "[contenteditable='true']:visible"
    )
    if candidates.count() == 0:
        raise ValueError(
            "fill or type needs a selector when the page has no visible editable control"
        )
    return candidates.last


def _retry_locator_action(locator: Any, operation: Any, *, timeout: int) -> None:
    """Retry a locator operation when a reactive page replaces the node mid-action."""
    # A locator is re-evaluated by Playwright on each operation. Retrying the
    # operation handles React-style detach/reinsert cycles without force-clicking
    # an unrelated element or injecting page-specific JavaScript.
    attempts = 3
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            operation(locator)
            return
        except Exception as exc:  # Playwright is an optional dependency here.
            last_error = exc
            time.sleep(0.15)
    assert last_error is not None
    raise last_error


def _click_page_target(
    page: Any,
    *,
    selector: str,
    x: Any,
    y: Any,
    button: str,
    timeout: int,
) -> None:
    """Click by screenshot coordinates or a DOM locator, with a clear contract."""
    if x is not None or y is not None:
        if x is None or y is None:
            raise ValueError("browser click coordinates require both x and y")
        try:
            point_x = float(x)
            point_y = float(y)
        except (TypeError, ValueError) as exc:
            raise ValueError("browser click coordinates must be finite numbers") from exc
        if not (0 <= point_x <= 32767 and 0 <= point_y <= 32767):
            raise ValueError("browser click coordinates must be between 0 and 32767")
        clean_button = str(button or "left").strip().lower()
        if clean_button not in {"left", "middle", "right"}:
            raise ValueError("browser click button must be left, middle, or right")
        page.mouse.click(point_x, point_y, button=clean_button, delay=30)
        return
    clean = selector.strip()
    if not clean:
        raise ValueError("browser click needs a selector or screenshot coordinates")
    locator = page.locator(clean)
    _retry_locator_action(
        locator,
        lambda target: target.click(timeout=timeout),
        timeout=timeout,
    )


def _workspace_path(workspace: Path, value: str) -> Path:
    """Resolve a screenshot path without allowing it to escape the workspace."""
    clean = str(value or "").strip()
    if not clean:
        raise ValueError("screenshot_path must not be empty")
    target = (workspace / clean).resolve()
    if target != workspace and workspace not in target.parents:
        raise ValueError("screenshot_path must stay inside the OSagent workspace")
    return target


def _is_kaiming_launcher(path: str) -> bool:
    normalized = path.replace("\\", "/").casefold()
    return "/kaiming/" in normalized or "top.openkylin.qaxbrowser" in normalized


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _action_timeout_ms() -> int:
    try:
        milliseconds = int(os.getenv("OS_AGENT_QIANXIN_ACTION_TIMEOUT_MS", "8000"))
    except ValueError:
        milliseconds = 8_000
    return max(1_000, min(milliseconds, 30_000))


def _normalize_mode(value: str) -> str:
    normalized_value = str(value).strip().lower()
    aliases = {
        "safe": "allowlist",
        "restricted": "allowlist",
        "open": "unrestricted",
    }
    normalized = aliases.get(normalized_value, normalized_value)
    if normalized not in COMPUTER_USE_MODES:
        raise ValueError("computer-use mode must be allowlist, public, or unrestricted")
    return normalized


def _public_mode_confirmed() -> bool:
    return (
        os.getenv("OS_AGENT_COMPUTER_USE_PUBLIC_CONFIRM", "")
        == "I_UNDERSTAND_PUBLIC_WEB"
    )


def _unrestricted_mode_confirmed() -> bool:
    return (
        os.getenv("OS_AGENT_COMPUTER_USE_UNRESTRICTED_CONFIRM", "")
        == "I_UNDERSTAND_UNRESTRICTED_WEB"
    )


def _kaiming_command(executable: str) -> list[str]:
    app_id = Path(executable).name
    command_name = app_id.removeprefix("top.openkylin.")
    return [
        "/opt/kaiming-tools/bin/kaiming",
        "run",
        f"--command=/usr/bin/{command_name}",
        app_id,
        "--",
    ]


def _is_public_host(hostname: str) -> bool:
    normalized = hostname.rstrip(".").casefold()
    if normalized in {"localhost", "localhost.localdomain"} or normalized.endswith(".localhost"):
        return False
    try:
        addresses = {
            ipaddress.ip_address(info[4][0])
            for info in socket.getaddrinfo(normalized, None, type=socket.SOCK_STREAM)
        }
    except (OSError, ValueError):
        return False
    return bool(addresses) and all(address.is_global for address in addresses)


def _looks_like_qianxin(path: str) -> bool:
    value = Path(path).name.casefold()
    return any(token in value for token in ("qianxin", "qax", "trusted", "可信"))
