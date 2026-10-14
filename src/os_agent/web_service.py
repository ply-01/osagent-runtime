"""Authenticated HTTP host for OSagent memory and its MCP endpoint."""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import os
import secrets
from concurrent.futures import Future, ThreadPoolExecutor
from collections.abc import Callable
from threading import RLock
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from os_agent.mcp_server import (
    OSAgentMCPConfig,
    OSAgentMCPService,
    _env_bool,
    _env_int,
    create_mcp_server,
)
from os_agent.plugins.marketplace_ui import PLUGIN_MARKETPLACE_HTML

DEFAULT_PORT = 8765
MAX_REQUEST_BYTES = 1024 * 1024
MAX_MOBILE_PROMPT_CHARS = 16_000
MAX_MOBILE_COMMANDS = 200


class MobileCommandQueueFull(RuntimeError):
    """Raised when the bounded mobile command history cannot accept work."""


class MobileCommandManager:
    """Bounded, authenticated command queue for mobile clients."""

    def __init__(self, service: OSAgentMCPService) -> None:
        workers = _env_int("OS_AGENT_MOBILE_WORKERS", 2)
        if workers < 1 or workers > 8:
            raise ValueError("OS_AGENT_MOBILE_WORKERS must be between 1 and 8")
        self._service = service
        self._executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="osagent-mobile",
        )
        self._lock = RLock()
        self._commands: dict[str, dict[str, Any]] = {}
        self._futures: dict[str, Future[Any]] = {}

    def submit(self, prompt: str, session_id: str = "") -> dict[str, Any]:
        clean_prompt = prompt.strip()
        if not clean_prompt:
            raise ValueError("prompt must not be empty")
        if len(clean_prompt) > MAX_MOBILE_PROMPT_CHARS:
            raise ValueError(
                f"prompt must not exceed {MAX_MOBILE_PROMPT_CHARS} characters"
            )
        command_id = secrets.token_urlsafe(12)
        now = _utc_now_iso()
        record: dict[str, Any] = {
            "command_id": command_id,
            "source": "mobile",
            "status": "queued",
            "prompt_preview": clean_prompt[:160],
            "session_id": session_id.strip(),
            "submitted_at": now,
            "started_at": "",
            "completed_at": "",
            "result": None,
            "error": None,
        }
        with self._lock:
            self._evict_finished()
            if len(self._commands) >= MAX_MOBILE_COMMANDS:
                raise MobileCommandQueueFull("mobile command queue is full")
            self._commands[command_id] = record
            self._futures[command_id] = self._executor.submit(
                self._execute,
                command_id,
                clean_prompt,
                session_id.strip(),
            )
        return self._public(record)

    def get(self, command_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._commands.get(command_id)
            return self._public(record) if record is not None else None

    def cancel(self, command_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._commands.get(command_id)
            if record is None:
                return None
            if record["status"] == "queued":
                future = self._futures.get(command_id)
                if future is not None and future.cancel():
                    record["status"] = "cancelled"
                    record["completed_at"] = _utc_now_iso()
                    return self._public(record)
            return self._public(record)

    def _execute(self, command_id: str, prompt: str, session_id: str) -> None:
        with self._lock:
            record = self._commands.get(command_id)
            if record is None or record["status"] == "cancelled":
                return
            record["status"] = "running"
            record["started_at"] = _utc_now_iso()
        try:
            result = self._service.run_agent(prompt, session_id=session_id)
        except Exception as exc:  # noqa: BLE001 - persist failure for polling clients
            with self._lock:
                record = self._commands.get(command_id)
                if record is not None:
                    record["status"] = "failed"
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    record["completed_at"] = _utc_now_iso()
            return
        with self._lock:
            record = self._commands.get(command_id)
            if record is not None and record["status"] != "cancelled":
                record["status"] = "completed"
                record["result"] = result
                record["completed_at"] = _utc_now_iso()

    def _evict_finished(self) -> None:
        finished = [
            (str(record.get("completed_at", "")), command_id)
            for command_id, record in self._commands.items()
            if record.get("status") in {"completed", "failed", "cancelled"}
        ]
        finished.sort()
        while len(self._commands) >= MAX_MOBILE_COMMANDS and finished:
            _, command_id = finished.pop(0)
            self._commands.pop(command_id, None)
            self._futures.pop(command_id, None)

    @staticmethod
    def _public(record: dict[str, Any] | None) -> dict[str, Any] | None:
        if record is None:
            return None
        return dict(record)


def _utc_now_iso() -> str:
    return _datetime.datetime.now(_datetime.UTC).isoformat()


class BearerAuthMiddleware:
    """Protect every endpoint except the liveness probe with one service key."""

    def __init__(self, app: ASGIApp, api_key: str) -> None:
        self.app = app
        self.api_key = api_key

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http" or scope.get("path") in {"/health", "/plugins", "/plugins/"}:
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        expected = f"Bearer {self.api_key}".encode()
        provided = headers.get(b"authorization", b"")
        if not secrets.compare_digest(provided, expected):
            response = JSONResponse(
                {"ok": False, "error": "unauthorized"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


def create_web_app(
    config: OSAgentMCPConfig,
    *,
    api_key: str,
    host: str = "127.0.0.1",
) -> ASGIApp:
    """Create one process that shares memory across REST and MCP calls."""
    if not api_key.strip():
        raise ValueError("api_key must not be empty")
    service = OSAgentMCPService(config)
    mcp_server = create_mcp_server(config, service=service)
    app = mcp_server.streamable_http_app(
        streamable_http_path="/mcp",
        host=host,
    )
    _add_rest_routes(app, service)
    return BearerAuthMiddleware(app, api_key.strip())


def _add_rest_routes(app: Starlette, service: OSAgentMCPService) -> None:
    async def health(_: Request) -> JSONResponse:
        return JSONResponse({"ok": True, "service": "os-agent-memory"})

    async def capabilities(_: Request) -> JSONResponse:
        return JSONResponse({"ok": True, "data": service.capabilities()})

    async def computer_use_settings(_: Request) -> JSONResponse:
        try:
            return JSONResponse(
                {"ok": True, "data": service.computer_use_settings()}
            )
        except Exception as exc:  # noqa: BLE001 - normalize the HTTP boundary
            return JSONResponse(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                status_code=500,
            )

    async def computer_use_configure(request: Request) -> JSONResponse:
        def run(payload: dict[str, Any]) -> dict[str, Any]:
            enabled = payload.get("enabled")
            mode = payload.get("mode")
            if enabled is None and mode is None:
                raise ValueError("enabled or mode is required")
            return service.configure_computer_use(
                enabled=enabled,
                mode=None if mode is None else str(mode),
            )

        return await _handle_json(request, run)

    async def computer_use_action(request: Request) -> JSONResponse:
        def run(payload: dict[str, Any]) -> dict[str, Any]:
            action = str(payload.get("action", "")).strip().lower()
            if action not in {"launch", "close"}:
                raise ValueError("UI action must be launch or close")
            return service.computer_use(action=action)

        return await _handle_json(request, run)

    async def file_access_settings(_: Request) -> JSONResponse:
        return JSONResponse({"ok": True, "data": service.file_access_settings()})

    async def file_access_configure(request: Request) -> JSONResponse:
        return await _handle_json(
            request,
            lambda payload: service.configure_file_access(
                mode=str(payload.get("mode", "")),
            ),
        )

    async def recall(request: Request) -> JSONResponse:
        return await _handle_json(
            request,
            lambda payload: service.recall(
                query=str(payload.get("query", "")),
                scope=_scope(payload.get("scope", "all"), allow_all=True),
                limit=int(payload.get("limit", 8)),
                entity=str(payload.get("entity", "")),
                attribute=str(payload.get("attribute", "")),
                fact_scope=str(payload.get("fact_scope", "")),
                days=_optional_int(payload.get("days")),
                query_variants=_string_list(payload.get("query_variants")),
                subqueries=_string_list(payload.get("subqueries")),
            ),
        )

    async def observe(request: Request) -> JSONResponse:
        return await _handle_json(
            request,
            lambda payload: service.observe_turn(
                session_id=str(payload.get("session_id", "")),
                user_input=str(payload.get("user_input", "")),
                assistant_output=str(payload.get("assistant_output", "")),
            ),
        )

    async def manage(request: Request) -> JSONResponse:
        def run(payload: dict[str, Any]) -> dict[str, Any]:
            arguments = payload.get("arguments", {})
            if not isinstance(arguments, dict):
                raise TypeError("arguments must be an object")
            return service.manage(
                action=str(payload.get("action", "")),
                arguments=arguments,
                session_id=str(payload.get("session_id", "")),
            )

        return await _handle_json(request, run)

    async def plugin_page(_: Request) -> HTMLResponse:
        return HTMLResponse(PLUGIN_MARKETPLACE_HTML)

    async def plugins(request: Request) -> JSONResponse:
        try:
            return JSONResponse(
                {"ok": True, "data": service.plugin_marketplace(query=request.query_params.get("query", ""))}
            )
        except Exception as exc:  # noqa: BLE001 - normalize the HTTP boundary
            return JSONResponse({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status_code=500)

    async def plugin_install(request: Request) -> JSONResponse:
        name = request.path_params["name"]
        return await _handle_json(
            request,
            lambda _: service.plugin_marketplace(install=name),
        )

    async def plugin_invoke(request: Request) -> JSONResponse:
        name = request.path_params["name"]
        return await _handle_json(
            request,
            lambda payload: service.plugin_marketplace(
                plugin=name,
                action="invoke",
                request=str(payload.get("request", "")),
            ),
        )

    async def plugin_state(request: Request) -> JSONResponse:
        name = request.path_params["name"]
        action = request.path_params["action"]
        return await _handle_json(
            request,
            lambda _: service.plugin_marketplace(plugin=name, action=action),
        )

    async def plugin_uninstall(request: Request) -> JSONResponse:
        name = request.path_params["name"]
        return await _handle_json(
            request,
            lambda _: service.plugin_marketplace(plugin=name, action="uninstall"),
        )

    async def knowledge_upload(request: Request) -> JSONResponse:
        try:
            filename = request.query_params.get("filename", "")
            content = await request.body()
            if len(content) > 10 * 1024 * 1024:
                raise ValueError("knowledge file exceeds the 10 MiB limit")
            return JSONResponse(
                {"ok": True, "data": service.ingest_knowledge_file(filename=filename, content=content)}
            )
        except (TypeError, ValueError, UnicodeError) as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        except PermissionError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=403)
        except Exception as exc:  # noqa: BLE001 - normalize the HTTP boundary
            return JSONResponse({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status_code=500)

    mobile_commands = MobileCommandManager(service)

    async def mobile_submit(request: Request) -> JSONResponse:
        return await _handle_json(
            request,
            lambda payload: mobile_commands.submit(
                prompt=str(payload.get("prompt", payload.get("input", ""))),
                session_id=str(payload.get("session_id", "")),
            ),
            success_status=202,
        )

    async def mobile_status(request: Request) -> JSONResponse:
        command = mobile_commands.get(str(request.path_params["command_id"]))
        if command is None:
            return JSONResponse({"ok": False, "error": "command not found"}, status_code=404)
        return JSONResponse({"ok": True, "data": command})

    async def mobile_cancel(request: Request) -> JSONResponse:
        command = mobile_commands.cancel(str(request.path_params["command_id"]))
        if command is None:
            return JSONResponse({"ok": False, "error": "command not found"}, status_code=404)
        if command["status"] == "running":
            return JSONResponse(
                {"ok": False, "error": "running commands cannot be cancelled", "data": command},
                status_code=409,
            )
        return JSONResponse({"ok": True, "data": command})

    app.add_route("/health", health, methods=["GET"])
    app.add_route("/v1/capabilities", capabilities, methods=["GET"])
    app.add_route("/v1/computer-use", computer_use_settings, methods=["GET"])
    app.add_route("/v1/computer-use", computer_use_configure, methods=["PUT"])
    app.add_route("/v1/computer-use/action", computer_use_action, methods=["POST"])
    app.add_route("/v1/file-access", file_access_settings, methods=["GET"])
    app.add_route("/v1/file-access", file_access_configure, methods=["PUT"])
    app.add_route("/v1/memory/recall", recall, methods=["POST"])
    app.add_route("/v1/memory/observe", observe, methods=["POST"])
    app.add_route("/v1/memory/manage", manage, methods=["POST"])
    app.add_route("/plugins", plugin_page, methods=["GET"])
    app.add_route("/v1/plugins", plugins, methods=["GET"])
    app.add_route("/v1/plugins/{name}/install", plugin_install, methods=["POST"])
    app.add_route("/v1/plugins/{name}/invoke", plugin_invoke, methods=["POST"])
    app.add_route("/v1/plugins/{name}/{action}", plugin_state, methods=["POST"])
    app.add_route("/v1/plugins/{name}", plugin_uninstall, methods=["DELETE"])
    app.add_route("/v1/knowledge/upload", knowledge_upload, methods=["POST"])
    app.add_route("/v1/mobile/commands", mobile_submit, methods=["POST"])
    app.add_route("/v1/mobile/commands/{command_id}", mobile_status, methods=["GET"])
    app.add_route("/v1/mobile/commands/{command_id}", mobile_cancel, methods=["DELETE"])


async def _handle_json(
    request: Request,
    operation: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    success_status: int = 200,
) -> JSONResponse:
    try:
        payload = await _read_json(request)
        return JSONResponse({"ok": True, "data": operation(payload)}, status_code=success_status)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    except PermissionError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=403)
    except MobileCommandQueueFull as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=429)
    except Exception as exc:  # noqa: BLE001 - normalize the HTTP boundary
        return JSONResponse(
            {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
            status_code=500,
        )


async def _read_json(request: Request) -> dict[str, Any]:
    content_length = request.headers.get("content-length")
    if content_length is not None and int(content_length) > MAX_REQUEST_BYTES:
        raise ValueError("request body is too large")
    raw = await request.body()
    if len(raw) > MAX_REQUEST_BYTES:
        raise ValueError("request body is too large")
    payload = json.loads(raw or b"{}")
    if not isinstance(payload, dict):
        raise TypeError("request body must be a JSON object")
    return payload


def _scope(value: Any, *, allow_all: bool) -> Any:
    scope = str(value).strip().lower()
    allowed = {"ltm", "stm", "all"} if allow_all else {"ltm", "stm"}
    if scope not in allowed:
        raise ValueError(f"scope must be one of: {', '.join(sorted(allowed))}")
    return scope


def _optional_int(value: Any) -> int | None:
    return None if value is None or value == "" else int(value)


def _string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise TypeError("query variants and subqueries must be arrays")
    return [str(item) for item in value]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Serve OSagent memory over authenticated REST and MCP.",
    )
    parser.add_argument("--workspace", default=os.getenv("OS_AGENT_MCP_WORKSPACE", "."))
    parser.add_argument(
        "--memory-root",
        default=os.getenv("OS_AGENT_MCP_MEMORY_ROOT", "memory"),
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        default=_env_bool("OS_AGENT_MCP_NO_LLM", False),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=_env_int("OS_AGENT_MCP_MAX_STEPS", 5),
    )
    parser.add_argument(
        "--allow-memory-mutations",
        action="store_true",
        default=_env_bool("OS_AGENT_MCP_ALLOW_MEMORY_MUTATIONS", False),
    )
    parser.add_argument(
        "--allow-workspace-mutations",
        action="store_true",
        default=_env_bool("OS_AGENT_MCP_ALLOW_WORKSPACE_MUTATIONS", False),
    )
    parser.add_argument(
        "--host",
        default=os.getenv("OS_AGENT_MEMORY_HOST", "127.0.0.1"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_env_int("OS_AGENT_MEMORY_PORT", DEFAULT_PORT),
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OS_AGENT_MEMORY_API_KEY", ""),
    )
    args = parser.parse_args(argv)
    config = OSAgentMCPConfig.from_values(
        workspace=args.workspace,
        memory_root=args.memory_root,
        use_llm=not args.no_llm,
        max_steps=args.max_steps,
        allow_memory_mutations=args.allow_memory_mutations,
        allow_workspace_mutations=args.allow_workspace_mutations,
    )
    app = create_web_app(config, api_key=args.api_key, host=args.host)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
