"""Loopback HTTP service for the native OSagent memory subagent."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from os_agent.subagents.native_memory import LocalMemorySubagent, hit_to_dict

MAX_REQUEST_BYTES = 1024 * 1024


class MemorySubagentHTTPServer(ThreadingHTTPServer):
    """HTTP server carrying one thread-safe memory specialist instance."""

    def __init__(
        self,
        server_address: tuple[str, int],
        subagent: LocalMemorySubagent,
    ) -> None:
        super().__init__(server_address, MemorySubagentRequestHandler)
        self.subagent = subagent


class MemorySubagentRequestHandler(BaseHTTPRequestHandler):
    server: MemorySubagentHTTPServer

    def do_GET(self) -> None:
        if self.path != "/health":
            self._send(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            return
        self._send(HTTPStatus.OK, {"ok": True, "data": self.server.subagent.health()})

    def do_POST(self) -> None:
        try:
            payload = self._read_payload()
            if self.path == "/v1/memory/recall":
                query = str(payload.get("query", "")).strip()
                session_id = str(payload.get("session_id", "")).strip()
                if not query:
                    raise ValueError("query is required")
                hits = self.server.subagent.recall(query, session_id=session_id)
                self._send(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "data": {"hits": [hit_to_dict(hit) for hit in hits]},
                    },
                )
                return
            if self.path == "/v1/memory/observe":
                session_id = str(payload.get("session_id", "")).strip()
                if not session_id:
                    raise ValueError("session_id is required")
                result = self.server.subagent.observe_turn(
                    session_id=session_id,
                    user_input=str(payload.get("user_input", "")),
                    assistant_output=str(payload.get("assistant_output", "")),
                )
                self._send(HTTPStatus.OK, {"ok": True, "data": result})
                return
            if self.path == "/v1/memory/manage":
                action = str(payload.get("action", "")).strip()
                session_id = str(payload.get("session_id", "")).strip()
                arguments = payload.get("arguments", {})
                if not session_id:
                    raise ValueError("session_id is required")
                if not isinstance(arguments, dict):
                    raise ValueError("arguments must be an object")
                result = self.server.subagent.manage(
                    action,
                    arguments,
                    session_id=session_id,
                )
                self._send(HTTPStatus.OK, {"ok": True, "data": result})
                return
            self._send(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
        except (TypeError, ValueError) as exc:
            self._send(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - normalize service failures
            self._send(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
            )

    def _read_payload(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length < 0 or length > MAX_REQUEST_BYTES:
            raise ValueError("request body is too large")
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("request body must be a UTF-8 JSON object") from exc
        if not isinstance(payload, dict):
            raise TypeError("request body must be a JSON object")
        return payload

    def _send(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def create_server(
    host: str,
    port: int,
    subagent: LocalMemorySubagent,
) -> MemorySubagentHTTPServer:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("native memory subagent only supports loopback binding")
    return MemorySubagentHTTPServer((host, port), subagent)


def serve(host: str, port: int, subagent: LocalMemorySubagent) -> None:
    server = create_server(host, port, subagent)
    address, bound_port = server.server_address[:2]
    print(f"OSagent native memory subagent listening on http://{address}:{bound_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
