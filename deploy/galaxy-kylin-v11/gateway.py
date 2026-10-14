"""Dependency-free Open WebUI reverse proxy with an OSagent UI injection."""

from __future__ import annotations

import http.client
import hmac
import os
import select
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


UPSTREAM_WEBUI = ("127.0.0.1", int(os.getenv("OSAGENT_WEBUI_UPSTREAM_PORT", "10201")))
UPSTREAM_OSAGENT = ("127.0.0.1", int(os.getenv("OSAGENT_SERVICE_PORT", "8765")))
BRIDGE = Path(os.getenv("OSAGENT_MARKETPLACE_BRIDGE", "/opt/osagent-gateway/marketplace-bridge.js"))
OSAGENT_API_KEY = os.getenv("OS_AGENT_MEMORY_API_KEY", "")
MOBILE_API_KEY = os.getenv("OS_AGENT_MOBILE_API_KEY", "")
MAX_BODY = 128 * 1024 * 1024
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade"}


class GatewayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self._is_websocket_upgrade():
            self._proxy_websocket()
            return
        if self.path == "/osagent-marketplace.js":
            self._serve_bridge()
            return
        self._proxy()

    def do_POST(self) -> None:
        self._proxy()

    def do_PUT(self) -> None:
        self._proxy()

    def do_PATCH(self) -> None:
        self._proxy()

    def do_DELETE(self) -> None:
        self._proxy()

    def do_OPTIONS(self) -> None:
        self._proxy()

    def _serve_bridge(self) -> None:
        try:
            content = BRIDGE.read_bytes()
        except OSError:
            self.send_error(404, "marketplace bridge is not installed")
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/javascript; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _proxy(self) -> None:
        parsed = urlsplit(self.path)
        target = UPSTREAM_OSAGENT if parsed.path.startswith("/osagent/") else UPSTREAM_WEBUI
        target_path = parsed.path.removeprefix("/osagent") or "/" if target == UPSTREAM_OSAGENT else self.path
        if parsed.query and target == UPSTREAM_OSAGENT:
            target_path += f"?{parsed.query}"
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length > MAX_BODY:
            self.send_error(413, "request body too large")
            return
        body = self.rfile.read(length) if length else None
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP and key.lower() != "host"
        }
        headers["Host"] = f"{target[0]}:{target[1]}"
        headers["Accept-Encoding"] = "identity"
        if target == UPSTREAM_OSAGENT and parsed.path.startswith("/osagent/v1/mobile/"):
            if not self._mobile_authorized(headers.get("Authorization", "")):
                return
            # The service only trusts its private bearer key. The mobile key is
            # intentionally terminated at this gateway boundary.
            if OSAGENT_API_KEY:
                headers["Authorization"] = f"Bearer {OSAGENT_API_KEY}"
        # The native workspace page is same-origin with Open WebUI and cannot
        # read the OSagent service key. The gateway adds it only for proxied
        # OSagent API calls; direct OSagent access remains bearer-protected.
        if target == UPSTREAM_OSAGENT and OSAGENT_API_KEY and not any(
            key.lower() == "authorization" for key in headers
        ):
            headers["Authorization"] = f"Bearer {OSAGENT_API_KEY}"
        try:
            connection = http.client.HTTPConnection(*target, timeout=120)
            connection.request(self.command, target_path, body=body, headers=headers)
            response = connection.getresponse()
            content_type = response.getheader("Content-Type", "")
            payload = response.read()
            if target == UPSTREAM_WEBUI and "text/html" in content_type:
                payload = payload.replace(
                    b"</head>", b'<script src="/osagent-marketplace.js"></script></head>', 1
                )
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() not in HOP_BY_HOP and key.lower() != "content-length":
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (OSError, http.client.HTTPException) as exc:
            self.send_error(502, f"upstream unavailable: {exc}")
        finally:
            try:
                connection.close()
            except UnboundLocalError:
                pass

    def _is_websocket_upgrade(self) -> bool:
        if urlsplit(self.path).path.startswith("/osagent/"):
            return False
        connection_tokens = {
            token.strip().casefold()
            for token in self.headers.get("Connection", "").split(",")
        }
        return (
            self.headers.get("Upgrade", "").strip().casefold() == "websocket"
            and "upgrade" in connection_tokens
        )

    def _proxy_websocket(self) -> None:
        upstream: socket.socket | None = None
        try:
            upstream = socket.create_connection(UPSTREAM_WEBUI, timeout=10)
            request_line = f"{self.command} {self.path} {self.request_version}\r\n"
            headers = [request_line]
            for key, value in self.headers.items():
                if key.casefold() == "host":
                    continue
                headers.append(f"{key}: {value}\r\n")
            headers.append(f"Host: {UPSTREAM_WEBUI[0]}:{UPSTREAM_WEBUI[1]}\r\n")
            headers.append("\r\n")
            upstream.sendall("".join(headers).encode("iso-8859-1"))

            client = self.connection
            client.settimeout(None)
            upstream.settimeout(None)
            while True:
                readable, _, _ = select.select((client, upstream), (), (), 300)
                if not readable:
                    continue
                for source in readable:
                    payload = source.recv(64 * 1024)
                    if not payload:
                        return
                    destination = upstream if source is client else client
                    destination.sendall(payload)
        except OSError as exc:
            if upstream is None:
                self.send_error(502, f"websocket upstream unavailable: {exc}")
        finally:
            self.close_connection = True
            if upstream is not None:
                upstream.close()

    def _mobile_authorized(self, provided_header: str) -> bool:
        expected = MOBILE_API_KEY or OSAGENT_API_KEY
        provided = provided_header.strip()
        if not expected or not hmac.compare_digest(provided, f"Bearer {expected}"):
            self.send_response(401, "Unauthorized")
            self.send_header("WWW-Authenticate", "Bearer")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return False
        return True

    def log_message(self, format: str, *args: object) -> None:
        print(f"gateway: {format % args}", flush=True)


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", int(os.getenv("OSAGENT_GATEWAY_PORT", "10200"))), GatewayHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
