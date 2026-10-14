from __future__ import annotations

import importlib.util
import socket
import threading
import unittest
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _load_gateway():
    path = (
        Path(__file__).resolve().parents[1]
        / "deploy"
        / "galaxy-kylin-v11"
        / "gateway.py"
    )
    spec = importlib.util.spec_from_file_location("osagent_kylin_gateway", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Galaxy Kylin gateway")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _EchoUpgradeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self.headers.get("Upgrade", "").casefold() != "websocket":
            self.send_error(400)
            return
        self.send_response(101, "Switching Protocols")
        self.send_header("Connection", "Upgrade")
        self.send_header("Upgrade", "websocket")
        self.end_headers()
        payload = self.connection.recv(4)
        self.connection.sendall(payload)

    def log_message(self, format: str, *args: object) -> None:
        pass


class GalaxyKylinGatewayTests(unittest.TestCase):
    def test_osagent_routes_are_not_sent_to_open_webui_websocket_upstream(self) -> None:
        gateway = _load_gateway()
        handler = object.__new__(gateway.GatewayHandler)
        handler.path = "/osagent/v1/events"
        handler.headers = Message()
        handler.headers["Connection"] = "Upgrade"
        handler.headers["Upgrade"] = "websocket"

        self.assertFalse(handler._is_websocket_upgrade())

    def test_websocket_upgrade_is_tunneled_to_open_webui(self) -> None:
        gateway = _load_gateway()
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), _EchoUpgradeHandler)
        gateway.UPSTREAM_WEBUI = upstream.server_address
        proxy = ThreadingHTTPServer(("127.0.0.1", 0), gateway.GatewayHandler)
        threads = [
            threading.Thread(target=upstream.serve_forever, daemon=True),
            threading.Thread(target=proxy.serve_forever, daemon=True),
        ]
        for thread in threads:
            thread.start()

        try:
            with socket.create_connection(proxy.server_address, timeout=3) as client:
                client.sendall(
                    b"GET /ws/socket.io/?EIO=4&transport=websocket HTTP/1.1\r\n"
                    b"Host: example.test\r\n"
                    b"Connection: keep-alive, Upgrade\r\n"
                    b"Upgrade: websocket\r\n"
                    b"Sec-WebSocket-Key: test-key\r\n"
                    b"Sec-WebSocket-Version: 13\r\n\r\n"
                )
                response = b""
                while b"\r\n\r\n" not in response:
                    response += client.recv(4096)
                self.assertIn(b" 101 ", response.split(b"\r\n", 1)[0])
                client.sendall(b"ping")
                self.assertEqual(client.recv(4), b"ping")
        finally:
            proxy.shutdown()
            upstream.shutdown()
            proxy.server_close()
            upstream.server_close()


if __name__ == "__main__":
    unittest.main()
