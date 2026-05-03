"""Shared pytest fixtures."""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Hermetic user-corrections path
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def user_corrections_path(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Point ``USER_CORRECTIONS_PATH`` at a per-test tmp path.

    Otherwise, a real file at ``~/.config/podcast_transcript/corrections.toml``
    (left over from manual ``add-correction`` runs by the developer) would
    leak into tests and could change behaviour.
    """
    target = tmp_path_factory.mktemp("user-corrections") / "corrections.toml"
    monkeypatch.setattr("podcast_transcript.cli.USER_CORRECTIONS_PATH", target)
    monkeypatch.setattr("podcast_transcript.corrections_user.USER_CORRECTIONS_PATH", target)
    return target


# ---------------------------------------------------------------------------
# Whisper mocking
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_whisper(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Inject a fake :mod:`whisper` (and ``whisper.utils``) into ``sys.modules``.

    Returns the root mock so tests can configure ``load_model`` / writer
    behaviour and assert on call args.
    """
    whisper_mod = MagicMock(name="whisper")
    utils_mod = MagicMock(name="whisper.utils")
    whisper_mod.utils = utils_mod
    monkeypatch.setitem(sys.modules, "whisper", whisper_mod)
    monkeypatch.setitem(sys.modules, "whisper.utils", utils_mod)
    return whisper_mod


# ---------------------------------------------------------------------------
# Tiny in-process HTTP server for download tests
# ---------------------------------------------------------------------------


# A handler factory yields an HTTPServer-bound handler that returns the
# response described by *responder* — a callable returning
# ``(status, headers, body)``. This lets each test specify its own response
# without subclassing.

ResponderResult = tuple[int, dict[str, str], bytes]
Responder = Callable[[str], ResponderResult]


def _make_handler(responder: Responder) -> type[BaseHTTPRequestHandler]:
    class _Handler(BaseHTTPRequestHandler):
        # Quiet test logs.
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            status, headers, body = responder(self.path)
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

    return _Handler


@pytest.fixture
def http_server() -> Iterator[Callable[[Responder], str]]:
    """Spin up a localhost HTTP server for the duration of a test.

    Yields a function that, given a *responder*, starts the server and returns
    its base URL. The server is shut down on teardown.
    """
    started: list[HTTPServer] = []

    def _start(responder: Responder) -> str:
        server = HTTPServer(("127.0.0.1", 0), _make_handler(responder))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        started.append(server)
        # ``server_address`` is generically typed; for AF_INET it is (host, port).
        host, port = server.server_address[0], server.server_address[1]
        host_str = host.decode() if isinstance(host, bytes) else host
        return f"http://{host_str}:{port}"

    yield _start

    for server in started:
        server.shutdown()
        server.server_close()
