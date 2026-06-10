"""Tests for :mod:`podcast_transcript.download`."""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import pytest

from podcast_transcript.download import (
    DownloadError,
    UnexpectedContentTypeError,
    download_podcast,
    read_capped,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from .conftest import Responder


AUDIO_BODY = b"ID3\x04\x00" + b"\x00" * 1024  # 1 KB of fake MP3 bytes


def _audio_responder(body: bytes = AUDIO_BODY, content_type: str = "audio/mpeg") -> Responder:
    def respond(_path: str) -> tuple[int, dict[str, str], bytes]:
        return (
            200,
            {"Content-Type": content_type, "Content-Length": str(len(body))},
            body,
        )

    return respond


def test_download_writes_file_atomically(
    tmp_path: Path, http_server: Callable[[Responder], str]
) -> None:
    base_url = http_server(_audio_responder())
    output = tmp_path / "episode.mp3"

    result = download_podcast(f"{base_url}/episode.mp3", output)

    assert result == output
    assert output.read_bytes() == AUDIO_BODY
    # No leftover .part file.
    assert not output.with_suffix(".mp3.part").exists()


def test_download_creates_parent_directories(
    tmp_path: Path, http_server: Callable[[Responder], str]
) -> None:
    base_url = http_server(_audio_responder())
    output = tmp_path / "nested" / "dir" / "episode.mp3"

    download_podcast(f"{base_url}/episode.mp3", output)

    assert output.is_file()


def test_download_accepts_octet_stream(
    tmp_path: Path, http_server: Callable[[Responder], str]
) -> None:
    base_url = http_server(_audio_responder(content_type="application/octet-stream"))
    output = tmp_path / "episode.mp3"

    download_podcast(f"{base_url}/episode.mp3", output)

    assert output.read_bytes() == AUDIO_BODY


def test_download_rejects_html_response(
    tmp_path: Path, http_server: Callable[[Responder], str]
) -> None:
    base_url = http_server(_audio_responder(body=b"<html>nope</html>", content_type="text/html"))
    output = tmp_path / "episode.mp3"

    with pytest.raises(UnexpectedContentTypeError):
        download_podcast(f"{base_url}/episode.mp3", output)

    # The target path must not be created on validation failure.
    assert not output.exists()
    assert not output.with_suffix(".mp3.part").exists()


def test_download_rejects_non_http_url(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="http"):
        download_podcast("file:///etc/passwd", tmp_path / "episode.mp3")


def test_download_raises_on_http_error(
    tmp_path: Path, http_server: Callable[[Responder], str]
) -> None:
    def responder(_path: str) -> tuple[int, dict[str, str], bytes]:
        return 404, {"Content-Type": "text/plain"}, b"missing"

    base_url = http_server(responder)
    output = tmp_path / "episode.mp3"

    with pytest.raises(DownloadError, match="HTTP 404"):
        download_podcast(f"{base_url}/episode.mp3", output)

    assert not output.exists()
    assert not output.with_suffix(".mp3.part").exists()


def test_download_rejects_zero_byte_body(
    tmp_path: Path, http_server: Callable[[Responder], str]
) -> None:
    base_url = http_server(_audio_responder(body=b""))
    output = tmp_path / "episode.mp3"

    with pytest.raises(DownloadError, match="zero bytes"):
        download_podcast(f"{base_url}/episode.mp3", output)

    assert not output.exists()


def _redirect_responder(location: str) -> Responder:
    def respond(_path: str) -> tuple[int, dict[str, str], bytes]:
        return 302, {"Location": location, "Content-Type": "text/plain"}, b""

    return respond


def test_download_blocks_redirect_to_link_local_address(
    tmp_path: Path, http_server: Callable[[Responder], str]
) -> None:
    """A 302 to the cloud metadata range must be refused before connecting."""
    base_url = http_server(_redirect_responder("http://169.254.169.254/latest/meta-data/"))

    with pytest.raises(DownloadError, match="non-public address"):
        download_podcast(f"{base_url}/episode.mp3", tmp_path / "episode.mp3")


def test_download_blocks_redirect_to_private_address(
    tmp_path: Path, http_server: Callable[[Responder], str]
) -> None:
    base_url = http_server(_redirect_responder("http://10.0.0.1/internal.mp3"))

    with pytest.raises(DownloadError, match="non-public address"):
        download_podcast(f"{base_url}/episode.mp3", tmp_path / "episode.mp3")


def test_download_blocks_redirect_to_file_scheme(
    tmp_path: Path, http_server: Callable[[Responder], str]
) -> None:
    # stdlib itself refuses file:// redirect targets (HTTPError → DownloadError);
    # this pins that the protection holds whichever layer provides it.
    base_url = http_server(_redirect_responder("file:///etc/passwd"))

    with pytest.raises(DownloadError):
        download_podcast(f"{base_url}/episode.mp3", tmp_path / "episode.mp3")


def test_download_blocks_redirect_to_ftp_scheme(
    tmp_path: Path, http_server: Callable[[Responder], str]
) -> None:
    # stdlib *allows* ftp redirect targets — our handler must not.
    base_url = http_server(_redirect_responder("ftp://203.0.113.7/episode.mp3"))

    with pytest.raises(DownloadError, match="non-http"):
        download_podcast(f"{base_url}/episode.mp3", tmp_path / "episode.mp3")


class _UnboundedStream:
    """A fake response whose body never ends; counts how much was consumed."""

    def __init__(self) -> None:
        self.bytes_served = 0

    def read(self, amt: int, /) -> bytes:
        self.bytes_served += amt
        return b"x" * amt


def test_read_capped_rejects_unbounded_stream_during_read() -> None:
    stream = _UnboundedStream()
    cap = 256 * 1024

    with pytest.raises(DownloadError, match="too large"):
        read_capped(stream, max_bytes=cap, url="https://example.com/feed.xml", what="feed")

    # The cap must be enforced *while* reading: at most one 64 KiB chunk past
    # the cap may be consumed — never the (unbounded) remainder. A
    # read-then-check implementation would spin here forever.
    assert stream.bytes_served <= cap + 64 * 1024


def test_read_capped_returns_body_under_cap_intact() -> None:
    body = b"cue line\n" * 1000
    assert read_capped(BytesIO(body), max_bytes=1 << 20, url="u", what="feed") == body
