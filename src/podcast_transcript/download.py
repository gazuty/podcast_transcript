"""Stream a podcast audio file to disk, plus shared HTTP-fetch plumbing.

This module is the home for the safety constraints every network fetch in
the package shares (:mod:`feed`, :mod:`transcript_fetch`, and
:mod:`page_scrape` all import from here):

- :func:`open_http` opens a request with **redirect-hop validation**: the
  initial URL is supplied by the user (who is the trust boundary for a
  local CLI), but redirect targets are chosen by the remote server, so
  each hop must stay on http(s) and must not resolve to a loopback,
  private, link-local, or otherwise non-public address. This blocks the
  classic SSRF shape where a public-looking feed 302s to
  ``http://169.254.169.254/`` or an internal host.
- :func:`read_capped` buffers a response into memory while enforcing a
  byte cap *during* the read, so an unbounded body can't exhaust memory.
- :func:`download_podcast` streams to ``<output>.part`` and renames
  atomically on success, validating the ``Content-Type`` is audio-shaped
  so an HTML error page is never saved as ``.mp3``.

All of it is :mod:`urllib` from the standard library — the package keeps
its zero runtime-dependency footprint.
"""

from __future__ import annotations

import shutil
import socket
from io import BytesIO
from ipaddress import ip_address
from pathlib import Path
from typing import IO, TYPE_CHECKING, Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

if TYPE_CHECKING:
    from http.client import HTTPMessage
    from urllib.response import addinfourl

__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_USER_AGENT",
    "DownloadError",
    "UnexpectedContentTypeError",
    "download_podcast",
    "open_http",
    "read_capped",
]

DEFAULT_USER_AGENT = "podcast-transcript/0.1 (+https://github.com/gazuty/podcast_transcript)"
DEFAULT_TIMEOUT_SECONDS = 60.0
_CHUNK_SIZE = 1 << 20  # 1 MiB

# Many podcast CDNs serve MP3 files as ``application/octet-stream`` rather
# than ``audio/mpeg``; treat that as acceptable.
_AUDIO_CONTENT_TYPE_PREFIXES: tuple[str, ...] = (
    "audio/",
    "application/octet-stream",
)


class DownloadError(Exception):
    """Raised when a podcast download fails for any reason."""


class UnexpectedContentTypeError(DownloadError):
    """Raised when the server returns a non-audio ``Content-Type``."""


def _assert_public_redirect_target(url: str) -> None:
    """Reject a redirect target that escapes the public http(s) web.

    Every address the hostname resolves to must be globally routable —
    loopback, RFC-1918, link-local (cloud metadata), and reserved ranges
    are all refused. Resolution happens here and again inside ``urlopen``,
    so a DNS-rebinding attacker with a sub-second TTL could in principle
    pass this check and connect elsewhere; for a single-user local CLI
    that residual risk is accepted (see SECURITY.md).
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise DownloadError(f"blocked redirect to non-http(s) URL: {url!r}")
    host = parsed.hostname
    if not host:
        raise DownloadError(f"blocked redirect with no hostname: {url!r}")
    try:
        infos = socket.getaddrinfo(host, parsed.port or 0, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise DownloadError(f"could not resolve redirect host {host!r}: {exc}") from exc
    for info in infos:
        # Strip any IPv6 zone id ("fe80::1%en0") — ip_address rejects it,
        # and a zoned address is link-local by definition anyway.
        bare = str(info[4][0]).partition("%")[0]
        try:
            addr = ip_address(bare)
        except ValueError as exc:
            raise DownloadError(
                f"blocked redirect to unparseable address {info[4][0]!r} for {url!r}",
            ) from exc
        if not addr.is_global:
            raise DownloadError(
                f"blocked redirect to non-public address: {url!r} resolves to {addr}",
            )


class _PublicRedirectHandler(HTTPRedirectHandler):
    """Default redirect handling, plus :func:`_assert_public_redirect_target` per hop."""

    def redirect_request(
        self,
        req: Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> Request | None:
        _assert_public_redirect_target(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = build_opener(_PublicRedirectHandler)


def open_http(request: Request, *, timeout: float) -> addinfourl:
    """Open *request* with redirect-hop validation; see the module docstring."""
    # ``OpenerDirector.open`` is typed ``Any`` in typeshed; the runtime type
    # for http(s) is the same response object ``urlopen`` returns.
    return cast("addinfourl", _OPENER.open(request, timeout=timeout))


class _Readable(Protocol):
    """The slice of a ``urlopen`` response that :func:`read_capped` needs."""

    def read(self, amt: int, /) -> bytes: ...


def read_capped(
    response: _Readable,
    *,
    max_bytes: int,
    url: str,
    what: str,
) -> bytes:
    """Read *response* into memory, enforcing *max_bytes* **during** the read.

    Reads in 64 KiB chunks and raises as soon as the running total exceeds
    the cap, so an oversized (or unbounded chunked) response is rejected
    after buffering at most one chunk past *max_bytes* — never the whole
    body. This is the in-memory counterpart of the streaming download above;
    every fetcher that slurps a response should go through it.
    """
    buf = BytesIO()
    while True:
        chunk = response.read(64 * 1024)
        if not chunk:
            return buf.getvalue()
        buf.write(chunk)
        if buf.tell() > max_bytes:
            raise DownloadError(
                f"{what} body too large fetching {url!r}: exceeds {max_bytes} bytes"
            )


def download_podcast(
    url: str,
    output_path: Path | str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    chunk_size: int = _CHUNK_SIZE,
    user_agent: str = DEFAULT_USER_AGENT,
    allowed_content_type_prefixes: tuple[str, ...] = _AUDIO_CONTENT_TYPE_PREFIXES,
) -> Path:
    """Download the audio file at *url* to *output_path*.

    Args:
        url: Direct URL to a podcast audio file. Must be ``http`` or ``https``.
        output_path: Destination path on disk. Parent directories are created
            if missing.
        timeout: Per-operation timeout in seconds, passed to ``urlopen``.
        chunk_size: Read buffer size in bytes.
        user_agent: ``User-Agent`` header to send with the request.
        allowed_content_type_prefixes: Acceptable ``Content-Type`` prefixes.
            Defaults to ``audio/*`` and ``application/octet-stream``.

    Returns:
        The resolved :class:`pathlib.Path` of the downloaded file.

    Raises:
        DownloadError: For any HTTP, network, or filesystem failure.
        UnexpectedContentTypeError: If the response ``Content-Type`` does not
            match one of *allowed_content_type_prefixes*.
        ValueError: If *url* is not an ``http``/``https`` URL.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(
            f"Only http(s) URLs are supported, got scheme {parsed.scheme!r}",
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = output_path.with_suffix(output_path.suffix + ".part")

    request = Request(url, headers={"User-Agent": user_agent})

    try:
        with open_http(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if not any(content_type.startswith(prefix) for prefix in allowed_content_type_prefixes):
                raise UnexpectedContentTypeError(
                    f"Refusing to save {url!r} with Content-Type {content_type!r} as audio.",
                )
            with part_path.open("wb") as part_file:
                shutil.copyfileobj(response, part_file, length=chunk_size)
    except HTTPError as exc:
        part_path.unlink(missing_ok=True)
        raise DownloadError(
            f"HTTP {exc.code} fetching {url!r}: {exc.reason}",
        ) from exc
    except URLError as exc:
        part_path.unlink(missing_ok=True)
        raise DownloadError(f"Network error fetching {url!r}: {exc.reason}") from exc
    except Exception:
        part_path.unlink(missing_ok=True)
        raise

    if part_path.stat().st_size == 0:
        part_path.unlink(missing_ok=True)
        raise DownloadError(f"Downloaded zero bytes from {url!r}")

    part_path.replace(output_path)
    return output_path
