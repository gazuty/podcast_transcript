"""Stream a podcast audio file to disk with basic validation.

Implementation notes
--------------------
- Uses :mod:`urllib.request` from the standard library to keep the runtime
  dependency footprint at zero. The default opener follows redirects.
- Streams the response to ``<output>.part`` and renames atomically on success
  so a failed/aborted download never leaves a half-written file at the target
  path.
- Validates the response ``Content-Type`` is audio-shaped before writing, to
  avoid silently saving an HTML error page from a misbehaving CDN as ``.mp3``.
- Restricts URL schemes to ``http``/``https`` to keep ``urlopen`` from being
  coerced into reading local files via ``file://``.
"""

from __future__ import annotations

import shutil
from io import BytesIO
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_USER_AGENT",
    "DownloadError",
    "UnexpectedContentTypeError",
    "download_podcast",
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
        with urlopen(request, timeout=timeout) as response:
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
