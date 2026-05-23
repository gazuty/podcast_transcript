"""Fetch publisher-hosted transcripts and convert them to plain text.

Two responsibilities, kept narrow on purpose:

1. **Fetch.** Download a ``<podcast:transcript>`` (or otherwise-discovered)
   URL into memory, with the same safety constraints as :mod:`download`
   (http(s) only, size cap, ``Content-Type`` sanity check).
2. **Convert.** Strip SRT/VTT cue indexes, timestamps, and header lines
   down to one prose line per cue — the same shape Whisper's ``.txt``
   writer emits, so :mod:`clean` can run unchanged on the result.

We intentionally do *not* accept HTML or JSON transcripts here. HTML
needs tag-stripping that's brittle in the general case; JSON has a
schema (Podcasting 2.0) but is rarely served and would only marginally
beat SRT/VTT on quality. Keeping the surface to two well-defined,
caption-shaped formats means the rest of the pipeline doesn't have to
branch on transcript shape.
"""

from __future__ import annotations

import re
import shutil
from io import BytesIO
from typing import TYPE_CHECKING
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .download import DEFAULT_TIMEOUT_SECONDS, DEFAULT_USER_AGENT, DownloadError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .feed import TranscriptRef

__all__ = [
    "MAX_TRANSCRIPT_BYTES",
    "SRT_MIME_TYPES",
    "VTT_MIME_TYPES",
    "TranscriptFetchError",
    "TranscriptKind",
    "classify_mime_type",
    "fetch_transcript_text",
    "preferred_transcript",
    "subrip_to_text",
    "vtt_to_text",
]


# Mime types we'll accept for each format. Some publishers serve SRT as
# ``application/x-subrip``; treat that as a synonym for ``application/srt``.
SRT_MIME_TYPES: frozenset[str] = frozenset(
    {
        "application/srt",
        "application/x-subrip",
        "text/srt",
    },
)
VTT_MIME_TYPES: frozenset[str] = frozenset(
    {
        "text/vtt",
        "application/vtt",
    },
)

# 5 MiB is comfortably more than any caption file we've seen in the wild
# and protects us from being told to slurp a multi-GB file.
MAX_TRANSCRIPT_BYTES: int = 5 * 1024 * 1024

# Plain-text content-type prefixes we'll accept on the wire. Servers vary
# wildly on what they advertise for SRT/VTT — text/* covers most of it.
_ACCEPTED_CONTENT_TYPE_PREFIXES: tuple[str, ...] = (
    "text/",
    "application/srt",
    "application/x-subrip",
    "application/vtt",
    "application/octet-stream",
)


TranscriptKind = str  # "srt" | "vtt"


class TranscriptFetchError(Exception):
    """Raised when a transcript URL can't be fetched or recognised."""


def classify_mime_type(mime_type: str) -> TranscriptKind | None:
    """Return ``"srt"`` / ``"vtt"`` if *mime_type* names a supported format.

    Comparison is case-insensitive and ignores any ``;charset=...`` suffix.
    Returns ``None`` for unsupported types (HTML, JSON, etc.).
    """
    normalized = mime_type.split(";")[0].strip().lower()
    if normalized in SRT_MIME_TYPES:
        return "srt"
    if normalized in VTT_MIME_TYPES:
        return "vtt"
    return None


def preferred_transcript(
    refs: Iterable[TranscriptRef],
) -> tuple[TranscriptRef, TranscriptKind] | None:
    """Pick the best transcript reference, preferring SRT over VTT.

    Returns ``(ref, kind)`` for the first SRT match, falling back to the
    first VTT match, or ``None`` if neither format is present.

    SRT wins ties because the cue grammar is simpler — no ``WEBVTT``
    headers, no cue settings, no ``NOTE`` blocks — which makes the
    text-extraction step slightly more robust.
    """
    srt: tuple[TranscriptRef, TranscriptKind] | None = None
    vtt: tuple[TranscriptRef, TranscriptKind] | None = None
    for ref in refs:
        kind = classify_mime_type(ref.mime_type)
        if kind == "srt" and srt is None:
            srt = (ref, "srt")
        elif kind == "vtt" and vtt is None:
            vtt = (ref, "vtt")
    return srt or vtt


def fetch_transcript_text(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    user_agent: str = DEFAULT_USER_AGENT,
    max_bytes: int = MAX_TRANSCRIPT_BYTES,
) -> tuple[str, str]:
    """Download a transcript URL into memory.

    Returns ``(body, content_type)`` where *content_type* is the raw header
    value (lowercased, no parameters), so the caller can route to the
    right converter even if the feed lied about the format.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Only http(s) URLs are supported, got scheme {parsed.scheme!r}")
    request = Request(url, headers={"User-Agent": user_agent})
    try:
        with urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if content_type and not any(
                content_type.startswith(prefix) for prefix in _ACCEPTED_CONTENT_TYPE_PREFIXES
            ):
                raise TranscriptFetchError(
                    f"Refusing to read {url!r} as a transcript: Content-Type {content_type!r}",
                )
            buf = BytesIO()
            shutil.copyfileobj(response, buf, length=64 * 1024)
            data = buf.getvalue()
    except HTTPError as exc:
        raise DownloadError(f"HTTP {exc.code} fetching {url!r}: {exc.reason}") from exc
    except URLError as exc:
        raise DownloadError(f"Network error fetching {url!r}: {exc.reason}") from exc
    if len(data) > max_bytes:
        raise DownloadError(f"transcript body too large: {len(data)} > {max_bytes}")
    return data.decode("utf-8", errors="replace"), content_type


# ---------------------------------------------------------------------------
# SRT → text
# ---------------------------------------------------------------------------


# An SRT timestamp line looks like:  "00:01:23,456 --> 00:01:25,789"
# (some files use a "." instead of "," for the millisecond separator, and
# VTT uses the same shape but with "."). The arrow may also have optional
# whitespace.
_TIMESTAMP_LINE = re.compile(
    r"^\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}.*$",
)
_INDEX_LINE = re.compile(r"^\d+$")


def _strip_cue_lines(text: str, *, drop_predicates: tuple[re.Pattern[str], ...]) -> list[str]:
    out: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            out.append(" ".join(s.strip() for s in current if s.strip()))
            current.clear()

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            flush()
            continue
        if any(p.match(line) for p in drop_predicates):
            continue
        current.append(line)
    flush()
    return [line for line in out if line]


def subrip_to_text(srt: str) -> str:
    """Convert SubRip (``.srt``) caption text to one prose line per cue.

    Strips cue index lines, ``HH:MM:SS,mmm --> ...`` timestamps, and inline
    formatting tags Whisper would never emit (``<i>``, ``<b>``, ``{...}``
    style overrides). Multi-line cues are joined with a single space.
    """
    lines = _strip_cue_lines(srt, drop_predicates=(_INDEX_LINE, _TIMESTAMP_LINE))
    cleaned = [_strip_inline_markup(line) for line in lines]
    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# VTT → text
# ---------------------------------------------------------------------------


_VTT_HEADER = re.compile(r"^(WEBVTT\b.*|NOTE\b.*|STYLE\b.*|REGION\b.*)$")
_VTT_CUE_ID = re.compile(r"^[^\s-]+$")  # very loose; only used when followed by a timestamp line


def vtt_to_text(vtt: str) -> str:
    """Convert WebVTT (``.vtt``) caption text to one prose line per cue.

    Drops the leading ``WEBVTT`` header, any ``NOTE``/``STYLE``/``REGION``
    blocks, timestamp lines, and inline markup. Cue identifiers (the
    optional non-blank line that may precede a timestamp) are also
    discarded — we keep only the spoken text.
    """
    # Two-pass: first split blocks separated by blank lines, then within each
    # block drop anything that's a header/timestamp/identifier so the rest is
    # the actual caption text.
    blocks: list[list[str]] = []
    current: list[str] = []
    for raw_line in vtt.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(line)
    if current:
        blocks.append(current)

    out: list[str] = []
    for block in blocks:
        # Drop full header blocks (WEBVTT / NOTE / STYLE / REGION).
        if _VTT_HEADER.match(block[0]):
            continue
        text_lines: list[str] = []
        seen_timestamp = False
        for line in block:
            if _TIMESTAMP_LINE.match(line):
                seen_timestamp = True
                continue
            # A bare line before any timestamp is a cue identifier; skip it.
            if not seen_timestamp and _VTT_CUE_ID.match(line) and len(block) > 1:
                continue
            text_lines.append(line)
        if text_lines:
            joined = " ".join(s.strip() for s in text_lines if s.strip())
            cleaned = _strip_inline_markup(joined)
            if cleaned:
                out.append(cleaned)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Shared inline markup stripper
# ---------------------------------------------------------------------------


# SRT/VTT both allow simple inline tags; some files also carry SSA-style
# {\\an8} overrides. None of these belong in the prose feed.
_INLINE_TAG = re.compile(r"<[^>]+>")
_INLINE_BRACE = re.compile(r"\{[^}]*\}")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def _strip_inline_markup(line: str) -> str:
    line = _INLINE_TAG.sub("", line)
    line = _INLINE_BRACE.sub("", line)
    line = _MULTI_SPACE.sub(" ", line)
    return line.strip()
