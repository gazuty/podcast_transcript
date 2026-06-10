"""Minimal HTML scraping for the ``--page`` source flow.

When the user nominates an episode by giving us a web page URL rather than
a direct MP3 link or an RSS feed, we need to discover two things from the
page's HTML:

1. A transcript URL (``.srt`` / ``.vtt``) if the publisher links to one.
2. The audio URL (``.mp3`` / ``.m4a``) so we can fall back to Whisper.

Implementation uses :mod:`html.parser` from the stdlib so we don't take on
``beautifulsoup4`` as a dep. The heuristics are deliberately conservative:

- Transcript link: an ``<a href>``, ``<link href>``, or ``<source src>`` whose
  href ends in ``.srt``/``.vtt`` (case-insensitive), or whose ``type``
  attribute matches a known SRT/VTT mime type. The latter handles
  ``<link rel="alternate" type="application/srt" href="...">``-style
  declarations.
- Audio link: ``<audio src>`` / ``<source src>`` / ``<a href>`` ending in
  a known audio extension.

Relative URLs are resolved against the page URL. If multiple candidates
exist, the first one wins — deterministic, easy to reason about, and the
fallback is "user passes ``--url`` directly" so there's no need to be clever.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request

from .download import (
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_USER_AGENT,
    DownloadError,
    open_http,
    read_capped,
)
from .feed import TranscriptRef
from .transcript_fetch import SRT_MIME_TYPES, VTT_MIME_TYPES

__all__ = [
    "MAX_PAGE_BYTES",
    "PageInfo",
    "discover_episode_links",
    "fetch_page_html",
    "parse_episode_links",
]


# Cap HTML downloads so a misconfigured URL pointing at a giant file can't
# OOM the process. 5 MiB is enormous for an episode page.
MAX_PAGE_BYTES: int = 5 * 1024 * 1024

_AUDIO_EXTENSIONS: tuple[str, ...] = (".mp3", ".m4a", ".aac", ".wav", ".ogg", ".opus")
_TRANSCRIPT_EXTENSIONS: tuple[str, ...] = (".srt", ".vtt")
_KNOWN_TRANSCRIPT_MIME_TYPES: frozenset[str] = SRT_MIME_TYPES | VTT_MIME_TYPES


@dataclass
class PageInfo:
    """Result of scraping an episode page."""

    transcripts: tuple[TranscriptRef, ...] = field(default_factory=tuple)
    audio_url: str | None = None


def fetch_page_html(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    user_agent: str = DEFAULT_USER_AGENT,
    max_bytes: int = MAX_PAGE_BYTES,
) -> str:
    """Download an HTML page into memory.

    Restricts URL schemes to http/https. The *max_bytes* cap is enforced
    while reading, so an oversized response is rejected without being
    buffered in full. Returns the decoded body (best-effort UTF-8);
    content-type is *not* enforced because publishers serve podcast pages
    as a mix of ``text/html``, ``application/xhtml+xml``, and occasionally
    ``application/xml``.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Only http(s) URLs are supported, got scheme {parsed.scheme!r}")
    request = Request(url, headers={"User-Agent": user_agent})
    try:
        with open_http(request, timeout=timeout) as response:
            data = read_capped(response, max_bytes=max_bytes, url=url, what="page")
    except HTTPError as exc:
        raise DownloadError(f"HTTP {exc.code} fetching {url!r}: {exc.reason}") from exc
    except URLError as exc:
        raise DownloadError(f"Network error fetching {url!r}: {exc.reason}") from exc
    return data.decode("utf-8", errors="replace")


class _EpisodePageParser(HTMLParser):
    """Collect transcript and audio link candidates from page HTML.

    We don't care about textual content, just attributes on a handful of
    tags. Anything we don't recognise is silently ignored.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.transcripts: list[TranscriptRef] = []
        self.audio_url: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {k: (v or "") for k, v in attrs}
        if tag in {"a", "link"}:
            self._consider_href_link(attr_map)
        elif tag in {"audio", "source"}:
            self._consider_media_source(attr_map)

    def _consider_href_link(self, attrs: dict[str, str]) -> None:
        href = attrs.get("href", "").strip()
        if not href:
            return
        type_attr = attrs.get("type", "").strip().lower()
        if type_attr in _KNOWN_TRANSCRIPT_MIME_TYPES:
            self.transcripts.append(TranscriptRef(url=href, mime_type=type_attr))
            return
        lowered = href.lower()
        for ext, mime in (
            (".srt", "application/srt"),
            (".vtt", "text/vtt"),
        ):
            # Match on the path portion so query strings don't defeat the check.
            if _path_endswith(lowered, ext):
                self.transcripts.append(TranscriptRef(url=href, mime_type=mime))
                return
        if self.audio_url is None and any(
            _path_endswith(lowered, ext) for ext in _AUDIO_EXTENSIONS
        ):
            self.audio_url = href

    def _consider_media_source(self, attrs: dict[str, str]) -> None:
        src = attrs.get("src", "").strip()
        if not src:
            return
        lowered = src.lower()
        if self.audio_url is None and any(
            _path_endswith(lowered, ext) for ext in _AUDIO_EXTENSIONS
        ):
            self.audio_url = src


def _path_endswith(url_lower: str, ext: str) -> bool:
    """Match *ext* against the URL path only, ignoring query/fragment."""
    path = urlparse(url_lower).path
    return path.endswith(ext)


def _is_http_url(url: str) -> bool:
    return urlparse(url).scheme in {"http", "https"}


def parse_episode_links(html: str, *, page_url: str) -> PageInfo:
    """Parse *html* and return absolute URLs for transcript / audio links.

    Relative URLs are resolved against *page_url* via
    :func:`urllib.parse.urljoin`. Candidates that resolve to anything other
    than http(s) — a page can legitimately contain ``file://`` or ``data:``
    hrefs — are dropped here so no downstream fetcher ever sees them.
    """
    parser = _EpisodePageParser()
    parser.feed(html)
    parser.close()
    transcripts: list[TranscriptRef] = []
    for ref in parser.transcripts:
        resolved = urljoin(page_url, ref.url)
        if _is_http_url(resolved):
            transcripts.append(TranscriptRef(url=resolved, mime_type=ref.mime_type))
    audio_url = urljoin(page_url, parser.audio_url) if parser.audio_url else None
    if audio_url is not None and not _is_http_url(audio_url):
        audio_url = None
    return PageInfo(transcripts=tuple(transcripts), audio_url=audio_url)


def discover_episode_links(
    page_url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> PageInfo:
    """Convenience: fetch the page and parse it in one call."""
    html = fetch_page_html(page_url, timeout=timeout)
    return parse_episode_links(html, page_url=page_url)
