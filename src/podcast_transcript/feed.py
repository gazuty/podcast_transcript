"""Minimal stdlib-only RSS-2.0 feed parser for podcast feeds.

Why stdlib over ``feedparser``: the rest of the package is zero-dep at
runtime, and we only need three fields (title, enclosure URL, optional
pubDate) to power the ``run`` subcommand. ``xml.etree.ElementTree`` parses
the namespaces correctly and is fast enough for any feed under ~10 MB.

This is a deliberately narrow parser:

- Only RSS 2.0 ``<rss><channel><item>`` is recognised — Atom feeds raise.
- Items missing an ``<enclosure url=…>`` are skipped (they aren't audio
  episodes).
- ``itunes:title`` and similar extensions are ignored; we read plain
  ``<title>`` and ``<pubDate>``.
- The Podcasting 2.0 ``<podcast:transcript>`` element *is* read, since the
  whole point of this package's transcript-discovery flow is to honour it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request
from xml.etree import ElementTree as ET  # community-standard alias

from .download import (
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_USER_AGENT,
    DownloadError,
    open_http,
    read_capped,
)

__all__ = [
    "PODCAST_NAMESPACE",
    "FeedItem",
    "FeedParseError",
    "TranscriptRef",
    "fetch_feed",
    "load_feed",
    "parse_feed",
    "select_item",
]


# Podcasting 2.0 namespace, registered at podcastindex.org.
PODCAST_NAMESPACE = "https://podcastindex.org/namespace/1.0"
_PODCAST_TRANSCRIPT_TAG = f"{{{PODCAST_NAMESPACE}}}transcript"


class FeedParseError(Exception):
    """Raised when the bytes do not look like an RSS-2.0 feed we can read."""


@dataclass(frozen=True)
class TranscriptRef:
    """A publisher-declared ``<podcast:transcript>`` reference.

    *mime_type* is the raw ``type=`` attribute, lowercased; *language* mirrors
    the optional ``language=`` attribute (BCP-47, e.g. ``en``).
    """

    url: str
    mime_type: str
    language: str | None = None


@dataclass(frozen=True)
class FeedItem:
    """A single ``<item>`` from an RSS feed (only the fields we need).

    *pub_date* is the raw, stripped text of the ``<pubDate>`` element —
    typically RFC 822 (``Wed, 03 Jan 2026 00:00:00 GMT``) but not parsed
    or validated here. Callers that need a real date must parse it.
    """

    title: str
    enclosure_url: str
    pub_date: str | None = None
    transcripts: tuple[TranscriptRef, ...] = ()


def fetch_feed(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    user_agent: str = DEFAULT_USER_AGENT,
    max_bytes: int = 10 * 1024 * 1024,
) -> bytes:
    """Fetch the raw bytes of an RSS feed via ``urllib``.

    The *max_bytes* cap (default 10 MiB) is enforced while reading, so an
    oversized or unbounded response is rejected without ever being buffered
    in full. Restricts URL schemes to http/https.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Only http(s) URLs are supported, got scheme {parsed.scheme!r}")
    request = Request(url, headers={"User-Agent": user_agent})
    try:
        with open_http(request, timeout=timeout) as response:
            return read_capped(response, max_bytes=max_bytes, url=url, what="feed")
    except HTTPError as exc:
        raise DownloadError(f"HTTP {exc.code} fetching {url!r}: {exc.reason}") from exc
    except URLError as exc:
        raise DownloadError(f"Network error fetching {url!r}: {exc.reason}") from exc


# DTDs enable entity-expansion attacks (billion laughs) that stdlib
# ElementTree does not bound, and the byte cap above measures *raw* bytes,
# not expanded output. No real podcast feed needs a DOCTYPE, so refuse them
# outright — the same posture defusedxml takes, without the dependency.
_DTD_MARKER = re.compile(rb"<!\s*(?:DOCTYPE|ENTITY)", re.IGNORECASE)


def parse_feed(xml_bytes: bytes) -> list[FeedItem]:
    """Parse RSS-2.0 bytes into a list of :class:`FeedItem` in feed order.

    Items without an ``<enclosure url=…>`` attribute are skipped silently —
    those are typically blog-only items mixed into a podcast feed. Feeds
    containing a ``<!DOCTYPE``/``<!ENTITY`` declaration are rejected before
    parsing (entity expansion is unbounded in stdlib ElementTree).
    """
    if _DTD_MARKER.search(xml_bytes):
        raise FeedParseError(
            "feed contains a DOCTYPE/ENTITY declaration; refusing to parse "
            "(entity expansion is a denial-of-service vector)",
        )
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise FeedParseError(f"could not parse XML: {exc}") from exc

    if root.tag.lower() != "rss":
        raise FeedParseError(
            f"expected an <rss> root element, got <{root.tag}>; Atom feeds are not supported",
        )

    channel = root.find("channel")
    if channel is None:
        raise FeedParseError("RSS feed has no <channel> element")

    items: list[FeedItem] = []
    for item_el in channel.findall("item"):
        enclosure = item_el.find("enclosure")
        if enclosure is None:
            continue
        url = enclosure.get("url", "").strip()
        if not url:
            continue
        title_el = item_el.find("title")
        title = (title_el.text or "").strip() if title_el is not None else ""
        pub_el = item_el.find("pubDate")
        pub_date = (pub_el.text or "").strip() if pub_el is not None and pub_el.text else None
        transcripts = _parse_transcripts(item_el)
        items.append(
            FeedItem(
                title=title,
                enclosure_url=url,
                pub_date=pub_date,
                transcripts=transcripts,
            ),
        )

    return items


def _parse_transcripts(item_el: ET.Element) -> tuple[TranscriptRef, ...]:
    """Pull all ``<podcast:transcript>`` children off an ``<item>``.

    Returns them in document order. Entries with no ``url`` or ``type``
    attribute are skipped silently; callers further downstream filter on
    mime type to pick formats they can actually consume.
    """
    refs: list[TranscriptRef] = []
    for el in item_el.findall(_PODCAST_TRANSCRIPT_TAG):
        url = (el.get("url") or "").strip()
        mime = (el.get("type") or "").strip().lower()
        if not url or not mime:
            continue
        language = el.get("language")
        language = language.strip() if language else None
        refs.append(TranscriptRef(url=url, mime_type=mime, language=language or None))
    return tuple(refs)


def load_feed(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> list[FeedItem]:
    """Convenience wrapper: fetch and parse in one call."""
    return parse_feed(fetch_feed(url, timeout=timeout))


def select_item(
    items: list[FeedItem],
    *,
    regex: str | None = None,
    index: int | None = None,
) -> FeedItem:
    """Pick a single item by title regex (first match) or 0-based index.

    Exactly one of *regex* or *index* must be provided. *regex* is matched
    against ``FeedItem.title`` with :func:`re.search` (case-insensitive). If
    both are passed, *regex* wins; if neither is passed, ``ValueError`` is
    raised.
    """
    if not items:
        raise ValueError("feed has no items with audio enclosures")
    if regex is not None:
        try:
            compiled = re.compile(regex, re.IGNORECASE)
        except re.error as exc:
            # ``re.error`` doesn't subclass ValueError, so without this a
            # bad --episode-regex escapes the CLI's error mapping as a
            # raw traceback.
            raise ValueError(f"invalid episode regex {regex!r}: {exc}") from exc
        for item in items:
            if compiled.search(item.title):
                return item
        raise ValueError(f"no item title matches regex {regex!r}")
    if index is not None:
        if index < 0 or index >= len(items):
            raise ValueError(
                f"episode index {index} out of range (feed has {len(items)} items)",
            )
        return items[index]
    raise ValueError("must pass exactly one of regex= or index=")
