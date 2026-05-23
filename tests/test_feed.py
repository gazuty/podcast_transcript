"""Tests for :mod:`podcast_transcript.feed`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from podcast_transcript.feed import (
    FeedItem,
    FeedParseError,
    TranscriptRef,
    fetch_feed,
    load_feed,
    parse_feed,
    select_item,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from .conftest import Responder


SAMPLE_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Show</title>
    <item>
      <title>Episode 3: Newest</title>
      <pubDate>Wed, 03 Jan 2026 00:00:00 GMT</pubDate>
      <enclosure url="https://example.com/ep3.mp3" type="audio/mpeg"/>
    </item>
    <item>
      <title>Episode 2: Middle</title>
      <enclosure url="https://example.com/ep2.mp3" type="audio/mpeg"/>
    </item>
    <item>
      <title>Item with no audio</title>
    </item>
    <item>
      <title>Episode 1: Oldest</title>
      <enclosure url="https://example.com/ep1.mp3" type="audio/mpeg"/>
    </item>
  </channel>
</rss>
"""


def test_parse_feed_extracts_items_in_order() -> None:
    items = parse_feed(SAMPLE_FEED)
    assert [item.enclosure_url for item in items] == [
        "https://example.com/ep3.mp3",
        "https://example.com/ep2.mp3",
        "https://example.com/ep1.mp3",
    ]
    assert items[0].pub_date is not None
    assert items[0].title == "Episode 3: Newest"


def test_parse_feed_rejects_atom() -> None:
    atom = b'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"/>'
    with pytest.raises(FeedParseError, match="Atom feeds"):
        parse_feed(atom)


def test_parse_feed_rejects_garbage() -> None:
    with pytest.raises(FeedParseError, match="parse XML"):
        parse_feed(b"not xml at all")


def test_select_item_by_regex_first_match() -> None:
    items = parse_feed(SAMPLE_FEED)
    chosen = select_item(items, regex=r"middle")
    assert chosen.enclosure_url == "https://example.com/ep2.mp3"


def test_select_item_by_index() -> None:
    items = parse_feed(SAMPLE_FEED)
    assert select_item(items, index=0).title == "Episode 3: Newest"
    assert select_item(items, index=2).title == "Episode 1: Oldest"


def test_select_item_no_match_raises() -> None:
    items = parse_feed(SAMPLE_FEED)
    with pytest.raises(ValueError, match="no item title matches"):
        select_item(items, regex=r"never going to match")


def test_select_item_index_out_of_range() -> None:
    items = parse_feed(SAMPLE_FEED)
    with pytest.raises(ValueError, match="out of range"):
        select_item(items, index=99)


def test_select_item_requires_argument() -> None:
    items = parse_feed(SAMPLE_FEED)
    with pytest.raises(ValueError, match="exactly one of"):
        select_item(items)


def test_fetch_feed_via_http(http_server: Callable[[Responder], str]) -> None:
    def respond(_path: str) -> tuple[int, dict[str, str], bytes]:
        return (200, {"Content-Type": "application/rss+xml"}, SAMPLE_FEED)

    base_url = http_server(respond)
    data = fetch_feed(f"{base_url}/feed.xml")
    assert b"<rss" in data


def test_load_feed_end_to_end(http_server: Callable[[Responder], str]) -> None:
    def respond(_path: str) -> tuple[int, dict[str, str], bytes]:
        return (200, {"Content-Type": "application/rss+xml"}, SAMPLE_FEED)

    base_url = http_server(respond)
    items = load_feed(f"{base_url}/feed.xml")
    assert isinstance(items[0], FeedItem)
    assert items[0].enclosure_url == "https://example.com/ep3.mp3"


def test_fetch_feed_rejects_non_http_scheme() -> None:
    with pytest.raises(ValueError, match="http"):
        fetch_feed("file:///etc/passwd")


# ---------------------------------------------------------------------------
# Podcasting 2.0 <podcast:transcript> parsing
# ---------------------------------------------------------------------------


TRANSCRIPT_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:podcast="https://podcastindex.org/namespace/1.0">
  <channel>
    <title>Show</title>
    <item>
      <title>With transcript</title>
      <enclosure url="https://example.com/ep1.mp3" type="audio/mpeg"/>
      <podcast:transcript url="https://example.com/ep1.srt" type="application/srt" language="en"/>
      <podcast:transcript url="https://example.com/ep1.vtt" type="text/vtt"/>
      <podcast:transcript url="https://example.com/ep1.html" type="text/html"/>
    </item>
    <item>
      <title>No transcript declared</title>
      <enclosure url="https://example.com/ep2.mp3" type="audio/mpeg"/>
    </item>
    <item>
      <title>Malformed transcript</title>
      <enclosure url="https://example.com/ep3.mp3" type="audio/mpeg"/>
      <podcast:transcript url="" type="application/srt"/>
      <podcast:transcript url="https://example.com/ep3.srt"/>
    </item>
  </channel>
</rss>
"""


def test_parse_feed_extracts_podcast_transcripts() -> None:
    items = parse_feed(TRANSCRIPT_FEED)
    with_t, without_t, malformed = items

    assert with_t.transcripts == (
        TranscriptRef(
            url="https://example.com/ep1.srt",
            mime_type="application/srt",
            language="en",
        ),
        TranscriptRef(url="https://example.com/ep1.vtt", mime_type="text/vtt"),
        TranscriptRef(url="https://example.com/ep1.html", mime_type="text/html"),
    )
    assert without_t.transcripts == ()
    # Both malformed entries (missing url, missing type) are dropped.
    assert malformed.transcripts == ()


def test_parse_feed_default_transcripts_is_empty_tuple() -> None:
    # Existing minimal feed has no podcast namespace — every item should
    # still parse cleanly with an empty transcripts tuple.
    items = parse_feed(SAMPLE_FEED)
    assert all(item.transcripts == () for item in items)
