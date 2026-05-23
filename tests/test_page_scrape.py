"""Tests for :mod:`podcast_transcript.page_scrape`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from podcast_transcript.download import DownloadError
from podcast_transcript.page_scrape import (
    discover_episode_links,
    fetch_page_html,
    parse_episode_links,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from .conftest import Responder


PAGE_URL = "https://example.com/episodes/42/"


def test_parse_finds_srt_and_audio() -> None:
    html = """
    <html><body>
      <a href="/audio/episode-42.mp3">listen</a>
      <a href="transcripts/ep42.srt">transcript (SRT)</a>
    </body></html>
    """
    info = parse_episode_links(html, page_url=PAGE_URL)
    assert info.audio_url == "https://example.com/audio/episode-42.mp3"
    assert len(info.transcripts) == 1
    assert info.transcripts[0].url == "https://example.com/episodes/42/transcripts/ep42.srt"
    assert info.transcripts[0].mime_type == "application/srt"


def test_parse_finds_vtt_via_link_tag() -> None:
    html = (
        '<link rel="alternate" type="text/vtt" href="/cc/ep42.vtt">'
        '<audio src="/media/ep42.mp3"></audio>'
    )
    info = parse_episode_links(html, page_url=PAGE_URL)
    assert info.audio_url == "https://example.com/media/ep42.mp3"
    assert len(info.transcripts) == 1
    assert info.transcripts[0].url == "https://example.com/cc/ep42.vtt"
    assert info.transcripts[0].mime_type == "text/vtt"


def test_parse_ignores_query_string_on_audio_extension() -> None:
    html = '<a href="https://cdn.example.com/ep42.mp3?token=abc">audio</a>'
    info = parse_episode_links(html, page_url=PAGE_URL)
    assert info.audio_url == "https://cdn.example.com/ep42.mp3?token=abc"


def test_parse_collects_multiple_transcripts_in_document_order() -> None:
    html = """
    <a href="/t/ep42.srt">SRT</a>
    <a href="/t/ep42.vtt">VTT</a>
    """
    info = parse_episode_links(html, page_url=PAGE_URL)
    assert [t.url for t in info.transcripts] == [
        "https://example.com/t/ep42.srt",
        "https://example.com/t/ep42.vtt",
    ]
    assert [t.mime_type for t in info.transcripts] == [
        "application/srt",
        "text/vtt",
    ]


def test_parse_returns_empty_when_nothing_found() -> None:
    html = "<html><body><p>No links here.</p></body></html>"
    info = parse_episode_links(html, page_url=PAGE_URL)
    assert info.audio_url is None
    assert info.transcripts == ()


def test_parse_audio_first_match_wins() -> None:
    html = """
    <a href="/m/first.mp3">first</a>
    <a href="/m/second.mp3">second</a>
    """
    info = parse_episode_links(html, page_url=PAGE_URL)
    assert info.audio_url == "https://example.com/m/first.mp3"


def test_fetch_page_html_via_http(http_server: Callable[[Responder], str]) -> None:
    body = b"<html><body><a href='/ep.srt'>x</a></body></html>"

    def respond(_path: str) -> tuple[int, dict[str, str], bytes]:
        return (200, {"Content-Type": "text/html"}, body)

    base_url = http_server(respond)
    html = fetch_page_html(f"{base_url}/episodes/42/")
    assert "ep.srt" in html


def test_fetch_page_html_rejects_non_http() -> None:
    with pytest.raises(ValueError, match="http"):
        fetch_page_html("file:///etc/passwd")


def test_fetch_page_html_size_cap(http_server: Callable[[Responder], str]) -> None:
    body = b"A" * 4096

    def respond(_path: str) -> tuple[int, dict[str, str], bytes]:
        return (200, {"Content-Type": "text/html"}, body)

    base_url = http_server(respond)
    with pytest.raises(DownloadError, match="too large"):
        fetch_page_html(f"{base_url}/big.html", max_bytes=1024)


def test_discover_episode_links_end_to_end(http_server: Callable[[Responder], str]) -> None:
    body = (
        b"<html><body>"
        b"<a href='/audio/ep.mp3'>listen</a>"
        b"<a href='/t/ep.srt'>transcript</a>"
        b"</body></html>"
    )

    def respond(_path: str) -> tuple[int, dict[str, str], bytes]:
        return (200, {"Content-Type": "text/html"}, body)

    base_url = http_server(respond)
    info = discover_episode_links(f"{base_url}/episodes/42/")
    assert info.audio_url == f"{base_url}/audio/ep.mp3"
    assert len(info.transcripts) == 1
    assert info.transcripts[0].url == f"{base_url}/t/ep.srt"
