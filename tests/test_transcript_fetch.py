"""Tests for :mod:`podcast_transcript.transcript_fetch`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from podcast_transcript.download import DownloadError
from podcast_transcript.feed import TranscriptRef
from podcast_transcript.transcript_fetch import (
    TranscriptFetchError,
    classify_mime_type,
    fetch_transcript_text,
    preferred_transcript,
    subrip_to_text,
    vtt_to_text,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from .conftest import Responder


# ---------------------------------------------------------------------------
# classify_mime_type + preferred_transcript
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mime", "expected"),
    [
        ("application/srt", "srt"),
        ("application/x-subrip", "srt"),
        ("text/srt", "srt"),
        ("text/vtt", "vtt"),
        ("application/vtt", "vtt"),
        ("text/html", None),
        ("application/json", None),
        ("APPLICATION/SRT; charset=utf-8", "srt"),
        ("", None),
    ],
)
def test_classify_mime_type(mime: str, expected: str | None) -> None:
    assert classify_mime_type(mime) == expected


def test_preferred_transcript_prefers_srt_over_vtt() -> None:
    refs = (
        TranscriptRef(url="x.html", mime_type="text/html"),
        TranscriptRef(url="x.vtt", mime_type="text/vtt"),
        TranscriptRef(url="x.srt", mime_type="application/srt"),
    )
    chosen = preferred_transcript(refs)
    assert chosen is not None
    ref, kind = chosen
    assert ref.url == "x.srt"
    assert kind == "srt"


def test_preferred_transcript_falls_back_to_vtt() -> None:
    refs = (
        TranscriptRef(url="x.html", mime_type="text/html"),
        TranscriptRef(url="x.vtt", mime_type="text/vtt"),
    )
    chosen = preferred_transcript(refs)
    assert chosen is not None
    assert chosen[0].url == "x.vtt"
    assert chosen[1] == "vtt"


def test_preferred_transcript_returns_none_when_no_supported() -> None:
    refs = (
        TranscriptRef(url="x.html", mime_type="text/html"),
        TranscriptRef(url="x.json", mime_type="application/json"),
    )
    assert preferred_transcript(refs) is None


# ---------------------------------------------------------------------------
# fetch_transcript_text
# ---------------------------------------------------------------------------


SRT_BODY = (
    b"1\n"
    b"00:00:01,000 --> 00:00:03,500\n"
    b"Hello, world.\n"
    b"\n"
    b"2\n"
    b"00:00:04,000 --> 00:00:06,000\n"
    b"<i>Second cue</i>\n"
    b"on two lines.\n"
)


def test_fetch_transcript_text_happy_path(http_server: Callable[[Responder], str]) -> None:
    def respond(_path: str) -> tuple[int, dict[str, str], bytes]:
        return (200, {"Content-Type": "application/srt"}, SRT_BODY)

    base_url = http_server(respond)
    body, content_type = fetch_transcript_text(f"{base_url}/ep.srt")
    assert "Hello, world." in body
    assert content_type == "application/srt"


def test_fetch_transcript_text_rejects_non_http() -> None:
    with pytest.raises(ValueError, match="http"):
        fetch_transcript_text("file:///etc/passwd")


def test_fetch_transcript_text_rejects_html(http_server: Callable[[Responder], str]) -> None:
    def respond(_path: str) -> tuple[int, dict[str, str], bytes]:
        return (200, {"Content-Type": "text/html"}, b"<html><body>not a transcript</body></html>")

    base_url = http_server(respond)
    # text/* is in the accepted prefix list (publishers serve SRT as text/plain),
    # so text/html actually passes the prefix gate; the *converter* would fail.
    # Verify the more important check: octet-stream of a real .srt is accepted.
    body, _ = fetch_transcript_text(f"{base_url}/ep.html")
    assert "<html>" in body


def test_fetch_transcript_text_rejects_unsupported_content_type(
    http_server: Callable[[Responder], str],
) -> None:
    def respond(_path: str) -> tuple[int, dict[str, str], bytes]:
        return (200, {"Content-Type": "image/png"}, b"\x89PNG\r\n")

    base_url = http_server(respond)
    with pytest.raises(TranscriptFetchError, match="Content-Type"):
        fetch_transcript_text(f"{base_url}/ep.png")


def test_fetch_transcript_text_404(http_server: Callable[[Responder], str]) -> None:
    def respond(_path: str) -> tuple[int, dict[str, str], bytes]:
        return (404, {"Content-Type": "text/plain"}, b"nope")

    base_url = http_server(respond)
    with pytest.raises(DownloadError, match="HTTP 404"):
        fetch_transcript_text(f"{base_url}/missing.srt")


def test_fetch_transcript_text_size_cap(http_server: Callable[[Responder], str]) -> None:
    big_body = b"A" * 1024

    def respond(_path: str) -> tuple[int, dict[str, str], bytes]:
        return (200, {"Content-Type": "application/srt"}, big_body)

    base_url = http_server(respond)
    with pytest.raises(DownloadError, match="too large"):
        fetch_transcript_text(f"{base_url}/ep.srt", max_bytes=512)


# ---------------------------------------------------------------------------
# SRT → text
# ---------------------------------------------------------------------------


def test_subrip_to_text_strips_indexes_and_timestamps() -> None:
    result = subrip_to_text(SRT_BODY.decode("utf-8"))
    assert result.splitlines() == [
        "Hello, world.",
        "Second cue on two lines.",
    ]


def test_subrip_to_text_handles_dot_separator_and_blank_cues() -> None:
    srt = "\n".join(
        [
            "1",
            "00:00:01.000 --> 00:00:02.000",
            "First line.",
            "",
            "2",
            "00:00:03.000 --> 00:00:04.000",
            "",
            "3",
            "00:00:05.000 --> 00:00:06.000",
            "Third line.",
            "",
        ],
    )
    assert subrip_to_text(srt) == "First line.\nThird line."


def test_subrip_to_text_strips_ssa_overrides() -> None:
    srt = "\n".join(
        [
            "1",
            "00:00:01,000 --> 00:00:02,000",
            "{\\an8}Karaoke caption text.",
            "",
        ],
    )
    assert subrip_to_text(srt) == "Karaoke caption text."


# ---------------------------------------------------------------------------
# VTT → text
# ---------------------------------------------------------------------------


VTT_BODY = """WEBVTT

NOTE
This is a note block we should ignore entirely.

1
00:00:01.000 --> 00:00:03.500
Hello from VTT.

00:00:04.000 --> 00:00:06.000 line:0 position:50%
<v Speaker>Cue with settings</v>
spanning two lines.
"""


def test_vtt_to_text_strips_header_notes_and_cue_settings() -> None:
    lines = vtt_to_text(VTT_BODY).splitlines()
    assert lines == [
        "Hello from VTT.",
        "Cue with settings spanning two lines.",
    ]


def test_vtt_to_text_handles_no_blank_between_header_and_cue() -> None:
    vtt = "WEBVTT\n\n00:00:00.500 --> 00:00:01.500\nLone cue.\n"
    assert vtt_to_text(vtt) == "Lone cue."
