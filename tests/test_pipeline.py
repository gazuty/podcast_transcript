"""Tests for :mod:`podcast_transcript.pipeline`."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from podcast_transcript.clean import CorrectionsFile
from podcast_transcript.pipeline import (
    PipelineError,
    apply_strip_patterns,
    run_pipeline,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from unittest.mock import MagicMock

    from .conftest import Responder


# ---------------------------------------------------------------------------
# apply_strip_patterns
# ---------------------------------------------------------------------------


def test_strip_before_drops_through_first_match() -> None:
    text = "ad line 1\nad line 2\n[host name] hello\nbody.\nbody.\n"
    out, head, tail = apply_strip_patterns(
        text,
        strip_before=[r"host name"],
        strip_after=[],
    )
    assert out == "body.\nbody."
    assert head == 3
    assert tail == 0


def test_strip_after_only_chops_when_match_in_tail_half() -> None:
    text = "\n".join(["body"] * 9 + ["thanks for listening"])
    out, _head, tail = apply_strip_patterns(
        text,
        strip_before=[],
        strip_after=[r"thanks for listening"],
    )
    assert out == "\n".join(["body"] * 9)
    assert tail == 1


def test_strip_after_ignores_match_in_head_half() -> None:
    # "thanks for listening" appearing in line 1 of 100 should not chop the body.
    text = "\n".join(["thanks for listening, here's today's show"] + ["body"] * 99)
    out, _head, tail = apply_strip_patterns(
        text,
        strip_before=[],
        strip_after=[r"thanks for listening"],
    )
    assert tail == 0
    assert out.count("body") == 99


def test_strip_patterns_repeatable() -> None:
    text = "preroll1\npreroll2\nepisode start\nbody\nbody\n"
    out, head, _tail = apply_strip_patterns(
        text,
        strip_before=[r"preroll1", r"episode start"],
        strip_after=[],
    )
    assert out == "body\nbody"
    assert head == 3


# ---------------------------------------------------------------------------
# run_pipeline (URL mode, fully mocked)
# ---------------------------------------------------------------------------


AUDIO_BODY = b"ID3\x04\x00" + b"\x00" * 1024


def _audio_responder() -> Responder:
    def respond(_path: str) -> tuple[int, dict[str, str], bytes]:
        return (200, {"Content-Type": "audio/mpeg"}, AUDIO_BODY)

    return respond


def _wire_fake_whisper(fake_whisper: MagicMock, transcripts_dir: Path, slug: str) -> None:
    """Make the fake whisper writer drop a real .txt on disk so the pipeline can read it."""
    raw_text = "\n".join(
        [
            "Razeeb Khan here.",
            "Stephen Ghazal said something interesting.",
            "Real body content.",
        ]
    )

    transcripts_dir.mkdir(parents=True, exist_ok=True)

    def writer_factory(fmt: str, _out_dir: str) -> object:
        def write(_result: object, audio_path: str) -> None:
            stem = Path(audio_path).stem
            (transcripts_dir / f"{stem}.{fmt}").write_text(raw_text, encoding="utf-8")

        return write

    fake_whisper.utils.get_writer.side_effect = writer_factory
    fake_whisper.load_model.return_value.transcribe.return_value = {
        "text": raw_text,
        "segments": [],
    }
    # Slug is used by the caller; reading it here is a hint for grep/debugging.
    assert slug


def test_run_pipeline_url_mode_writes_clean_file(
    tmp_path: Path,
    http_server: Callable[[Responder], str],
    fake_whisper: MagicMock,
) -> None:
    base_url = http_server(_audio_responder())
    audio_dir = tmp_path / "audio"
    transcripts_dir = tmp_path / "transcripts"
    _wire_fake_whisper(fake_whisper, transcripts_dir, "show1")

    result = run_pipeline(
        url=f"{base_url}/episode.mp3",
        rss_url=None,
        episode_regex=None,
        episode_index=None,
        slug="show1",
        audio_dir=audio_dir,
        transcripts_dir=transcripts_dir,
        corrections=CorrectionsFile(
            corrections={"Razeeb": "Razib"},
            uncertain={"Stephen Ghazal": "Stephen Gazal"},
        ),
        strip_before=[],
        strip_after=[],
        timeout=5.0,
    )

    assert result.audio_path == audio_dir / "show1.mp3"
    assert result.audio_path.read_bytes() == AUDIO_BODY
    assert result.raw_transcript_path == transcripts_dir / "show1.txt"
    assert result.clean_transcript_path == transcripts_dir / "show1_clean.txt"

    cleaned = result.clean_transcript_path.read_text(encoding="utf-8")
    assert "Razib Khan" in cleaned
    assert "[?: Stephen Ghazal → Stephen Gazal]" in cleaned
    assert result.stats.corrections_applied >= 1
    assert result.stats.uncertain_applied == 1


def test_run_pipeline_rejects_both_url_and_rss(tmp_path: Path) -> None:
    with pytest.raises(PipelineError, match="exactly one"):
        run_pipeline(
            url="https://x/y.mp3",
            rss_url="https://x/feed.xml",
            episode_regex=None,
            episode_index=None,
            slug="x",
            audio_dir=tmp_path,
            transcripts_dir=tmp_path,
            corrections=CorrectionsFile(),
            strip_before=[],
            strip_after=[],
            timeout=5.0,
        )


def test_run_pipeline_rss_requires_selector(tmp_path: Path) -> None:
    with pytest.raises(PipelineError, match="--episode-regex"):
        run_pipeline(
            url=None,
            rss_url="https://x/feed.xml",
            episode_regex=None,
            episode_index=None,
            slug="x",
            audio_dir=tmp_path,
            transcripts_dir=tmp_path,
            corrections=CorrectionsFile(),
            strip_before=[],
            strip_after=[],
            timeout=5.0,
        )


def test_run_pipeline_rss_mode_picks_episode_by_regex(
    tmp_path: Path,
    http_server: Callable[[Responder], str],
    fake_whisper: MagicMock,
) -> None:
    rss_body = b"""<?xml version="1.0"?>
    <rss version="2.0"><channel>
      <item><title>Wrong One</title><enclosure url="REPLACE_ME_WRONG" type="audio/mpeg"/></item>
      <item><title>Selection in Western Eurasia</title><enclosure url="REPLACE_ME_RIGHT" type="audio/mpeg"/></item>
    </channel></rss>
    """

    audio_dir = tmp_path / "audio"
    transcripts_dir = tmp_path / "transcripts"
    _wire_fake_whisper(fake_whisper, transcripts_dir, "selection")

    def respond(path: str) -> tuple[int, dict[str, str], bytes]:
        if path.endswith(".mp3"):
            return (200, {"Content-Type": "audio/mpeg"}, AUDIO_BODY)
        # Otherwise it's the feed. Patch in actual mp3 URLs pointing at this server.
        return (
            200,
            {"Content-Type": "application/rss+xml"},
            rss_body.replace(b"REPLACE_ME_WRONG", b"http://_/wrong.mp3").replace(
                b"REPLACE_ME_RIGHT",
                f"{http_server_base_url}/right.mp3".encode(),
            ),
        )

    # Server URL needs to be resolvable inside the responder, so we start the
    # server first, capture its base URL, then close the loop.
    http_server_base_url: str = http_server(respond)

    result = run_pipeline(
        url=None,
        rss_url=f"{http_server_base_url}/feed.xml",
        episode_regex=r"Selection in Western",
        episode_index=None,
        slug="selection",
        audio_dir=audio_dir,
        transcripts_dir=transcripts_dir,
        corrections=CorrectionsFile(),
        strip_before=[],
        strip_after=[],
        timeout=5.0,
    )

    assert result.feed_item is not None
    assert "Selection" in result.feed_item.title
    assert result.audio_path is not None
    assert result.audio_path.read_bytes() == AUDIO_BODY
    assert result.transcript_source == "whisper"


# ---------------------------------------------------------------------------
# Publisher-transcript discovery (RSS path)
# ---------------------------------------------------------------------------


SRT_BODY = (
    b"1\n"
    b"00:00:01,000 --> 00:00:03,000\n"
    b"Razeeb Khan welcomes you.\n"
    b"\n"
    b"2\n"
    b"00:00:03,500 --> 00:00:06,000\n"
    b"Real episode body content.\n"
)


def test_run_pipeline_rss_uses_publisher_transcript_when_present(
    tmp_path: Path,
    http_server: Callable[[Responder], str],
    fake_whisper: MagicMock,
) -> None:
    """When the RSS item declares a <podcast:transcript>, skip Whisper entirely."""
    rss_template = (
        b'<?xml version="1.0"?>\n'
        b'<rss version="2.0" xmlns:podcast="https://podcastindex.org/namespace/1.0">\n'
        b"<channel>\n"
        b"  <item>\n"
        b"    <title>Episode One</title>\n"
        b'    <enclosure url="SHOULD_NOT_BE_CALLED" type="audio/mpeg"/>\n'
        b'    <podcast:transcript url="TRANSCRIPT_URL" type="application/srt"/>\n'
        b"  </item>\n"
        b"</channel></rss>\n"
    )

    audio_dir = tmp_path / "audio"
    transcripts_dir = tmp_path / "transcripts"

    def respond(path: str) -> tuple[int, dict[str, str], bytes]:
        if path.endswith(".srt"):
            return (200, {"Content-Type": "application/srt"}, SRT_BODY)
        return (
            200,
            {"Content-Type": "application/rss+xml"},
            rss_template.replace(
                b"TRANSCRIPT_URL",
                f"{base_url}/ep.srt".encode(),
            ),
        )

    base_url: str = http_server(respond)

    result = run_pipeline(
        url=None,
        rss_url=f"{base_url}/feed.xml",
        episode_regex=r"Episode One",
        episode_index=None,
        slug="ep1",
        audio_dir=audio_dir,
        transcripts_dir=transcripts_dir,
        corrections=CorrectionsFile(corrections={"Razeeb": "Razib"}),
        strip_before=[],
        strip_after=[],
        timeout=5.0,
    )

    assert result.transcript_source == "rss"
    assert result.audio_path is None
    # Crucial: Whisper never ran.
    fake_whisper.load_model.assert_not_called()
    cleaned = result.clean_transcript_path.read_text(encoding="utf-8")
    assert "Razib Khan" in cleaned
    raw = result.raw_transcript_path.read_text(encoding="utf-8")
    assert "Razeeb Khan welcomes you." in raw


def test_run_pipeline_falls_back_to_whisper_when_publisher_transcript_unusable(
    tmp_path: Path,
    http_server: Callable[[Responder], str],
    fake_whisper: MagicMock,
) -> None:
    """A declared transcript that isn't real captions must not kill the run."""
    rss_template = (
        b'<?xml version="1.0"?>\n'
        b'<rss version="2.0" xmlns:podcast="https://podcastindex.org/namespace/1.0">\n'
        b"<channel>\n"
        b"  <item>\n"
        b"    <title>Episode One</title>\n"
        b'    <enclosure url="AUDIO_URL" type="audio/mpeg"/>\n'
        b'    <podcast:transcript url="TRANSCRIPT_URL" type="application/srt"/>\n'
        b"  </item>\n"
        b"</channel></rss>\n"
    )
    audio_dir = tmp_path / "audio"
    transcripts_dir = tmp_path / "transcripts"
    _wire_fake_whisper(fake_whisper, transcripts_dir, "ep1")

    def respond(path: str) -> tuple[int, dict[str, str], bytes]:
        if path.endswith(".srt"):
            # Mislabelled: right content type, but the body is an error page
            # with no cue timestamps in it.
            return (200, {"Content-Type": "application/srt"}, b"Sorry, that file has moved.")
        if path.endswith(".mp3"):
            return (200, {"Content-Type": "audio/mpeg"}, AUDIO_BODY)
        return (
            200,
            {"Content-Type": "application/rss+xml"},
            rss_template.replace(b"TRANSCRIPT_URL", f"{base_url}/ep.srt".encode()).replace(
                b"AUDIO_URL", f"{base_url}/ep.mp3".encode()
            ),
        )

    base_url: str = http_server(respond)
    result = run_pipeline(
        url=None,
        rss_url=f"{base_url}/feed.xml",
        episode_regex=r"Episode One",
        episode_index=None,
        slug="ep1",
        audio_dir=audio_dir,
        transcripts_dir=transcripts_dir,
        corrections=CorrectionsFile(),
        strip_before=[],
        strip_after=[],
        timeout=5.0,
    )

    assert result.transcript_source == "whisper"
    assert result.audio_path is not None
    fake_whisper.load_model.assert_called_once()


def test_run_pipeline_falls_back_when_feed_declares_non_http_transcript(
    tmp_path: Path,
    http_server: Callable[[Responder], str],
    fake_whisper: MagicMock,
) -> None:
    """A feed-declared file:// transcript URL is refused but doesn't kill the run."""
    rss_template = (
        b'<?xml version="1.0"?>\n'
        b'<rss version="2.0" xmlns:podcast="https://podcastindex.org/namespace/1.0">\n'
        b"<channel>\n"
        b"  <item>\n"
        b"    <title>Episode One</title>\n"
        b'    <enclosure url="AUDIO_URL" type="audio/mpeg"/>\n'
        b'    <podcast:transcript url="file:///etc/captions.srt" type="application/srt"/>\n'
        b"  </item>\n"
        b"</channel></rss>\n"
    )
    transcripts_dir = tmp_path / "transcripts"
    _wire_fake_whisper(fake_whisper, transcripts_dir, "ep1")

    def respond(path: str) -> tuple[int, dict[str, str], bytes]:
        if path.endswith(".mp3"):
            return (200, {"Content-Type": "audio/mpeg"}, AUDIO_BODY)
        return (
            200,
            {"Content-Type": "application/rss+xml"},
            rss_template.replace(b"AUDIO_URL", f"{base_url}/ep.mp3".encode()),
        )

    base_url: str = http_server(respond)
    result = run_pipeline(
        url=None,
        rss_url=f"{base_url}/feed.xml",
        episode_regex=r"Episode One",
        episode_index=None,
        slug="ep1",
        audio_dir=tmp_path / "audio",
        transcripts_dir=transcripts_dir,
        corrections=CorrectionsFile(),
        strip_before=[],
        strip_after=[],
        timeout=5.0,
    )

    assert result.transcript_source == "whisper"


def test_run_pipeline_rss_falls_back_to_whisper_when_no_transcript(
    tmp_path: Path,
    http_server: Callable[[Responder], str],
    fake_whisper: MagicMock,
) -> None:
    rss_body = (
        b'<?xml version="1.0"?>'
        b'<rss version="2.0"><channel>'
        b"<item><title>Plain</title>"
        b'<enclosure url="AUDIO" type="audio/mpeg"/></item>'
        b"</channel></rss>"
    )
    audio_dir = tmp_path / "audio"
    transcripts_dir = tmp_path / "transcripts"
    _wire_fake_whisper(fake_whisper, transcripts_dir, "plain")

    def respond(path: str) -> tuple[int, dict[str, str], bytes]:
        if path.endswith(".mp3"):
            return (200, {"Content-Type": "audio/mpeg"}, AUDIO_BODY)
        return (
            200,
            {"Content-Type": "application/rss+xml"},
            rss_body.replace(b"AUDIO", f"{base_url}/ep.mp3".encode()),
        )

    base_url: str = http_server(respond)
    result = run_pipeline(
        url=None,
        rss_url=f"{base_url}/feed.xml",
        episode_regex=r"Plain",
        episode_index=None,
        slug="plain",
        audio_dir=audio_dir,
        transcripts_dir=transcripts_dir,
        corrections=CorrectionsFile(),
        strip_before=[],
        strip_after=[],
        timeout=5.0,
    )
    assert result.transcript_source == "whisper"
    assert result.audio_path is not None
    fake_whisper.load_model.assert_called_once()


def test_run_pipeline_no_discover_transcript_forces_whisper(
    tmp_path: Path,
    http_server: Callable[[Responder], str],
    fake_whisper: MagicMock,
) -> None:
    """Even when a publisher transcript is declared, --no-discover-transcript wins."""
    rss_template = (
        b'<?xml version="1.0"?>\n'
        b'<rss version="2.0" xmlns:podcast="https://podcastindex.org/namespace/1.0">\n'
        b"<channel><item><title>Has Transcript</title>"
        b'<enclosure url="AUDIO" type="audio/mpeg"/>'
        b'<podcast:transcript url="TRANSCRIPT" type="application/srt"/>'
        b"</item></channel></rss>"
    )
    audio_dir = tmp_path / "audio"
    transcripts_dir = tmp_path / "transcripts"
    _wire_fake_whisper(fake_whisper, transcripts_dir, "forced")

    def respond(path: str) -> tuple[int, dict[str, str], bytes]:
        if path.endswith(".mp3"):
            return (200, {"Content-Type": "audio/mpeg"}, AUDIO_BODY)
        if path.endswith(".srt"):  # would be a test failure if called
            return (200, {"Content-Type": "application/srt"}, SRT_BODY)
        return (
            200,
            {"Content-Type": "application/rss+xml"},
            rss_template.replace(b"AUDIO", f"{base_url}/ep.mp3".encode()).replace(
                b"TRANSCRIPT",
                f"{base_url}/ep.srt".encode(),
            ),
        )

    base_url: str = http_server(respond)
    result = run_pipeline(
        url=None,
        rss_url=f"{base_url}/feed.xml",
        episode_regex=r"Has Transcript",
        episode_index=None,
        slug="forced",
        audio_dir=audio_dir,
        transcripts_dir=transcripts_dir,
        corrections=CorrectionsFile(),
        strip_before=[],
        strip_after=[],
        timeout=5.0,
        discover_transcript=False,
    )
    assert result.transcript_source == "whisper"
    fake_whisper.load_model.assert_called_once()


# ---------------------------------------------------------------------------
# Page-URL source
# ---------------------------------------------------------------------------


def test_run_pipeline_page_uses_transcript_link(
    tmp_path: Path,
    http_server: Callable[[Responder], str],
    fake_whisper: MagicMock,
) -> None:
    transcripts_dir = tmp_path / "transcripts"

    def respond(path: str) -> tuple[int, dict[str, str], bytes]:
        if path.endswith(".srt"):
            return (200, {"Content-Type": "application/srt"}, SRT_BODY)
        if path.endswith(".mp3"):
            return (200, {"Content-Type": "audio/mpeg"}, AUDIO_BODY)
        html = (
            f'<a href="{base_url}/audio/ep.mp3">audio</a>'
            f'<a href="{base_url}/t/ep.srt">transcript</a>'
        ).encode()
        return (200, {"Content-Type": "text/html"}, html)

    base_url: str = http_server(respond)
    result = run_pipeline(
        url=None,
        rss_url=None,
        page_url=f"{base_url}/episode/1",
        episode_regex=None,
        episode_index=None,
        slug="page1",
        audio_dir=tmp_path / "audio",
        transcripts_dir=transcripts_dir,
        corrections=CorrectionsFile(),
        strip_before=[],
        strip_after=[],
        timeout=5.0,
    )
    assert result.transcript_source == "page"
    assert result.audio_path is None
    fake_whisper.load_model.assert_not_called()
    cleaned = result.clean_transcript_path.read_text(encoding="utf-8")
    assert "Real episode body content." in cleaned


def test_run_pipeline_page_falls_back_to_audio_when_no_transcript(
    tmp_path: Path,
    http_server: Callable[[Responder], str],
    fake_whisper: MagicMock,
) -> None:
    transcripts_dir = tmp_path / "transcripts"
    _wire_fake_whisper(fake_whisper, transcripts_dir, "page2")

    def respond(path: str) -> tuple[int, dict[str, str], bytes]:
        if path.endswith(".mp3"):
            return (200, {"Content-Type": "audio/mpeg"}, AUDIO_BODY)
        html = f'<a href="{base_url}/audio/ep.mp3">audio</a>'.encode()
        return (200, {"Content-Type": "text/html"}, html)

    base_url: str = http_server(respond)
    result = run_pipeline(
        url=None,
        rss_url=None,
        page_url=f"{base_url}/episode/2",
        episode_regex=None,
        episode_index=None,
        slug="page2",
        audio_dir=tmp_path / "audio",
        transcripts_dir=transcripts_dir,
        corrections=CorrectionsFile(),
        strip_before=[],
        strip_after=[],
        timeout=5.0,
    )
    assert result.transcript_source == "whisper"
    assert result.audio_path is not None
    fake_whisper.load_model.assert_called_once()


def test_run_pipeline_page_with_neither_link_errors(
    tmp_path: Path,
    http_server: Callable[[Responder], str],
) -> None:
    def respond(_path: str) -> tuple[int, dict[str, str], bytes]:
        return (200, {"Content-Type": "text/html"}, b"<html>nothing</html>")

    base_url = http_server(respond)
    with pytest.raises(PipelineError, match="no audio or transcript"):
        run_pipeline(
            url=None,
            rss_url=None,
            page_url=f"{base_url}/episode/3",
            episode_regex=None,
            episode_index=None,
            slug="page3",
            audio_dir=tmp_path / "audio",
            transcripts_dir=tmp_path / "transcripts",
            corrections=CorrectionsFile(),
            strip_before=[],
            strip_after=[],
            timeout=5.0,
        )
