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
    with pytest.raises(PipelineError, match="not both"):
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
    assert result.audio_path.read_bytes() == AUDIO_BODY
