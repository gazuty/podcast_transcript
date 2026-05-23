"""Tests for :mod:`podcast_transcript.library.episode`."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from podcast_transcript.library.episode import (
    Episode,
    EpisodeValidationError,
    SourceUrls,
    SummaryRef,
    TranscriptRef,
    compute_transcript_checksum,
    make_episode_id,
    slugify,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# slugify / make_episode_id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("The Modern Lipid Playbook", "the-modern-lipid-playbook"),
        ("Dr. Hillary Lin, MD", "dr-hillary-lin-md"),
        ("ALL CAPS!!!", "all-caps"),
        ("  leading-trailing  ", "leading-trailing"),
        ("part 2: the testing", "part-2-the-testing"),
    ],
)
def test_slugify(raw: str, expected: str) -> None:
    assert slugify(raw) == expected


def test_slugify_rejects_empty() -> None:
    with pytest.raises(EpisodeValidationError):
        slugify("")
    with pytest.raises(EpisodeValidationError):
        slugify("!!!")


def test_make_episode_id_format() -> None:
    eid = make_episode_id(
        podcast_slug="longevity-show-lin",
        pub_date="2026-04-17",
        title="The Modern Lipid Playbook Part 2",
    )
    assert eid == "longevity-show-lin__2026-04-17__the-modern-lipid-playbook-part-2"


def test_make_episode_id_rejects_bad_date() -> None:
    with pytest.raises(EpisodeValidationError):
        make_episode_id(podcast_slug="x", pub_date="04/17/2026", title="Hello")


# ---------------------------------------------------------------------------
# Episode round-trip
# ---------------------------------------------------------------------------


def _make_valid_episode(**overrides: object) -> Episode:
    base = {
        "id": "show__2026-04-17__hello-world",
        "podcast": "Show",
        "podcast_slug": "show",
        "episode_title": "Hello World",
        "pub_date": "2026-04-17",
        "transcript": TranscriptRef(
            path="transcripts/show/show__2026-04-17__hello-world.txt",
            source="whisper",
            model="large-v3",
            has_timestamps=False,
        ),
        "summary": SummaryRef(
            path="summaries/show/show__2026-04-17__hello-world.md",
            generated_at="2026-04-18T12:00:00+00:00",
            model="claude-opus-4-7",
            qc_status="passed",
            qc_notes_path="summaries/show/show__2026-04-17__hello-world.qc.md",
        ),
        "ingested_at": "2026-04-18T12:00:00+00:00",
        "checksum": "0" * 64,
    }
    base.update(overrides)
    return Episode(**base)  # type: ignore[arg-type]


def test_episode_round_trip_through_json() -> None:
    ep = _make_valid_episode(
        speakers=["Hillary Lin"],
        topics=["ApoB"],
        guests=["A Guest"],
        source_urls=SourceUrls(audio="https://example.com/ep.mp3"),
    )
    ep.validate()
    payload = json.dumps(ep.to_dict())
    reloaded = Episode.from_dict(json.loads(payload))
    reloaded.validate()
    assert reloaded == ep


def test_episode_drops_empty_optionals() -> None:
    ep = _make_valid_episode()
    data = ep.to_dict()
    assert "guests" not in data
    assert "speakers" not in data
    assert "topics" not in data
    assert "source_urls" not in data
    assert "pending_topics" not in data


def test_episode_keeps_populated_optionals() -> None:
    ep = _make_valid_episode(pending_topics=["lipidology"])
    data = ep.to_dict()
    assert data["pending_topics"] == ["lipidology"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_rejects_bad_id() -> None:
    ep = _make_valid_episode(id="not-a-real-id")
    with pytest.raises(EpisodeValidationError, match="id"):
        ep.validate()


def test_validate_rejects_bad_pub_date() -> None:
    ep = _make_valid_episode(pub_date="April 17 2026")
    with pytest.raises(EpisodeValidationError, match="pub_date"):
        ep.validate()


def test_validate_rejects_bad_qc_status() -> None:
    bad_summary = SummaryRef(
        path="x",
        generated_at="2026-04-18T12:00:00+00:00",
        model="claude-opus-4-7",
        qc_status="approved",
    )
    ep = _make_valid_episode(summary=bad_summary)
    with pytest.raises(EpisodeValidationError, match="qc_status"):
        ep.validate()


def test_validate_rejects_bad_transcript_source() -> None:
    bad_transcript = TranscriptRef(path="x", source="ai", has_timestamps=False)
    ep = _make_valid_episode(transcript=bad_transcript)
    with pytest.raises(EpisodeValidationError, match=r"transcript\.source"):
        ep.validate()


def test_validate_rejects_bad_checksum() -> None:
    ep = _make_valid_episode(checksum="not-a-sha")
    with pytest.raises(EpisodeValidationError, match="checksum"):
        ep.validate()


def test_validate_rejects_unknown_fields_on_load() -> None:
    payload = _make_valid_episode().to_dict()
    payload["unexpected"] = "field"
    with pytest.raises(EpisodeValidationError, match="unknown episode fields"):
        Episode.from_dict(payload)


# ---------------------------------------------------------------------------
# Checksum
# ---------------------------------------------------------------------------


def test_compute_transcript_checksum_is_sha256(tmp_path: Path) -> None:
    f = tmp_path / "t.txt"
    f.write_text("hello world\n", encoding="utf-8")
    checksum = compute_transcript_checksum(f)
    # SHA-256 hex for "hello world\n"
    assert checksum == "a948904f2f0f479b8f8197694b30184b0d2ed1c1cd2a1ec0fb85d299a192a447"
