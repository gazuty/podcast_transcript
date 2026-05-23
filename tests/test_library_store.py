"""Tests for :mod:`podcast_transcript.library.store`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from podcast_transcript.library.episode import (
    Episode,
    EpisodeValidationError,
    SummaryRef,
    TranscriptRef,
)
from podcast_transcript.library.store import load_all, load_index, save_all, upsert

if TYPE_CHECKING:
    from pathlib import Path


def _episode(id_: str, *, qc_status: str = "passed") -> Episode:
    return Episode(
        id=id_,
        podcast="Show",
        podcast_slug=id_.split("__")[0],
        episode_title="Hello",
        pub_date=id_.split("__")[1],
        transcript=TranscriptRef(
            path=f"transcripts/{id_}.txt",
            source="whisper",
            model="large-v3",
            has_timestamps=False,
        ),
        summary=SummaryRef(
            path=f"summaries/{id_}.md",
            generated_at="2026-04-18T12:00:00+00:00",
            model="claude-opus-4-7",
            qc_status=qc_status,
        ),
        ingested_at="2026-04-18T12:00:00+00:00",
        checksum="0" * 64,
    )


def test_load_all_returns_empty_for_missing_file(tmp_path: Path) -> None:
    assert load_all(tmp_path / "nope.jsonl") == []


def test_save_all_writes_sorted_by_id(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    eps = [
        _episode("show__2026-04-17__b-episode"),
        _episode("show__2026-04-17__a-episode"),
    ]
    save_all(path, eps)
    raw = path.read_text(encoding="utf-8").splitlines()
    assert len(raw) == 2
    # Sorted alphabetically by id
    assert "a-episode" in raw[0]
    assert "b-episode" in raw[1]


def test_save_all_rejects_invalid_episode(tmp_path: Path) -> None:
    bad = _episode("show__2026-04-17__hello", qc_status="approved")
    with pytest.raises(EpisodeValidationError):
        save_all(tmp_path / "episodes.jsonl", [bad])


def test_load_all_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    eps = [_episode("show__2026-04-17__a"), _episode("show__2026-04-17__b")]
    save_all(path, eps)
    loaded = load_all(path)
    assert [e.id for e in loaded] == [
        "show__2026-04-17__a",
        "show__2026-04-17__b",
    ]


def test_load_all_tolerates_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    save_all(path, [_episode("show__2026-04-17__a")])
    # Add a couple of blank lines manually
    path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
    assert len(load_all(path)) == 1


def test_load_all_raises_on_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(EpisodeValidationError, match="invalid JSON"):
        load_all(path)


def test_load_index_detects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    ep = _episode("show__2026-04-17__a")
    # Manually duplicate the line
    payload = "\n".join([__as_json(ep), __as_json(ep)])
    path.write_text(payload + "\n", encoding="utf-8")
    with pytest.raises(EpisodeValidationError, match="duplicate"):
        load_index(path)


def __as_json(ep: Episode) -> str:
    import json

    return json.dumps(ep.to_dict(), sort_keys=False)


def test_upsert_inserts_then_replaces(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    ep_v1 = _episode("show__2026-04-17__a", qc_status="passed")
    assert upsert(path, ep_v1) is False
    assert len(load_all(path)) == 1

    ep_v2 = _episode("show__2026-04-17__a", qc_status="flagged")
    assert upsert(path, ep_v2) is True
    rows = load_all(path)
    assert len(rows) == 1
    assert rows[0].summary.qc_status == "flagged"
