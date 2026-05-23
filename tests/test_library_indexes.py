"""Tests for :mod:`podcast_transcript.library.indexes`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from podcast_transcript.library.episode import (
    Episode,
    SummaryRef,
    TranscriptRef,
)
from podcast_transcript.library.indexes import (
    build_by_date,
    build_by_podcast,
    build_by_speaker,
    build_by_topic,
    build_pending_vocab,
    rebuild_all,
)
from podcast_transcript.library.store import save_all

if TYPE_CHECKING:
    from pathlib import Path


def _episode(
    id_: str,
    *,
    podcast: str = "Show",
    speakers: list[str] | None = None,
    topics: list[str] | None = None,
    series: str | None = None,
    series_part: int | None = None,
    qc_status: str = "passed",
    pending_topics: list[str] | None = None,
    pending_speakers: list[str] | None = None,
) -> Episode:
    return Episode(
        id=id_,
        podcast=podcast,
        podcast_slug=id_.split("__")[0],
        episode_title=id_.split("__")[2].replace("-", " ").title(),
        pub_date=id_.split("__")[1],
        speakers=speakers or [],
        topics=topics or [],
        series=series,
        series_part=series_part,
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
        pending_topics=pending_topics or [],
        pending_speakers=pending_speakers or [],
    )


def test_by_date_is_reverse_chronological() -> None:
    eps = [
        _episode("a__2026-04-17__ep-one"),
        _episode("b__2026-05-01__ep-two"),
        _episode("c__2026-03-01__ep-three"),
    ]
    body = build_by_date(eps)
    assert "2026-05-01" in body
    # Newest must appear first
    assert body.index("2026-05-01") < body.index("2026-04-17") < body.index("2026-03-01")


def test_by_date_links_relative_to_index_dir() -> None:
    ep = _episode("a__2026-04-17__ep-one")
    body = build_by_date([ep])
    assert "../summaries/a__2026-04-17__ep-one.md" in body


def test_by_podcast_groups_and_subgroups_series() -> None:
    eps = [
        _episode(
            "show__2026-04-17__part-1",
            podcast="Lipid Show",
            series="Lipid Playbook",
            series_part=1,
        ),
        _episode(
            "show__2026-04-24__part-2",
            podcast="Lipid Show",
            series="Lipid Playbook",
            series_part=2,
        ),
        _episode("show__2026-05-01__standalone", podcast="Lipid Show"),
    ]
    body = build_by_podcast(eps)
    assert "## Lipid Show" in body
    assert "### Lipid Playbook" in body
    assert "### (standalone episodes)" in body
    # Part 1 must come before Part 2 within the series block
    assert body.index("Part 1:") < body.index("Part 2:")


def test_by_speaker_lists_episodes_under_each_speaker() -> None:
    eps = [
        _episode("a__2026-04-17__one", speakers=["Hillary Lin"]),
        _episode("b__2026-04-18__two", speakers=["Hillary Lin", "Andrej Karpathy"]),
    ]
    body = build_by_speaker(eps)
    assert "## Andrej Karpathy" in body
    assert "## Hillary Lin" in body
    # "Hillary Lin" appears in both episodes, each as a list item
    assert body.count("__one") == 1
    assert body.count("__two") == 2  # once under each speaker


def test_by_topic_lists_episodes_under_each_topic() -> None:
    eps = [
        _episode("a__2026-04-17__one", topics=["ApoB", "lipidology"]),
        _episode("b__2026-04-18__two", topics=["ApoB"]),
    ]
    body = build_by_topic(eps)
    assert "## ApoB" in body
    assert "## lipidology" in body
    assert body.count("__one") == 2
    assert body.count("__two") == 1


def test_qc_badge_appears_for_non_passed() -> None:
    ep = _episode("a__2026-04-17__one", qc_status="flagged")
    body = build_by_date([ep])
    assert "_(QC: flagged)_" in body


def test_pending_vocab_groups_by_term() -> None:
    eps = [
        _episode(
            "a__2026-04-17__one",
            pending_topics=["lipidology"],
            pending_speakers=["Andrej Karpathy"],
        ),
        _episode(
            "b__2026-04-18__two",
            pending_topics=["lipidology"],
        ),
    ]
    body = build_pending_vocab(eps)
    assert "## Topics" in body
    assert "## Speakers" in body
    assert "**lipidology**" in body
    assert "**Andrej Karpathy**" in body
    # Both episodes show up under lipidology
    assert body.count("__one") >= 1
    assert body.count("__two") >= 1


def test_rebuild_all_writes_five_files(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    jsonl = index_dir / "episodes.jsonl"
    save_all(jsonl, [_episode("a__2026-04-17__one", topics=["ApoB"], speakers=["X"])])

    written = rebuild_all(index_dir=index_dir, jsonl_path=jsonl)
    expected = {"by-speaker.md", "by-topic.md", "by-date.md", "by-podcast.md", "pending-vocab.md"}
    assert set(written.keys()) == expected
    for path in written.values():
        assert path.is_file()
        assert path.read_text(encoding="utf-8").startswith("# ")


def test_rebuild_all_idempotent_body_apart_from_timestamp(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    jsonl = index_dir / "episodes.jsonl"
    save_all(jsonl, [_episode("a__2026-04-17__one", topics=["X"])])

    first = rebuild_all(index_dir=index_dir, jsonl_path=jsonl)
    body1 = first["by-date.md"].read_text(encoding="utf-8")
    second = rebuild_all(index_dir=index_dir, jsonl_path=jsonl)
    body2 = second["by-date.md"].read_text(encoding="utf-8")

    # The body apart from the "Generated at" line should be byte-identical.
    def _strip_stamp(body: str) -> str:
        return "\n".join(line for line in body.splitlines() if "Generated by" not in line)

    assert _strip_stamp(body1) == _strip_stamp(body2)


def test_indexes_handle_empty_jsonl(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    jsonl = index_dir / "episodes.jsonl"
    jsonl.write_text("", encoding="utf-8")

    written = rebuild_all(index_dir=index_dir, jsonl_path=jsonl)
    body = written["by-date.md"].read_text(encoding="utf-8")
    assert "No episodes yet" in body
