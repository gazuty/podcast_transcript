"""Integration test for :mod:`podcast_transcript.library.ingest`.

Exercises the full orchestration on a pre-supplied transcript so we
don't have to drive the Whisper pipeline. The QC client is faked via
the ``fake_anthropic`` fixture from :mod:`conftest`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from podcast_transcript.library.indexes import rebuild_all
from podcast_transcript.library.ingest import (
    IngestPaths,
    IngestRequest,
    ingest_episode,
)
from podcast_transcript.library.store import load_index
from podcast_transcript.library.vocab import load_vocab

if TYPE_CHECKING:
    from pathlib import Path

    from .conftest import FakeAnthropic


SAMPLE_TRANSCRIPT = "\n".join(
    [
        "Welcome back to the Longevity Show, I'm Hillary Lin.",
        "Today's episode is about ApoB and Lp(a) testing.",
        "ApoB is a more accurate marker than LDL-C alone.",
        "Lp(a) is genetically determined and worth measuring once.",
    ]
)


SAMPLE_SUMMARY_MD = """# The Modern Lipid Playbook Part 2

## TL;DR
Hillary Lin walks through ApoB and Lp(a) testing.

## Key points
- ApoB is more accurate than LDL-C.

## Key learnings / takeaways
- Measure Lp(a) at least once in your life.

## Notable quotes
> "ApoB is a more accurate marker than LDL-C alone." — Hillary Lin

## Numbers, studies, named entities
- ApoB
- Lp(a)

## Open questions / things to verify
- _None._

## Glossary (if technical)
- ApoB — apolipoprotein B.
- Lp(a) — lipoprotein little a.
"""


def _seed_library(library_root: Path) -> IngestPaths:
    paths = IngestPaths(library_root=library_root)
    paths.transcripts_dir.mkdir(parents=True, exist_ok=True)
    paths.summaries_dir.mkdir(parents=True, exist_ok=True)
    paths.audio_dir.mkdir(parents=True, exist_ok=True)
    (paths.index_dir / "vocab").mkdir(parents=True, exist_ok=True)
    paths.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    paths.jsonl_path.write_text("", encoding="utf-8")
    # Seed empty vocab files so we can observe pending entries getting added.
    paths.topics_path.write_text(json.dumps({"canonical": {}, "aliases": {}}), encoding="utf-8")
    paths.speakers_path.write_text(json.dumps({"canonical": {}, "aliases": {}}), encoding="utf-8")
    return paths


def test_ingest_episode_end_to_end(tmp_path: Path, fake_anthropic: FakeAnthropic) -> None:
    paths = _seed_library(tmp_path / "library")

    # First call is the summariser stream, second is the QC `messages.create`
    fake_anthropic.enqueue_stream(SAMPLE_SUMMARY_MD)
    fake_anthropic.enqueue_create(json.dumps({"verdict": "passed", "issues": []}))

    transcript_src = tmp_path / "raw_transcript.txt"
    transcript_src.write_text(SAMPLE_TRANSCRIPT, encoding="utf-8")

    request = IngestRequest(
        podcast="The Longevity Show with Dr. Hillary Lin",
        episode_title="The Modern Lipid Playbook Part 2",
        pub_date="2026-04-17",
        transcript_path=transcript_src,
        host="Hillary Lin",
        proposed_topics=["ApoB", "Lp(a)"],
        tags=["lipids", "longevity"],
        series="Modern Lipid Playbook",
        series_part=2,
    )
    result = ingest_episode(request, paths=paths, client=fake_anthropic)

    # JSONL upsert
    eid = "the-longevity-show-with-dr-hillary-lin__2026-04-17__the-modern-lipid-playbook-part-2"
    assert result.episode.id == eid
    index = load_index(paths.jsonl_path)
    assert eid in index
    stored = index[eid]
    assert stored.summary.qc_status == "passed"
    assert stored.transcript.source == "official"  # transcript-path branch
    assert stored.series == "Modern Lipid Playbook"

    # Pending vocab — both ApoB and Lp(a) were unknown, so both should be flagged
    assert set(result.pending_topics) == {"ApoB", "Lp(a)"}
    assert set(result.pending_speakers) == {"Hillary Lin"}
    topics_vocab = load_vocab(paths.topics_path)
    assert "ApoB" in topics_vocab.canonical
    assert topics_vocab.canonical["ApoB"].get("pending") is True

    # Summary + QC files on disk
    summary_md = result.summary_path.read_text(encoding="utf-8")
    assert "## TL;DR" in summary_md
    qc_md = result.qc_path.read_text(encoding="utf-8")
    assert "Verdict" in qc_md
    assert "passed" in qc_md

    # Indexes regenerated
    assert (paths.index_dir / "by-date.md").is_file()
    assert eid in (paths.index_dir / "by-date.md").read_text(encoding="utf-8")
    assert "Modern Lipid Playbook" in (paths.index_dir / "by-podcast.md").read_text(
        encoding="utf-8",
    )


def test_ingest_episode_failed_qc_preserves_existing_summary(
    tmp_path: Path,
    fake_anthropic: FakeAnthropic,
) -> None:
    """On QC failure, an existing summary must not be overwritten."""
    paths = _seed_library(tmp_path / "library")

    # First ingest: passes QC — establishes a "good" summary on disk
    fake_anthropic.enqueue_stream(SAMPLE_SUMMARY_MD)
    fake_anthropic.enqueue_create(json.dumps({"verdict": "passed", "issues": []}))
    transcript_src = tmp_path / "raw.txt"
    transcript_src.write_text(SAMPLE_TRANSCRIPT, encoding="utf-8")
    base_request = IngestRequest(
        podcast="Show",
        episode_title="Hello World",
        pub_date="2026-04-17",
        transcript_path=transcript_src,
    )
    first = ingest_episode(base_request, paths=paths, client=fake_anthropic)
    good_summary = first.summary_path.read_text(encoding="utf-8")

    # Second ingest: fails both attempts. Summariser is called twice (initial + retry);
    # QC is called twice (initial + retry).
    bad_summary = "# Hello World\n\n## TL;DR\nHallucinated nonsense.\n"
    fail_payload = json.dumps(
        {
            "verdict": "failed",
            "issues": [
                {
                    "category": "faithfulness",
                    "severity": "high",
                    "description": "Hallucinated a study.",
                },
            ],
        },
    )
    fake_anthropic.enqueue_stream(bad_summary)
    fake_anthropic.enqueue_create(fail_payload)
    fake_anthropic.enqueue_stream(bad_summary)
    fake_anthropic.enqueue_create(fail_payload)

    second = ingest_episode(base_request, paths=paths, client=fake_anthropic)
    assert second.qc_result.report.verdict == "failed"

    # Original summary must still be intact
    assert second.summary_path.read_text(encoding="utf-8") == good_summary
    # And the failed retry should sit under .failed.md
    failed_path = second.summary_path.with_suffix(".failed.md")
    assert failed_path.is_file()
    assert "Hallucinated nonsense" in failed_path.read_text(encoding="utf-8")


def _enqueue_failed_attempt(fake_anthropic: FakeAnthropic, summary_md: str) -> None:
    """Programme one fully-failed ingest: summarise + QC, then the retry pair."""
    fail_payload = json.dumps(
        {
            "verdict": "failed",
            "issues": [
                {
                    "category": "faithfulness",
                    "severity": "high",
                    "description": "Hallucinated a study.",
                },
            ],
        },
    )
    fake_anthropic.enqueue_stream(summary_md)
    fake_anthropic.enqueue_create(fail_payload)
    fake_anthropic.enqueue_stream(summary_md)
    fake_anthropic.enqueue_create(fail_payload)


def test_ingest_overwrites_prior_failed_summary(
    tmp_path: Path,
    fake_anthropic: FakeAnthropic,
) -> None:
    """A summary that itself failed QC is replaced, not "preserved" as if good."""
    paths = _seed_library(tmp_path / "library")
    transcript_src = tmp_path / "raw.txt"
    transcript_src.write_text(SAMPLE_TRANSCRIPT, encoding="utf-8")
    request = IngestRequest(
        podcast="Show",
        episode_title="Hello World",
        pub_date="2026-04-17",
        transcript_path=transcript_src,
    )

    _enqueue_failed_attempt(fake_anthropic, "# v1\n\nFirst failed attempt.\n")
    first = ingest_episode(request, paths=paths, client=fake_anthropic)
    assert first.qc_result.report.verdict == "failed"
    assert "First failed" in first.summary_path.read_text(encoding="utf-8")

    _enqueue_failed_attempt(fake_anthropic, "# v2\n\nSecond failed attempt.\n")
    second = ingest_episode(request, paths=paths, client=fake_anthropic)

    # The file on disk was a failed artifact — the fresh attempt replaces it
    # and nothing is squirrelled away as a bogus "preserved good summary".
    assert "Second failed" in second.summary_path.read_text(encoding="utf-8")
    assert not second.summary_path.with_suffix(".failed.md").exists()


def test_ingest_versions_repeated_failed_retries(
    tmp_path: Path,
    fake_anthropic: FakeAnthropic,
) -> None:
    """Repeated failures never clobber a good summary; sidecars are versioned."""
    paths = _seed_library(tmp_path / "library")
    transcript_src = tmp_path / "raw.txt"
    transcript_src.write_text(SAMPLE_TRANSCRIPT, encoding="utf-8")
    request = IngestRequest(
        podcast="Show",
        episode_title="Hello World",
        pub_date="2026-04-17",
        transcript_path=transcript_src,
    )

    fake_anthropic.enqueue_stream(SAMPLE_SUMMARY_MD)
    fake_anthropic.enqueue_create(json.dumps({"verdict": "passed", "issues": []}))
    good = ingest_episode(request, paths=paths, client=fake_anthropic)
    good_md = good.summary_path.read_text(encoding="utf-8")

    _enqueue_failed_attempt(fake_anthropic, "# bad1\n\nRetry one.\n")
    ingest_episode(request, paths=paths, client=fake_anthropic)
    _enqueue_failed_attempt(fake_anthropic, "# bad2\n\nRetry two.\n")
    second = ingest_episode(request, paths=paths, client=fake_anthropic)

    # The good summary survives both failed re-ingests…
    assert second.summary_path.read_text(encoding="utf-8") == good_md
    # …the JSONL row still describes the preserved file, not the failed attempt…
    index = load_index(paths.jsonl_path)
    assert index[second.episode.id].summary.qc_status == "passed"
    # …and each failed attempt got its own versioned sidecar.
    failed_1 = second.summary_path.with_suffix(".failed.md")
    failed_2 = second.summary_path.with_suffix(".failed.2.md")
    assert "bad1" in failed_1.read_text(encoding="utf-8")
    assert "bad2" in failed_2.read_text(encoding="utf-8")


def test_ingest_failure_before_upsert_does_not_persist_vocab(
    tmp_path: Path,
    fake_anthropic: FakeAnthropic,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed spine commit must not leave orphaned pending-vocab entries.

    If vocab were saved first, the re-ingest after a transient failure would
    resolve the names as already-canonical and silently drop their pending
    flags from the episode row.
    """
    paths = _seed_library(tmp_path / "library")
    fake_anthropic.enqueue_stream(SAMPLE_SUMMARY_MD)
    fake_anthropic.enqueue_create(json.dumps({"verdict": "passed", "issues": []}))
    transcript_src = tmp_path / "raw.txt"
    transcript_src.write_text(SAMPLE_TRANSCRIPT, encoding="utf-8")

    def _boom(path: Path, episode: object) -> bool:
        raise RuntimeError("simulated spine failure")

    monkeypatch.setattr("podcast_transcript.library.ingest.upsert", _boom)

    request = IngestRequest(
        podcast="Show",
        episode_title="Hello",
        pub_date="2026-04-17",
        transcript_path=transcript_src,
        host="Hillary Lin",
        proposed_topics=["ApoB"],
    )
    with pytest.raises(RuntimeError, match="simulated spine failure"):
        ingest_episode(request, paths=paths, client=fake_anthropic)

    # Vocab files untouched — the eventual successful re-ingest must still
    # see these names as unknown and flag them pending.
    assert load_vocab(paths.topics_path).canonical == {}
    assert load_vocab(paths.speakers_path).canonical == {}


def test_rebuild_indexes_script_idempotent_on_seeded_library(
    tmp_path: Path,
    fake_anthropic: FakeAnthropic,
) -> None:
    """Smoke test the rebuild_indexes path on a tiny seeded library."""
    paths = _seed_library(tmp_path / "library")
    fake_anthropic.enqueue_stream(SAMPLE_SUMMARY_MD)
    fake_anthropic.enqueue_create(json.dumps({"verdict": "passed", "issues": []}))
    transcript_src = tmp_path / "raw.txt"
    transcript_src.write_text(SAMPLE_TRANSCRIPT, encoding="utf-8")
    ingest_episode(
        IngestRequest(
            podcast="Show",
            episode_title="Hello",
            pub_date="2026-04-17",
            transcript_path=transcript_src,
        ),
        paths=paths,
        client=fake_anthropic,
    )

    # Re-run rebuild_all directly; should produce byte-identical bodies apart from timestamps.
    a = rebuild_all(index_dir=paths.index_dir, jsonl_path=paths.jsonl_path)
    b = rebuild_all(index_dir=paths.index_dir, jsonl_path=paths.jsonl_path)
    for name in a:
        body_a = a[name].read_text(encoding="utf-8")
        body_b = b[name].read_text(encoding="utf-8")

        def _strip(body: str) -> str:
            return "\n".join(line for line in body.splitlines() if "Generated by" not in line)

        assert _strip(body_a) == _strip(body_b), f"{name} not idempotent"
