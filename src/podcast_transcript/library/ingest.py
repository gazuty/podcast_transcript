"""End-to-end ingest: source → transcript → summary → QC → JSONL → indexes.

The orchestrator wires together everything else in this package plus
``run_pipeline`` from the main package. It is deliberately a top-level
function rather than a class — one call, one episode, one row added to
``episodes.jsonl``.

Resolution order for the transcript:

1. If the caller passed ``--transcript`` (a path to an already-existing
   ``.txt``), use it as-is and skip the pipeline.
2. Otherwise call :func:`podcast_transcript.pipeline.run_pipeline` with
   the supplied source flags, copy the cleaned transcript into the
   library's ``transcripts/<slug>/`` directory, and record the source
   label (``rss`` / ``page`` / ``whisper``).

Resolution order for vocab:

- Speakers and topics are normalised through ``vocab/{speakers,topics}.json``.
- Unknown entries are auto-added with ``pending: true`` (see
  :meth:`Vocab.add_pending`) and the original (resolved) name is also
  recorded on ``Episode.pending_topics`` / ``pending_speakers`` so
  ``pending-vocab.md`` can surface it for review.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ..clean import CorrectionsFile
from ..pipeline import run_pipeline

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from ..pipeline import PipelineResult
from .episode import (
    Episode,
    SourceUrls,
    SummaryRef,
    TranscriptRef,
    compute_transcript_checksum,
    make_episode_id,
    slugify,
)
from .indexes import rebuild_all
from .qc import QCResult, format_qc_markdown, run_summary_with_qc
from .store import load_index, upsert
from .summarise import AnthropicClientLike, SummariseInput
from .vocab import Vocab, load_vocab, save_vocab

__all__ = [
    "IngestError",
    "IngestPaths",
    "IngestRequest",
    "IngestResult",
    "ingest_episode",
]

logger = logging.getLogger("podcast_transcript.library")


class IngestError(RuntimeError):
    """Raised for ingest-level orchestration failures."""


@dataclass
class IngestPaths:
    """Where on disk the library lives.

    Defaults to ``./podcast-library/`` relative to the current working
    directory; override for tests by passing a different root.
    """

    library_root: Path

    @property
    def transcripts_dir(self) -> Path:
        return self.library_root / "transcripts"

    @property
    def summaries_dir(self) -> Path:
        return self.library_root / "summaries"

    @property
    def audio_dir(self) -> Path:
        return self.library_root / "audio"

    @property
    def index_dir(self) -> Path:
        return self.library_root / "index"

    @property
    def jsonl_path(self) -> Path:
        return self.index_dir / "episodes.jsonl"

    @property
    def topics_path(self) -> Path:
        return self.index_dir / "vocab" / "topics.json"

    @property
    def speakers_path(self) -> Path:
        return self.index_dir / "vocab" / "speakers.json"


@dataclass
class IngestRequest:
    """All the knobs ingest needs for one episode.

    Exactly one of *transcript_path*, *url*, *rss_url*, or *page_url*
    should be provided. The first non-None wins, in that order.
    """

    podcast: str
    episode_title: str
    pub_date: str  # YYYY-MM-DD

    # Source selectors (mutually exclusive)
    transcript_path: Path | None = None
    url: str | None = None
    rss_url: str | None = None
    page_url: str | None = None
    episode_regex: str | None = None
    episode_index: int | None = None

    # Optional metadata
    host: str | None = None
    guests: list[str] = field(default_factory=list)
    proposed_topics: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    series: str | None = None
    series_part: int | None = None
    episode_number: int | None = None
    duration_seconds: int | None = None

    # Pipeline knobs (only used when transcript_path is None)
    corrections_pack: list[str] = field(default_factory=list)
    strip_before: list[str] = field(default_factory=list)
    strip_after: list[str] = field(default_factory=list)
    timeout: float = 60.0


@dataclass
class IngestResult:
    """What the orchestrator did and where it put things."""

    episode: Episode
    summary_path: Path
    qc_path: Path
    transcript_path: Path
    qc_result: QCResult
    indexes: dict[str, Path]
    pending_topics: list[str]
    pending_speakers: list[str]


def ingest_episode(
    request: IngestRequest,
    *,
    paths: IngestPaths,
    client: AnthropicClientLike,
) -> IngestResult:
    """Run one episode end-to-end and update the library."""
    podcast_slug = slugify(request.podcast)
    episode_id = make_episode_id(
        podcast_slug=podcast_slug,
        pub_date=request.pub_date,
        title=request.episode_title,
    )
    logger.info("ingesting %s", episode_id)

    transcript_dest = paths.transcripts_dir / podcast_slug / f"{episode_id}.txt"
    transcript_source_label, pipeline_result = _resolve_transcript(
        request,
        paths=paths,
        podcast_slug=podcast_slug,
        episode_id=episode_id,
        transcript_dest=transcript_dest,
    )

    transcript_text = transcript_dest.read_text(encoding="utf-8")
    checksum = compute_transcript_checksum(transcript_dest)

    # Summarise + QC
    summarise_input = SummariseInput(
        transcript=transcript_text,
        podcast=request.podcast,
        episode_title=request.episode_title,
        pub_date=request.pub_date,
        host=request.host,
        guests=tuple(request.guests),
        series=request.series,
        series_part=request.series_part,
        source_label=transcript_source_label,
    )
    qc_result = run_summary_with_qc(client, summarise_input, seed=episode_id)

    # Write summary + QC report
    summary_dest = paths.summaries_dir / podcast_slug / f"{episode_id}.md"
    summary_dest.parent.mkdir(parents=True, exist_ok=True)
    qc_markdown = format_qc_markdown(qc_result.report, episode_id=episode_id)
    preserved_summary: SummaryRef | None = None
    if qc_result.report.verdict == "failed":
        # Spec: don't silently overwrite a *good* summary. The file on disk
        # alone can't tell good from bad — a prior run that itself failed QC
        # also wrote to this path — so consult the stored record's qc_status.
        # On preserve, the prior SummaryRef is carried into the new row so
        # the record keeps describing what actually sits at the path; the
        # failed attempt lands in versioned ``.failed[.N].md`` sidecars.
        preserved_summary = _preservable_prior_summary(paths, episode_id)
        if summary_dest.is_file() and preserved_summary is not None:
            failed_path = _next_failed_path(summary_dest)
            failed_path.write_text(qc_result.summary_md, encoding="utf-8")
            qc_dest = failed_path.with_suffix(".qc.md")
            qc_dest.write_text(qc_markdown, encoding="utf-8")
            logger.warning(
                "QC failed; preserved existing summary at %s, wrote retry to %s",
                summary_dest,
                failed_path,
            )
        else:
            preserved_summary = None
            summary_dest.write_text(qc_result.summary_md, encoding="utf-8")
            qc_dest = summary_dest.with_suffix(".qc.md")
            qc_dest.write_text(qc_markdown, encoding="utf-8")
            logger.warning(
                "QC failed; wrote summary at %s anyway (no prior good summary)",
                summary_dest,
            )
    else:
        summary_dest.write_text(qc_result.summary_md, encoding="utf-8")
        qc_dest = summary_dest.with_suffix(".qc.md")
        qc_dest.write_text(qc_markdown, encoding="utf-8")

    # Vocab normalisation. Mutations stay in memory here; the files are
    # persisted only after the JSONL upsert below succeeds, so a failed
    # ingest can't leave vocab entries that no episode row references
    # (which would suppress the pending flags on the eventual re-ingest).
    speakers_vocab = load_vocab(paths.speakers_path)
    topics_vocab = load_vocab(paths.topics_path)
    resolved_speakers, pending_speakers = _normalise_through_vocab(
        [*(request.guests or []), *([request.host] if request.host else [])],
        speakers_vocab,
    )
    resolved_topics, pending_topics = _normalise_through_vocab(
        request.proposed_topics,
        topics_vocab,
    )
    for name in pending_speakers:
        speakers_vocab.add_pending(name)
    for name in pending_topics:
        topics_vocab.add_pending(name)

    now = datetime.now(UTC).isoformat(timespec="seconds")
    episode = Episode(
        id=episode_id,
        podcast=request.podcast,
        podcast_slug=podcast_slug,
        episode_title=request.episode_title,
        pub_date=request.pub_date,
        episode_number=request.episode_number,
        duration_seconds=request.duration_seconds,
        host=request.host,
        guests=request.guests,
        speakers=resolved_speakers,
        topics=resolved_topics,
        tags=list(request.tags),
        series=request.series,
        series_part=request.series_part,
        source_urls=_source_urls_from(request, pipeline_result),
        transcript=TranscriptRef(
            path=str(transcript_dest.relative_to(paths.library_root)),
            source=transcript_source_label,
            model=("large-v3" if transcript_source_label == "whisper" else None),
            has_timestamps=False,
        ),
        summary=preserved_summary
        if preserved_summary is not None
        else SummaryRef(
            path=str(summary_dest.relative_to(paths.library_root)),
            generated_at=now,
            model="claude-opus-4-7",
            qc_status=qc_result.report.verdict,
            qc_notes_path=str(qc_dest.relative_to(paths.library_root)),
        ),
        ingested_at=now,
        checksum=checksum,
        pending_topics=pending_topics,
        pending_speakers=pending_speakers,
    )
    upsert(paths.jsonl_path, episode)

    # The spine committed — now persist the vocab additions it references.
    if pending_speakers:
        save_vocab(paths.speakers_path, speakers_vocab)
    if pending_topics:
        save_vocab(paths.topics_path, topics_vocab)

    indexes = rebuild_all(index_dir=paths.index_dir, jsonl_path=paths.jsonl_path)
    return IngestResult(
        episode=episode,
        summary_path=summary_dest,
        qc_path=qc_dest,
        transcript_path=transcript_dest,
        qc_result=qc_result,
        indexes=indexes,
        pending_topics=pending_topics,
        pending_speakers=pending_speakers,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _preservable_prior_summary(paths: IngestPaths, episode_id: str) -> SummaryRef | None:
    """The stored summary record for *episode_id*, if it didn't fail QC.

    Only such a summary is worth preserving over a freshly failed one; if
    there is no prior row (or its qc_status is ``failed``), whatever file
    sits at the summary path is itself a failed artifact and may be
    overwritten. Returning the :class:`SummaryRef` (not a bool) lets the
    caller carry it into the new row, so a preserved summary keeps its
    original metadata and a *later* failed run still sees it as good.
    """
    prior = load_index(paths.jsonl_path).get(episode_id)
    if prior is not None and prior.summary.qc_status != "failed":
        return prior.summary
    return None


def _next_failed_path(summary_dest: Path) -> Path:
    """First free ``<id>.failed.md`` / ``<id>.failed.N.md`` next to *summary_dest*.

    Versioned so a second failed retry doesn't silently overwrite the
    diagnostics from the first.
    """
    candidate = summary_dest.with_suffix(".failed.md")
    n = 2
    while candidate.exists():
        candidate = summary_dest.with_suffix(f".failed.{n}.md")
        n += 1
    return candidate


def _resolve_transcript(
    request: IngestRequest,
    *,
    paths: IngestPaths,
    podcast_slug: str,
    episode_id: str,
    transcript_dest: Path,
) -> tuple[str, PipelineResult | None]:
    """Either copy in an existing transcript or run the pipeline.

    Returns ``(source_label, pipeline_result_or_None)``. *source_label*
    is one of ``"whisper"`` / ``"rss"`` / ``"page"`` / ``"official"``
    (the latter when the caller passed an existing file).
    """
    transcript_dest.parent.mkdir(parents=True, exist_ok=True)
    if request.transcript_path is not None:
        src = request.transcript_path
        if not src.is_file():
            raise IngestError(f"transcript not found: {src}")
        if src.resolve() != transcript_dest.resolve():
            shutil.copyfile(src, transcript_dest)
        return "official", None

    if request.url is None and request.rss_url is None and request.page_url is None:
        raise IngestError(
            "must pass exactly one of --transcript, --url, --rss, or --page",
        )

    result = run_pipeline(
        url=request.url,
        rss_url=request.rss_url,
        page_url=request.page_url,
        episode_regex=request.episode_regex,
        episode_index=request.episode_index,
        slug=episode_id,
        audio_dir=paths.audio_dir,
        transcripts_dir=paths.transcripts_dir / podcast_slug,
        corrections=CorrectionsFile(),  # corrections happen at summarise time
        strip_before=request.strip_before,
        strip_after=request.strip_after,
        timeout=request.timeout,
    )

    # run_pipeline writes <slug>_clean.txt next to <slug>.txt; we want the
    # *_clean.txt version as the canonical transcript for the library.
    clean_path = result.clean_transcript_path
    if clean_path.resolve() != transcript_dest.resolve():
        shutil.copyfile(clean_path, transcript_dest)

    return result.transcript_source, result


def _normalise_through_vocab(
    names: Sequence[str | None],
    vocab: Vocab,
) -> tuple[list[str], list[str]]:
    """Resolve each name; collect any pending entries for surfacing later."""
    resolved: list[str] = []
    pending: list[str] = []
    seen: set[str] = set()
    for raw in names:
        if not raw:
            continue
        canonical, is_pending = vocab.resolve(raw)
        if canonical not in seen:
            resolved.append(canonical)
            seen.add(canonical)
        if is_pending and canonical not in pending:
            pending.append(canonical)
    return resolved, pending


def _source_urls_from(
    request: IngestRequest,
    pipeline_result: PipelineResult | None,
) -> SourceUrls:
    # Pull the resolved enclosure URL off the pipeline result when we have
    # one — that way an RSS-driven ingest records the actual audio URL the
    # feed pointed to, not just the feed URL the user supplied.
    audio = request.url
    if audio is None and pipeline_result is not None and pipeline_result.feed_item is not None:
        audio = pipeline_result.feed_item.enclosure_url
    return SourceUrls(
        rss_item=request.rss_url,
        audio=audio,
        episode_page=request.page_url,
        youtube=None,
    )
