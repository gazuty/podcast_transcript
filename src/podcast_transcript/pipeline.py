"""End-to-end ``run`` pipeline: source → (discover transcript | transcribe) → clean.

Stitches the existing modules together. Kept separate from :mod:`cli` so the
orchestration is testable without parsing argv, and so importing :mod:`cli`
stays cheap (no torch via :mod:`transcribe`, no XML parser via :mod:`feed`)
for the lighter subcommands.

Output layout for ``--slug foo``::

    <audio_dir>/foo.mp3              # downloaded audio (only when we transcribed)
    <transcripts_dir>/foo.txt        # raw text (Whisper or fetched SRT/VTT)
    <transcripts_dir>/foo_clean.txt  # cleaned transcript (this module's job)

When a publisher-hosted SRT/VTT transcript is discovered (via the Podcasting
2.0 ``<podcast:transcript>`` tag for ``--rss``, or via HTML scrape for
``--page``), we skip the audio download and Whisper steps entirely — the
fetched caption text is normalised to one line per cue and fed through the
same :func:`apply_strip_patterns` + :func:`clean_transcript` post-processing
as a Whisper transcript would be, so the output shape is identical.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .clean import (
    DEFAULT_REFLOW_SENTENCES,
    CleanStats,
    CorrectionsFile,
    clean_transcript,
)
from .download import DownloadError, download_podcast
from .feed import FeedItem, TranscriptRef, load_feed, select_item
from .page_scrape import PageInfo, discover_episode_links
from .transcribe import (
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    transcribe_audio,
)
from .transcript_fetch import (
    TranscriptFetchError,
    fetch_transcript_text,
    preferred_transcript,
    subrip_to_text,
    vtt_to_text,
)

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "PipelineError",
    "PipelineResult",
    "TranscriptSource",
    "apply_strip_patterns",
    "run_pipeline",
]


TranscriptSource = Literal["whisper", "rss", "page"]

logger = logging.getLogger("podcast_transcript")


class PipelineError(Exception):
    """Raised for orchestration-level failures (e.g. bad RSS selection args)."""


@dataclass
class PipelineResult:
    """Where the pipeline put each artifact, plus cleanup stats.

    *audio_path* is ``None`` when a publisher transcript was found and we
    therefore skipped the audio download. *transcript_source* records which
    branch produced the raw text — ``"whisper"`` means we transcribed
    locally, ``"rss"`` / ``"page"`` mean we fetched a publisher transcript.
    """

    raw_transcript_path: Path
    clean_transcript_path: Path
    stats: CleanStats
    transcript_source: TranscriptSource
    audio_path: Path | None = None
    feed_item: FeedItem | None = None


@dataclass
class _ResolvedSource:
    """Outcome of source resolution before any download/transcribe runs.

    Exactly one of *audio_url* (Whisper path) or *transcripts* (publisher
    transcript path candidates) is populated — *transcripts* may also be
    populated alongside *audio_url* when the caller will choose later
    based on the ``--no-discover-transcript`` flag.
    """

    audio_url: str | None
    transcripts: tuple[TranscriptRef, ...]
    feed_item: FeedItem | None
    source_label: TranscriptSource  # what to record if we end up fetching


def _resolve_source(
    *,
    url: str | None,
    rss_url: str | None,
    page_url: str | None,
    episode_regex: str | None,
    episode_index: int | None,
    timeout: float,
) -> _ResolvedSource:
    sources = [s for s in (url, rss_url, page_url) if s is not None]
    if len(sources) > 1:
        raise PipelineError("pass exactly one of --url, --rss, or --page")
    if not sources:
        raise PipelineError("must pass one of --url, --rss, or --page")

    if url is not None:
        if episode_regex is not None or episode_index is not None:
            raise PipelineError("--episode-regex/--episode-index only apply with --rss")
        return _ResolvedSource(
            audio_url=url,
            transcripts=(),
            feed_item=None,
            source_label="whisper",
        )

    if page_url is not None:
        if episode_regex is not None or episode_index is not None:
            raise PipelineError("--episode-regex/--episode-index only apply with --rss")
        info = discover_episode_links(page_url, timeout=timeout)
        return _resolved_source_from_page(info, page_url)

    assert rss_url is not None
    if episode_regex is None and episode_index is None:
        raise PipelineError("with --rss, also pass --episode-regex or --episode-index")
    if episode_regex is not None and episode_index is not None:
        raise PipelineError("pass --episode-regex or --episode-index, not both")

    items = load_feed(rss_url, timeout=timeout)
    item = select_item(items, regex=episode_regex, index=episode_index)
    return _ResolvedSource(
        audio_url=item.enclosure_url,
        transcripts=item.transcripts,
        feed_item=item,
        source_label="rss",
    )


def _resolved_source_from_page(info: PageInfo, page_url: str) -> _ResolvedSource:
    if info.audio_url is None and not info.transcripts:
        raise PipelineError(
            f"no audio or transcript links found on {page_url!r}; "
            "pass --url directly with the MP3 location",
        )
    return _ResolvedSource(
        audio_url=info.audio_url,
        transcripts=info.transcripts,
        feed_item=None,
        source_label="page",
    )


def apply_strip_patterns(
    text: str,
    *,
    strip_before: list[str],
    strip_after: list[str],
) -> tuple[str, int, int]:
    """Apply pre-roll and outro strip regexes to a per-line transcript.

    For each pattern in *strip_before*, drop everything up to and including
    the first matching line. For each pattern in *strip_after*, drop
    everything from the last matching line onward — but only if the match
    falls in the *tail half* of the (then-current) transcript, to avoid
    chopping the body of the conversation when an outro phrase happens to
    occur naturally mid-show.

    Returns the rewritten text and the number of lines stripped from the
    head and tail respectively. Patterns are matched case-insensitively.
    """
    lines = text.splitlines()
    head_stripped = 0
    for pattern in strip_before:
        compiled = re.compile(pattern, re.IGNORECASE)
        for idx, line in enumerate(lines):
            if compiled.search(line):
                head_stripped += idx + 1
                lines = lines[idx + 1 :]
                break
    tail_stripped = 0
    for pattern in strip_after:
        compiled = re.compile(pattern, re.IGNORECASE)
        last_match = -1
        for idx, line in enumerate(lines):
            if compiled.search(line):
                last_match = idx
        if last_match >= 0 and last_match >= len(lines) // 2:
            tail_stripped += len(lines) - last_match
            lines = lines[:last_match]
    return "\n".join(lines), head_stripped, tail_stripped


def run_pipeline(
    *,
    url: str | None,
    rss_url: str | None,
    page_url: str | None = None,
    episode_regex: str | None,
    episode_index: int | None,
    slug: str,
    audio_dir: Path,
    transcripts_dir: Path,
    model: str = DEFAULT_MODEL,
    language: str = DEFAULT_LANGUAGE,
    corrections: CorrectionsFile,
    strip_before: list[str],
    strip_after: list[str],
    reflow: bool = False,
    sentences_per_paragraph: int = DEFAULT_REFLOW_SENTENCES,
    timeout: float,
    discover_transcript: bool = True,
) -> PipelineResult:
    """Run the full source → (fetch | transcribe) → clean pipeline.

    Each step is logged at INFO. The raw text is written to
    ``<transcripts_dir>/<slug>.txt`` (either Whisper's writer output or the
    converted SRT/VTT). The cleaned output is written to
    ``<transcripts_dir>/<slug>_clean.txt``.

    If *discover_transcript* is False, the publisher-transcript branch is
    skipped entirely — useful when a publisher's transcript is known to be
    worse than what local Whisper produces.
    """
    resolved = _resolve_source(
        url=url,
        rss_url=rss_url,
        page_url=page_url,
        episode_regex=episode_regex,
        episode_index=episode_index,
        timeout=timeout,
    )
    if resolved.feed_item is not None:
        logger.info("selected episode: %s", resolved.feed_item.title)

    raw_path = transcripts_dir / f"{slug}.txt"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    audio_path: Path | None = None
    used_transcript = False
    if discover_transcript:
        chosen = preferred_transcript(resolved.transcripts)
        if chosen is not None:
            transcript_ref, kind = chosen
            logger.info(
                "found publisher %s transcript: %s",
                kind.upper(),
                transcript_ref.url,
            )
            try:
                raw_text = _fetch_publisher_transcript(transcript_ref.url, kind, timeout=timeout)
            except (DownloadError, ValueError) as exc:
                # A declared transcript that 404s, is oversized, turns out
                # not to be captions, or carries a non-http URL (ValueError
                # from the scheme check) shouldn't kill the run when we can
                # still transcribe the audio ourselves.
                logger.warning("publisher transcript unusable (%s); falling back to Whisper", exc)
            else:
                raw_path.write_text(raw_text, encoding="utf-8")
                used_transcript = True

    if not used_transcript:
        if resolved.audio_url is None:
            raise PipelineError(
                "no audio URL available and no usable publisher transcript "
                "was found; cannot transcribe",
            )
        audio_path = _download_audio(
            resolved.audio_url,
            audio_dir=audio_dir,
            slug=slug,
            timeout=timeout,
        )
        _transcribe_to_disk(
            audio_path,
            transcripts_dir=transcripts_dir,
            model=model,
            language=language,
        )
        if not raw_path.is_file():
            raise PipelineError(
                f"whisper did not produce expected transcript at {raw_path}",
            )

    raw_text = raw_path.read_text(encoding="utf-8")
    stripped_text, head_stripped, tail_stripped = apply_strip_patterns(
        raw_text,
        strip_before=strip_before,
        strip_after=strip_after,
    )
    if head_stripped or tail_stripped:
        logger.info(
            "ad-strip removed %d head line(s) and %d tail line(s)",
            head_stripped,
            tail_stripped,
        )

    cleaned, stats = clean_transcript(
        stripped_text,
        corrections=corrections.corrections,
        uncertain=corrections.uncertain,
        reflow=reflow,
        sentences_per_paragraph=sentences_per_paragraph,
    )

    clean_path = transcripts_dir / f"{slug}_clean.txt"
    clean_path.write_text(cleaned, encoding="utf-8")

    logger.info(
        "wrote %s (lines %d → %d, loops collapsed: %d, outro lines stripped: %d, "
        "corrections: %d, uncertain: %d%s)",
        clean_path,
        stats.lines_in,
        stats.lines_out,
        stats.loops_collapsed,
        stats.outro_lines_stripped,
        stats.corrections_applied,
        stats.uncertain_applied,
        ", reflowed" if stats.reflowed else "",
    )
    if stats.preview_cut_reason is not None:
        logger.warning(
            "source MP3 may be a preview cut — tail contains %s",
            stats.preview_cut_reason,
        )

    transcript_source: TranscriptSource = resolved.source_label if used_transcript else "whisper"
    return PipelineResult(
        audio_path=audio_path,
        raw_transcript_path=raw_path,
        clean_transcript_path=clean_path,
        stats=stats,
        transcript_source=transcript_source,
        feed_item=resolved.feed_item,
    )


def _fetch_publisher_transcript(url: str, kind: str, *, timeout: float) -> str:
    """Download a publisher SRT/VTT URL and convert it to one-line-per-cue text.

    The body must contain at least one ``-->`` cue arrow before conversion —
    the content-type gate in :func:`fetch_transcript_text` can't catch a
    server that mislabels an HTML or plain-text page, but no real SRT/VTT
    file lacks timestamps.
    """
    body, _content_type = fetch_transcript_text(url, timeout=timeout)
    if "-->" not in body:
        raise TranscriptFetchError(
            f"no cue timestamps in {url!r}; body does not look like SRT/VTT",
        )
    text = subrip_to_text(body) if kind == "srt" else vtt_to_text(body)
    if not text.strip():
        raise TranscriptFetchError(f"transcript at {url!r} converted to empty text")
    return text


def _download_audio(audio_url: str, *, audio_dir: Path, slug: str, timeout: float) -> Path:
    audio_path = audio_dir / f"{slug}.mp3"
    logger.info("downloading %s → %s", audio_url, audio_path)
    download_podcast(audio_url, audio_path, timeout=timeout)
    return audio_path


def _transcribe_to_disk(
    audio_path: Path,
    *,
    transcripts_dir: Path,
    model: str,
    language: str,
) -> None:
    logger.info("transcribing %s with model=%s language=%s", audio_path, model, language)
    transcribe_audio(
        audio_path,
        model_name=model,
        language=language,
        output_dir=transcripts_dir,
    )
