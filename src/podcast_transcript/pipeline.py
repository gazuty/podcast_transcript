"""End-to-end ``run`` pipeline: source → download → transcribe → clean.

Stitches the existing modules together. Kept separate from :mod:`cli` so the
orchestration is testable without parsing argv, and so importing :mod:`cli`
stays cheap (no torch via :mod:`transcribe`, no XML parser via :mod:`feed`)
for the lighter subcommands.

Output layout for ``--slug foo``::

    <audio_dir>/foo.mp3              # downloaded audio
    <transcripts_dir>/foo.txt        # raw whisper output (also .srt/.vtt/.tsv/.json)
    <transcripts_dir>/foo_clean.txt  # cleaned transcript (this module's job)

The raw whisper ``.txt`` is left untouched so users can diff against the
cleaned version.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .clean import (
    DEFAULT_REFLOW_SENTENCES,
    CleanStats,
    CorrectionsFile,
    clean_transcript,
)
from .download import download_podcast
from .feed import FeedItem, load_feed, select_item
from .transcribe import (
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    transcribe_audio,
)

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "PipelineError",
    "PipelineResult",
    "apply_strip_patterns",
    "run_pipeline",
]

logger = logging.getLogger("podcast_transcript")


class PipelineError(Exception):
    """Raised for orchestration-level failures (e.g. bad RSS selection args)."""


@dataclass
class PipelineResult:
    """Where the pipeline put each artifact, plus cleanup stats."""

    audio_path: Path
    raw_transcript_path: Path
    clean_transcript_path: Path
    stats: CleanStats
    feed_item: FeedItem | None = None


def _resolve_source(
    *,
    url: str | None,
    rss_url: str | None,
    episode_regex: str | None,
    episode_index: int | None,
    timeout: float,
) -> tuple[str, FeedItem | None]:
    if url is not None and rss_url is not None:
        raise PipelineError("pass --url or --rss, not both")
    if url is not None:
        if episode_regex is not None or episode_index is not None:
            raise PipelineError("--episode-regex/--episode-index only apply with --rss")
        return url, None
    if rss_url is None:
        raise PipelineError("must pass either --url or --rss")
    if episode_regex is None and episode_index is None:
        raise PipelineError("with --rss, also pass --episode-regex or --episode-index")
    if episode_regex is not None and episode_index is not None:
        raise PipelineError("pass --episode-regex or --episode-index, not both")

    items = load_feed(rss_url, timeout=timeout)
    item = select_item(items, regex=episode_regex, index=episode_index)
    return item.enclosure_url, item


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
) -> PipelineResult:
    """Run the full download → transcribe → clean pipeline.

    Each step is logged at INFO. The whisper ``.txt`` is preserved at
    ``<transcripts_dir>/<slug>.txt``; the cleaned output is written to
    ``<transcripts_dir>/<slug>_clean.txt``.
    """
    audio_url, feed_item = _resolve_source(
        url=url,
        rss_url=rss_url,
        episode_regex=episode_regex,
        episode_index=episode_index,
        timeout=timeout,
    )
    if feed_item is not None:
        logger.info("selected episode: %s", feed_item.title)

    audio_path = audio_dir / f"{slug}.mp3"
    logger.info("downloading %s → %s", audio_url, audio_path)
    download_podcast(audio_url, audio_path, timeout=timeout)

    logger.info("transcribing %s with model=%s language=%s", audio_path, model, language)
    transcribe_audio(
        audio_path,
        model_name=model,
        language=language,
        output_dir=transcripts_dir,
    )
    raw_path = transcripts_dir / f"{slug}.txt"
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

    return PipelineResult(
        audio_path=audio_path,
        raw_transcript_path=raw_path,
        clean_transcript_path=clean_path,
        stats=stats,
        feed_item=feed_item,
    )
