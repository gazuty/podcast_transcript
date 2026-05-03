"""Rule-based cleanup for Whisper transcripts.

Whisper makes a small set of recurring mistakes on long-form podcast audio:

1. **Looping hallucinations.** When a stretch of audio is hard to hear (music
   bed, low SNR, mumbled name), Whisper sometimes emits the same sentence two
   to five times in a row.
2. **Outro garbage.** Over closing music or applause Whisper hallucinates text
   in unrelated languages/scripts (Cyrillic, CJK, Polish, etc.) when the rest
   of the transcript is English.
3. **Domain-specific mistranscriptions.** Technical names ("Tajima's D",
   "Wright-Fisher", author names) come back phonetically wrong in
   reproducible ways.
4. **Per-segment line wrapping.** Whisper's ``txt`` writer emits one line per
   timestamp segment, which is fine for grep but harder to read as prose.

This module ships deterministic, dependency-free fixes for each. None of
them call out to a model — every transformation is something a senior
engineer can read and reason about.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "DEFAULT_LOOP_MIN_RUN",
    "DEFAULT_LOOP_THRESHOLD",
    "DEFAULT_REFLOW_SENTENCES",
    "CleanStats",
    "apply_corrections",
    "clean_transcript",
    "collapse_repeated_lines",
    "load_corrections",
    "load_default_corrections",
    "reflow_paragraphs",
    "strip_outro_artifacts",
]

DEFAULT_LOOP_THRESHOLD = 0.85
DEFAULT_LOOP_MIN_RUN = 3
DEFAULT_REFLOW_SENTENCES = 5


@dataclass
class CleanStats:
    """Summary of what :func:`clean_transcript` did to a transcript."""

    lines_in: int = 0
    lines_out: int = 0
    loops_collapsed: int = 0
    outro_lines_stripped: int = 0
    corrections_applied: int = 0
    reflowed: bool = False
    corrections_breakdown: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loop collapser
# ---------------------------------------------------------------------------


def _normalize(line: str) -> str:
    """Lowercase + strip + collapse internal whitespace for similarity."""
    return re.sub(r"\s+", " ", line.strip().lower())


def _similar(a: str, b: str, threshold: float) -> bool:
    a_norm, b_norm = _normalize(a), _normalize(b)
    if not a_norm or not b_norm:
        return False
    if a_norm == b_norm:
        return True
    return SequenceMatcher(None, a_norm, b_norm).ratio() >= threshold


def collapse_repeated_lines(
    lines: list[str],
    *,
    threshold: float = DEFAULT_LOOP_THRESHOLD,
    min_run: int = DEFAULT_LOOP_MIN_RUN,
) -> tuple[list[str], int]:
    """Collapse runs of near-identical adjacent lines down to one.

    A run of *min_run* or more adjacent lines that are pairwise similar (each
    line's :class:`difflib.SequenceMatcher` ratio against the run-leader is
    ``>= threshold``) is replaced by a single occurrence. Shorter runs are
    left alone, since two identical sentences are often correct (a speaker
    actually repeating themselves).

    Returns the new line list and the number of duplicate lines that were
    removed.
    """
    if min_run < 2:
        raise ValueError(f"min_run must be >= 2, got {min_run}")

    result: list[str] = []
    collapsed = 0
    i = 0
    n = len(lines)
    while i < n:
        j = i + 1
        while j < n and _similar(lines[i], lines[j], threshold):
            j += 1
        run_len = j - i
        if run_len >= min_run:
            result.append(lines[i])
            collapsed += run_len - 1
        else:
            result.extend(lines[i:j])
        i = j
    return result, collapsed


# ---------------------------------------------------------------------------
# Outro artifact stripper
# ---------------------------------------------------------------------------


_SENTENCE_TERMINATORS: tuple[str, ...] = (".", "!", "?", '"', "'")
_GOOD_LINE_MIN_LEN = 30


def _is_well_formed_english_line(line: str) -> bool:
    """A line that *looks like* a real English transcript segment.

    A line counts as well-formed iff it is pure-ASCII alphabetic content AND
    either ends in sentence-final punctuation/quote OR is at least
    :data:`_GOOD_LINE_MIN_LEN` characters long. Whisper's outro hallucinations
    are typically short, unpunctuated, or contain non-Latin script, so they
    fail this check.
    """
    stripped = line.strip()
    if not stripped:
        return False
    if any(c.isalpha() and ord(c) > 127 for c in stripped):
        return False
    if stripped.endswith(_SENTENCE_TERMINATORS):
        return True
    return len(stripped) >= _GOOD_LINE_MIN_LEN


def strip_outro_artifacts(lines: list[str]) -> tuple[list[str], int]:
    """Trim trailing Whisper outro hallucinations.

    Strategy: scan forward to find the *last* well-formed English line, then
    drop everything after it. This handles the common pattern where the
    actual outro hallucinations (non-Latin script, short fragments,
    single-word filler) are mixed together at the tail — a stricter
    "walk from the end" heuristic gives up too early when an ASCII fragment
    like ``"you"`` sits between the real content and the script-mismatch
    junk.

    If no well-formed line is found anywhere in the input (e.g. an extremely
    short transcript), the input is returned unchanged.
    """
    last_good = -1
    for i, line in enumerate(lines):
        if _is_well_formed_english_line(line):
            last_good = i
    if last_good >= 0:
        end = last_good + 1
    else:
        # No well-formed line found anywhere; conservatively only strip
        # trailing blank lines so we never wipe out a short transcript.
        end = len(lines)
        while end > 0 and not lines[end - 1].strip():
            end -= 1
    stripped = len(lines) - end
    return list(lines[:end]), stripped


# ---------------------------------------------------------------------------
# Corrections
# ---------------------------------------------------------------------------


def apply_corrections(
    text: str,
    corrections: Mapping[str, str],
) -> tuple[str, dict[str, int]]:
    """Apply word-bounded substitutions from *corrections* to *text*.

    Each ``wrong → right`` pair is compiled as a case-sensitive regex with
    word boundaries on each side, so ``"Tehima"`` won't match inside another
    word. Returns the rewritten text and a per-pattern hit count (only
    patterns that matched at least once are included).
    """
    breakdown: dict[str, int] = {}
    for wrong, right in corrections.items():
        pattern = re.compile(rf"\b{re.escape(wrong)}\b")
        text, count = pattern.subn(right, text)
        if count:
            breakdown[wrong] = count
    return text, breakdown


def load_corrections(path: Path | str) -> dict[str, str]:
    """Load a ``[corrections]`` table from a TOML file.

    The file format is::

        [corrections]
        "Tehima's D" = "Tajima's D"
        "right Fisher" = "Wright-Fisher"
    """
    path = Path(path)
    with path.open("rb") as f:
        data = tomllib.load(f)
    table = data.get("corrections", {})
    if not isinstance(table, dict):
        raise ValueError(f"{path}: [corrections] must be a table, got {type(table).__name__}")
    return {str(k): str(v) for k, v in table.items()}


def load_default_corrections() -> dict[str, str]:
    """Load the corrections dictionary that ships with the package."""
    ref = resources.files("podcast_transcript.data").joinpath("corrections.toml")
    with resources.as_file(ref) as concrete_path:
        return load_corrections(concrete_path)


# ---------------------------------------------------------------------------
# Paragraph reflow
# ---------------------------------------------------------------------------


# A naive sentence boundary: end-of-sentence punctuation, then whitespace,
# then a capital letter or opening quote. Good enough for podcast prose.
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'])')


def reflow_paragraphs(
    text: str,
    *,
    sentences_per_paragraph: int = DEFAULT_REFLOW_SENTENCES,
) -> str:
    """Collapse Whisper's per-segment lines into prose paragraphs.

    Whisper's ``.txt`` output puts every audio segment on its own line. For
    reading prose, that's noisy — this function joins all the lines, splits
    on sentence boundaries, and re-groups into paragraphs of
    *sentences_per_paragraph* sentences.
    """
    if sentences_per_paragraph < 1:
        raise ValueError(
            f"sentences_per_paragraph must be >= 1, got {sentences_per_paragraph}",
        )
    flat = " ".join(line.strip() for line in text.splitlines() if line.strip())
    if not flat:
        return ""
    sentences = _SENTENCE_SPLIT.split(flat)
    paragraphs = [
        " ".join(sentences[i : i + sentences_per_paragraph])
        for i in range(0, len(sentences), sentences_per_paragraph)
    ]
    return "\n\n".join(paragraphs)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def clean_transcript(
    text: str,
    *,
    corrections: Mapping[str, str] | None = None,
    loop_threshold: float = DEFAULT_LOOP_THRESHOLD,
    loop_min_run: int = DEFAULT_LOOP_MIN_RUN,
    reflow: bool = False,
    sentences_per_paragraph: int = DEFAULT_REFLOW_SENTENCES,
) -> tuple[str, CleanStats]:
    """Run the full cleanup pipeline on *text* and return the cleaned text + stats.

    Order is fixed and deliberate:

    1. Collapse looping hallucinations.
    2. Strip outro artifacts from the tail.
    3. Apply text corrections.
    4. (Optional) Reflow into paragraphs.

    If *corrections* is None, the bundled default dictionary is used. Pass an
    empty dict to skip corrections entirely.
    """
    if corrections is None:
        corrections = load_default_corrections()

    stats = CleanStats()
    lines = text.splitlines()
    stats.lines_in = len(lines)

    lines, stats.loops_collapsed = collapse_repeated_lines(
        lines,
        threshold=loop_threshold,
        min_run=loop_min_run,
    )
    lines, stats.outro_lines_stripped = strip_outro_artifacts(lines)
    stats.lines_out = len(lines)

    cleaned = "\n".join(lines)
    cleaned, breakdown = apply_corrections(cleaned, corrections)
    stats.corrections_breakdown = breakdown
    stats.corrections_applied = sum(breakdown.values())

    if reflow:
        cleaned = reflow_paragraphs(cleaned, sentences_per_paragraph=sentences_per_paragraph)
        stats.reflowed = True

    return cleaned, stats
