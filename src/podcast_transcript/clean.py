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
    from collections.abc import Iterable, Mapping

__all__ = [
    "DEFAULT_LOOP_MIN_RUN",
    "DEFAULT_LOOP_THRESHOLD",
    "DEFAULT_PHRASE_LOOP_MAX_WORDS",
    "DEFAULT_PHRASE_LOOP_MIN_WORDS",
    "DEFAULT_PREVIEW_TAIL_FRACTION",
    "DEFAULT_REFLOW_SENTENCES",
    "DEFAULT_WINDOW_DUP_MIN_LEN",
    "DEFAULT_WINDOW_DUP_SIZE",
    "CleanStats",
    "CorrectionsFile",
    "apply_corrections",
    "apply_uncertain_corrections",
    "clean_transcript",
    "collapse_repeated_lines",
    "collapse_repeated_phrases",
    "collapse_windowed_near_duplicates",
    "detect_preview_cut",
    "load_corrections",
    "load_corrections_file",
    "load_default_corrections",
    "merge_corrections_files",
    "reflow_paragraphs",
    "strip_outro_artifacts",
]

DEFAULT_LOOP_THRESHOLD = 0.85
DEFAULT_LOOP_MIN_RUN = 3
DEFAULT_REFLOW_SENTENCES = 5
DEFAULT_PREVIEW_TAIL_FRACTION = 0.05
DEFAULT_PHRASE_LOOP_MIN_WORDS = 2
DEFAULT_PHRASE_LOOP_MAX_WORDS = 8
DEFAULT_WINDOW_DUP_SIZE = 5
DEFAULT_WINDOW_DUP_MIN_LEN = 30


@dataclass
class CorrectionsFile:
    """Parsed contents of a corrections TOML file.

    A file may contribute confident replacements (the ``[corrections]`` table)
    and/or uncertain candidates (the ``[uncertain]`` table). Uncertain entries
    are *annotated* in the transcript rather than silently replaced — see
    :func:`apply_uncertain_corrections`.
    """

    corrections: dict[str, str] = field(default_factory=dict)
    uncertain: dict[str, str] = field(default_factory=dict)


@dataclass
class CleanStats:
    """Summary of what :func:`clean_transcript` did to a transcript."""

    lines_in: int = 0
    lines_out: int = 0
    loops_collapsed: int = 0
    phrase_loops_collapsed: int = 0
    windowed_duplicates_collapsed: int = 0
    outro_lines_stripped: int = 0
    corrections_applied: int = 0
    uncertain_applied: int = 0
    reflowed: bool = False
    corrections_breakdown: dict[str, int] = field(default_factory=dict)
    uncertain_breakdown: dict[str, int] = field(default_factory=dict)
    preview_cut_reason: str | None = None


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
# Intra-line / cross-line phrase loop collapser
# ---------------------------------------------------------------------------


_PHRASE_LOOP_MIN_REPETITIONS = 3
_TOKEN_PATTERN = re.compile(r"(\S+)(\s*)")


def collapse_repeated_phrases(
    text: str,
    *,
    min_words: int = DEFAULT_PHRASE_LOOP_MIN_WORDS,
    max_words: int = DEFAULT_PHRASE_LOOP_MAX_WORDS,
) -> tuple[str, int]:
    """Collapse a short phrase that repeats 3+ times in a row down to one.

    Catches Whisper hallucinations of the form
    ``"specific, you know, specific, you know, specific, you know"`` —
    cases the line-level :func:`collapse_repeated_lines` misses because
    the loop sits *inside* a single segment, or *spans* a few segments
    glued together by Whisper's wrapping.

    The implementation is token-based (rather than a regex with a
    backreference) so that variable whitespace between repetitions —
    e.g. a space in one repetition and a newline in the next — does not
    defeat the match. Trailing whitespace is preserved on the surviving
    occurrence so the surrounding prose is unchanged.

    Args:
        text: The transcript text.
        min_words: Minimum phrase length in whitespace-separated tokens.
            Defaults to 2 — single-word repetition like ``"yes yes yes"``
            is left alone (often legitimate emphasis).
        max_words: Maximum phrase length to consider.

    Returns:
        ``(cleaned_text, n_collapses)``.
    """
    if min_words < 1:
        raise ValueError(f"min_words must be >= 1, got {min_words}")
    if max_words < min_words:
        raise ValueError(
            f"max_words ({max_words}) must be >= min_words ({min_words})",
        )

    tokens: list[tuple[str, str]] = _TOKEN_PATTERN.findall(text)
    if not tokens:
        return text, 0

    leading_ws_match = re.match(r"\s*", text)
    leading = leading_ws_match.group(0) if leading_ws_match else ""

    out: list[tuple[str, str]] = []
    n = len(tokens)
    i = 0
    collapses = 0
    while i < n:
        collapsed_here = False
        # Prefer the SHORTEST repeating unit so we catch
        # "specific, you know," rather than a longer accidental match.
        for phrase_len in range(min_words, max_words + 1):
            run_end = i + phrase_len
            if run_end > n:
                break
            phrase_words = [tokens[i + k][0] for k in range(phrase_len)]
            reps = 1
            while True:
                start = i + reps * phrase_len
                if start + phrase_len > n:
                    break
                if [tokens[start + k][0] for k in range(phrase_len)] != phrase_words:
                    break
                reps += 1
            if reps >= _PHRASE_LOOP_MIN_REPETITIONS:
                # Keep one occurrence (preserve its original whitespace).
                out.extend(tokens[i:run_end])
                i += reps * phrase_len
                collapses += 1
                collapsed_here = True
                break
        if not collapsed_here:
            out.append(tokens[i])
            i += 1

    cleaned = leading + "".join(word + ws for word, ws in out)
    return cleaned, collapses


# ---------------------------------------------------------------------------
# Windowed near-duplicate line collapser
# ---------------------------------------------------------------------------


def collapse_windowed_near_duplicates(
    lines: list[str],
    *,
    window: int = DEFAULT_WINDOW_DUP_SIZE,
    threshold: float = DEFAULT_LOOP_THRESHOLD,
    min_line_length: int = DEFAULT_WINDOW_DUP_MIN_LEN,
) -> tuple[list[str], int]:
    """Drop a line if a near-duplicate exists within the previous *window* lines.

    Catches paragraph-level Whisper duplicates that are separated by a few
    intervening lines (e.g. interspersed ``"Yeah."`` lines), which the
    adjacent-line :func:`collapse_repeated_lines` cannot see.

    Short conversational lines (``len < min_line_length`` after stripping)
    are passed through untouched so legitimate ``"Yes."`` / ``"Yeah."``
    repetitions in a back-and-forth aren't deduplicated.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    result: list[str] = []
    dropped = 0
    for line in lines:
        stripped = line.strip()
        if len(stripped) < min_line_length:
            result.append(line)
            continue
        recent = result[-window:]
        if any(_similar(line, prev, threshold) for prev in recent if prev.strip()):
            dropped += 1
            continue
        result.append(line)
    return result, dropped


# ---------------------------------------------------------------------------
# Outro artifact stripper
# ---------------------------------------------------------------------------


_SENTENCE_TERMINATORS: tuple[str, ...] = (".", "!", "?", '"', "'")
_GOOD_LINE_MIN_LEN = 30


def _is_well_formed_english_line(line: str) -> bool:
    """A line that *looks like* a real English transcript segment.

    A line counts as well-formed iff every letter is ASCII or Latin-1
    Supplement (the accents English prose actually borrows: José, café,
    naïve) AND it either ends in sentence-final punctuation/quote OR is at
    least :data:`_GOOD_LINE_MIN_LEN` characters long. Whisper's outro
    hallucinations are typically short, unpunctuated, or carry non-Latin
    or Latin-Extended letters (Cyrillic, CJK, the ě/ł/ř of Polish and
    Czech junk), so they fail this check.
    """
    stripped = line.strip()
    if not stripped:
        return False
    if any(c.isalpha() and ord(c) > 0xFF for c in stripped):
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


def apply_uncertain_corrections(
    text: str,
    uncertain: Mapping[str, str],
) -> tuple[str, dict[str, int]]:
    """Annotate uncertain candidate corrections inline.

    For each ``wrong → suggestion`` pair, every word-bounded match of *wrong*
    is replaced with ``[?: wrong → suggestion]`` so a human reviewer can
    grep, accept, or override later. If *suggestion* is empty, the term is
    flagged with ``[?: wrong]`` (no proposal yet).

    Returns the annotated text and per-pattern hit count.
    """
    breakdown: dict[str, int] = {}
    for wrong, suggestion in uncertain.items():
        annotation = f"[?: {wrong} → {suggestion}]" if suggestion else f"[?: {wrong}]"
        pattern = re.compile(rf"\b{re.escape(wrong)}\b")
        # A function replacer (rather than a string) so backslashes in user
        # data aren't interpreted as regex back-references.

        def _replace(_m: re.Match[str], ann: str = annotation) -> str:
            return ann

        text, count = pattern.subn(_replace, text)
        if count:
            breakdown[wrong] = count
    return text, breakdown


def _load_table(data: Mapping[str, object], path: Path, key: str) -> dict[str, str]:
    table = data.get(key, {})
    if not isinstance(table, dict):
        raise ValueError(f"{path}: [{key}] must be a table, got {type(table).__name__}")
    return {str(k): str(v) for k, v in table.items()}


def load_corrections_file(path: Path | str) -> CorrectionsFile:
    """Load both ``[corrections]`` and ``[uncertain]`` tables from a TOML file."""
    path = Path(path)
    with path.open("rb") as f:
        data = tomllib.load(f)
    return CorrectionsFile(
        corrections=_load_table(data, path, "corrections"),
        uncertain=_load_table(data, path, "uncertain"),
    )


def load_corrections(path: Path | str) -> dict[str, str]:
    """Load just the ``[corrections]`` table from a TOML file (back-compat helper)."""
    return load_corrections_file(path).corrections


def load_default_corrections() -> dict[str, str]:
    """Load the bundled ``[corrections]`` dictionary."""
    return load_default_corrections_file().corrections


def load_default_corrections_file() -> CorrectionsFile:
    """Load the bundled corrections TOML (both tables) that ships with the package."""
    ref = resources.files("podcast_transcript.data").joinpath("corrections.toml")
    with resources.as_file(ref) as concrete_path:
        return load_corrections_file(concrete_path)


def merge_corrections_files(files: Iterable[CorrectionsFile]) -> CorrectionsFile:
    """Merge a sequence of corrections files in order; later entries win on conflicts."""
    merged = CorrectionsFile()
    for f in files:
        merged.corrections.update(f.corrections)
        merged.uncertain.update(f.uncertain)
    return merged


# ---------------------------------------------------------------------------
# Preview-cut detector
# ---------------------------------------------------------------------------


# Phrases that strongly suggest the source MP3 was a paywall preview, not the
# full episode. Matched case-insensitively against the tail of the transcript
# (default last 5%). Keep this list tight — false positives turn into noise.
_PREVIEW_CUT_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"to hear the rest", "'to hear the rest'"),
    (r"hear the rest of (the|this) (episode|monologue|conversation)", "'hear the rest of ...'"),
    (r"subscribe to hear", "'subscribe to hear'"),
    (r"for the (full|rest of the) episode", "'for the full/rest of the episode'"),
    (r"become a (paid )?subscriber", "'become a (paid) subscriber'"),
    (r"head (over )?to .{0,40}\.substack\.com", "'head over to ...substack.com'"),
    (r"members? only", "'members only'"),
    (r"behind the paywall", "'behind the paywall'"),
)


def detect_preview_cut(
    lines: list[str],
    *,
    tail_fraction: float = DEFAULT_PREVIEW_TAIL_FRACTION,
) -> str | None:
    """Return a human-readable reason if the transcript tail looks paywalled.

    Scans the last *tail_fraction* of non-empty lines for known preview-cut
    phrases; returns ``None`` if nothing matches. The returned string is the
    matched phrase label, suitable for a log warning.
    """
    if not 0 < tail_fraction <= 1:
        raise ValueError(f"tail_fraction must be in (0, 1], got {tail_fraction}")
    non_empty = [line for line in lines if line.strip()]
    if not non_empty:
        return None
    tail_len = max(1, int(len(non_empty) * tail_fraction))
    tail_text = "\n".join(non_empty[-tail_len:])
    for pattern, label in _PREVIEW_CUT_PATTERNS:
        if re.search(pattern, tail_text, re.IGNORECASE):
            return label
    return None


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
    uncertain: Mapping[str, str] | None = None,
    loop_threshold: float = DEFAULT_LOOP_THRESHOLD,
    loop_min_run: int = DEFAULT_LOOP_MIN_RUN,
    reflow: bool = False,
    sentences_per_paragraph: int = DEFAULT_REFLOW_SENTENCES,
    detect_preview: bool = True,
    preview_tail_fraction: float = DEFAULT_PREVIEW_TAIL_FRACTION,
) -> tuple[str, CleanStats]:
    """Run the full cleanup pipeline on *text* and return the cleaned text + stats.

    Order is fixed and deliberate:

    1. Collapse runs of near-identical adjacent lines (whole-line loops).
    2. Collapse intra-line / cross-line repeated phrases (sub-line loops).
    3. Drop windowed near-duplicate lines (paragraph-level dedup).
    4. Strip outro artifacts from the tail.
    5. Apply confident text corrections.
    6. Annotate uncertain candidate corrections inline.
    7. (Optional) Reflow into paragraphs.

    Whole-line collapse goes first because it is the strictest, cheapest
    pass; running the phrase-loop collapser before it would steal those
    matches and leave the line collapser nothing to do.

    Preview-cut detection runs against the post-collapse, pre-reflow line list
    so paywall phrases at the genuine end of the audio are still findable.

    If *corrections* is None, the bundled default ``[corrections]`` dictionary
    is used. Pass an empty dict to skip confident corrections entirely.
    *uncertain* defaults to empty (no inline annotations) for backward
    compatibility with callers that only want replacements.
    """
    if corrections is None:
        corrections = load_default_corrections()
    if uncertain is None:
        uncertain = {}

    stats = CleanStats()
    lines = text.splitlines()
    stats.lines_in = len(lines)

    # Whole-line dupes first (cheapest, strictest).
    lines, stats.loops_collapsed = collapse_repeated_lines(
        lines,
        threshold=loop_threshold,
        min_run=loop_min_run,
    )
    # Then sub-line / cross-line phrase loops on the rejoined text.
    rejoined = "\n".join(lines)
    rejoined, stats.phrase_loops_collapsed = collapse_repeated_phrases(rejoined)
    lines = rejoined.splitlines()
    # Then paragraph-level near-duplicates within a small window.
    lines, stats.windowed_duplicates_collapsed = collapse_windowed_near_duplicates(
        lines,
        threshold=loop_threshold,
    )
    lines, stats.outro_lines_stripped = strip_outro_artifacts(lines)
    stats.lines_out = len(lines)

    if detect_preview:
        stats.preview_cut_reason = detect_preview_cut(
            lines,
            tail_fraction=preview_tail_fraction,
        )

    cleaned = "\n".join(lines)
    cleaned, breakdown = apply_corrections(cleaned, corrections)
    stats.corrections_breakdown = breakdown
    stats.corrections_applied = sum(breakdown.values())

    cleaned, uncertain_breakdown = apply_uncertain_corrections(cleaned, uncertain)
    stats.uncertain_breakdown = uncertain_breakdown
    stats.uncertain_applied = sum(uncertain_breakdown.values())

    if reflow:
        cleaned = reflow_paragraphs(cleaned, sentences_per_paragraph=sentences_per_paragraph)
        stats.reflowed = True

    return cleaned, stats
