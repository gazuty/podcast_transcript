"""Tests for :mod:`podcast_transcript.clean`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from podcast_transcript.clean import (
    apply_corrections,
    clean_transcript,
    collapse_repeated_lines,
    load_corrections,
    load_default_corrections,
    reflow_paragraphs,
    strip_outro_artifacts,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# collapse_repeated_lines
# ---------------------------------------------------------------------------


def test_collapse_collapses_long_run() -> None:
    lines = [
        "intro line",
        "So it's a major carrier in your body.",
        "So it's a major carrier in your body.",
        "So it's a major carrier in your body.",
        "So it's a major carrier in your body.",
        "outro line",
    ]
    result, collapsed = collapse_repeated_lines(lines)
    assert result == [
        "intro line",
        "So it's a major carrier in your body.",
        "outro line",
    ]
    assert collapsed == 3


def test_collapse_keeps_short_runs() -> None:
    # A speaker repeating themselves twice is left alone (default min_run=3).
    lines = ["yes", "yes", "no"]
    result, collapsed = collapse_repeated_lines(lines)
    assert result == ["yes", "yes", "no"]
    assert collapsed == 0


def test_collapse_handles_near_matches() -> None:
    # Slight variations should still collapse with the default 0.85 threshold.
    lines = [
        "It's a major carrier in your body.",
        "It's a major carrier in your body",  # missing period
        "It's a major carrier in your body.",
        "It is a major carrier in your body.",
    ]
    result, collapsed = collapse_repeated_lines(lines)
    assert len(result) == 1
    assert collapsed == 3


def test_collapse_rejects_invalid_min_run() -> None:
    with pytest.raises(ValueError, match="min_run"):
        collapse_repeated_lines(["a"], min_run=1)


def test_collapse_empty_input() -> None:
    result, collapsed = collapse_repeated_lines([])
    assert result == []
    assert collapsed == 0


# ---------------------------------------------------------------------------
# strip_outro_artifacts
# ---------------------------------------------------------------------------


def test_strip_drops_trailing_non_latin() -> None:
    lines = [
        "Thank you for listening.",
        "With a StepperB który Raděgoje",
        "A dedicated, ли",
    ]
    result, stripped = strip_outro_artifacts(lines)
    assert result == ["Thank you for listening."]
    assert stripped == 2


def test_strip_preserves_pure_english_tail() -> None:
    lines = ["Goodbye.", "See you next time."]
    result, stripped = strip_outro_artifacts(lines)
    assert result == lines
    assert stripped == 0


def test_strip_only_touches_tail() -> None:
    # Non-Latin text in the middle (e.g. a quoted name) must not be removed.
    lines = ["He said 你好.", "Then continued in English.", "End."]
    result, stripped = strip_outro_artifacts(lines)
    assert result == lines
    assert stripped == 0


def test_strip_drops_trailing_blank_lines() -> None:
    lines = ["body", "", "   ", ""]
    result, stripped = strip_outro_artifacts(lines)
    assert result == ["body"]
    assert stripped == 3


# ---------------------------------------------------------------------------
# apply_corrections
# ---------------------------------------------------------------------------


def test_apply_corrections_word_bounded() -> None:
    text = "We use Tehima's D, not Tajimaian D, in pop gen."
    cleaned, breakdown = apply_corrections(
        text,
        {"Tehima's D": "Tajima's D"},
    )
    assert cleaned == "We use Tajima's D, not Tajimaian D, in pop gen."
    assert breakdown == {"Tehima's D": 1}


def test_apply_corrections_counts_all_hits() -> None:
    text = "Tehima Tehima Tehima"
    cleaned, breakdown = apply_corrections(text, {"Tehima": "Tajima"})
    assert cleaned == "Tajima Tajima Tajima"
    assert breakdown == {"Tehima": 3}


def test_apply_corrections_skips_unmatched_keys() -> None:
    cleaned, breakdown = apply_corrections("hello world", {"foo": "bar"})
    assert cleaned == "hello world"
    assert breakdown == {}


# ---------------------------------------------------------------------------
# reflow_paragraphs
# ---------------------------------------------------------------------------


def test_reflow_groups_sentences() -> None:
    text = "\n".join(
        [
            "First sentence.",
            "Second sentence.",
            "Third sentence.",
            "Fourth sentence.",
            "Fifth sentence.",
            "Sixth sentence.",
            "Seventh sentence.",
        ]
    )
    out = reflow_paragraphs(text, sentences_per_paragraph=3)
    paragraphs = out.split("\n\n")
    assert len(paragraphs) == 3
    assert paragraphs[0] == "First sentence. Second sentence. Third sentence."
    assert paragraphs[2] == "Seventh sentence."


def test_reflow_handles_empty_input() -> None:
    assert reflow_paragraphs("") == ""
    assert reflow_paragraphs("   \n\n  ") == ""


def test_reflow_rejects_invalid_grouping() -> None:
    with pytest.raises(ValueError, match="sentences_per_paragraph"):
        reflow_paragraphs("a.", sentences_per_paragraph=0)


# ---------------------------------------------------------------------------
# load_corrections
# ---------------------------------------------------------------------------


def test_load_corrections_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "corrections.toml"
    path.write_text(
        '[corrections]\n"foo" = "bar"\n"baz qux" = "BAZ"\n',
        encoding="utf-8",
    )
    assert load_corrections(path) == {"foo": "bar", "baz qux": "BAZ"}


def test_load_corrections_rejects_non_table(tmp_path: Path) -> None:
    path = tmp_path / "broken.toml"
    path.write_text("corrections = 42\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a table"):
        load_corrections(path)


def test_load_default_corrections_has_known_entries() -> None:
    defaults = load_default_corrections()
    assert defaults["Tehima's D"] == "Tajima's D"
    assert defaults["right Fisher"] == "Wright-Fisher"


# ---------------------------------------------------------------------------
# clean_transcript
# ---------------------------------------------------------------------------


def test_clean_transcript_runs_full_pipeline() -> None:
    text = "\n".join(
        [
            "We use Tehima's D in this paper.",
            "So it's a major carrier in your body.",
            "So it's a major carrier in your body.",
            "So it's a major carrier in your body.",
            "Thanks for listening.",
            "Polski tekst który nie powinien być",
        ]
    )
    cleaned, stats = clean_transcript(text)
    assert "Tajima's D" in cleaned
    assert "Polski" not in cleaned
    assert cleaned.count("So it's a major carrier in your body.") == 1
    assert stats.lines_in == 6
    assert stats.loops_collapsed == 2
    assert stats.outro_lines_stripped == 1
    assert stats.corrections_applied >= 1
    assert stats.reflowed is False


def test_clean_transcript_with_explicit_empty_corrections() -> None:
    cleaned, stats = clean_transcript("Tehima's D rules.", corrections={})
    assert cleaned == "Tehima's D rules."
    assert stats.corrections_applied == 0


def test_clean_transcript_reflow_flag() -> None:
    text = "One sentence.\nTwo sentence.\nThree sentence."
    cleaned, stats = clean_transcript(
        text,
        corrections={},
        reflow=True,
        sentences_per_paragraph=2,
    )
    assert "\n\n" in cleaned
    assert stats.reflowed is True
