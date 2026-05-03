"""Tests for :mod:`podcast_transcript.clean`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from podcast_transcript.clean import (
    CorrectionsFile,
    apply_corrections,
    apply_uncertain_corrections,
    clean_transcript,
    collapse_repeated_lines,
    collapse_repeated_phrases,
    collapse_windowed_near_duplicates,
    detect_preview_cut,
    load_corrections,
    load_corrections_file,
    load_default_corrections,
    merge_corrections_files,
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


# ---------------------------------------------------------------------------
# apply_uncertain_corrections
# ---------------------------------------------------------------------------


def test_uncertain_annotates_with_suggestion() -> None:
    text = "Stephen Ghazal said something."
    out, breakdown = apply_uncertain_corrections(
        text,
        {"Stephen Ghazal": "Stephen Gazal"},
    )
    assert out == "[?: Stephen Ghazal → Stephen Gazal] said something."
    assert breakdown == {"Stephen Ghazal": 1}


def test_uncertain_flag_only_when_suggestion_blank() -> None:
    out, breakdown = apply_uncertain_corrections(
        "ABO is detected as benorephora.",
        {"benorephora": ""},
    )
    assert out == "ABO is detected as [?: benorephora]."
    assert breakdown == {"benorephora": 1}


def test_uncertain_word_bounded() -> None:
    # 'Mit' alone shouldn't match inside 'Mitnick'.
    out, breakdown = apply_uncertain_corrections("Mitnick", {"Mit": ""})
    assert out == "Mitnick"
    assert breakdown == {}


def test_uncertain_handles_backslashes_in_data() -> None:
    # Pathological replacement string — make sure it's not interpreted as a
    # regex back-reference.
    out, _ = apply_uncertain_corrections(
        "name foo end",
        {"foo": r"\1bar"},
    )
    assert out == r"name [?: foo → \1bar] end"


# ---------------------------------------------------------------------------
# detect_preview_cut
# ---------------------------------------------------------------------------


def test_detect_preview_cut_flags_substack_outro() -> None:
    lines = ["Real content."] * 100 + [
        "To hear the rest of the monologue, please go to razib.substack.com and subscribe."
    ]
    reason = detect_preview_cut(lines)
    assert reason is not None
    assert "rest" in reason


def test_detect_preview_cut_returns_none_for_full_episode() -> None:
    lines = ["Real content."] * 50 + ["See you next time."]
    assert detect_preview_cut(lines) is None


def test_detect_preview_cut_ignores_phrase_in_body() -> None:
    # "subscribe to hear" appearing early in the show isn't a preview cut.
    lines = ["intro: subscribe to hear when new episodes drop."] + ["Real body content."] * 200
    assert detect_preview_cut(lines) is None


def test_detect_preview_cut_rejects_invalid_tail_fraction() -> None:
    with pytest.raises(ValueError, match="tail_fraction"):
        detect_preview_cut(["x"], tail_fraction=0)


# ---------------------------------------------------------------------------
# load_corrections_file + merge
# ---------------------------------------------------------------------------


def test_load_corrections_file_parses_both_tables(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    path.write_text(
        '[corrections]\n"a" = "b"\n[uncertain]\n"x" = "y"\n"q" = ""\n',
        encoding="utf-8",
    )
    parsed = load_corrections_file(path)
    assert parsed.corrections == {"a": "b"}
    assert parsed.uncertain == {"x": "y", "q": ""}


def test_merge_corrections_files_later_wins() -> None:
    a = CorrectionsFile(corrections={"x": "1", "y": "1"}, uncertain={"u": "1"})
    b = CorrectionsFile(corrections={"y": "2"}, uncertain={"u": "2"})
    merged = merge_corrections_files([a, b])
    assert merged.corrections == {"x": "1", "y": "2"}
    assert merged.uncertain == {"u": "2"}


# ---------------------------------------------------------------------------
# clean_transcript with uncertain + preview-cut
# ---------------------------------------------------------------------------


def test_clean_transcript_threads_uncertain_and_preview() -> None:
    text = "\n".join(
        [
            "We use Tehima's D in this paper.",
            "Stephen Ghazal joined the team.",
            "Real body content " * 30,
        ]
        + ["Real body content"] * 50
        + ["To hear the rest of the episode, subscribe."]
    )
    cleaned, stats = clean_transcript(
        text,
        uncertain={"Stephen Ghazal": "Stephen Gazal"},
    )
    assert "[?: Stephen Ghazal → Stephen Gazal]" in cleaned
    assert stats.uncertain_applied == 1
    assert stats.preview_cut_reason is not None


# ---------------------------------------------------------------------------
# collapse_repeated_phrases
# ---------------------------------------------------------------------------


def test_phrase_collapse_in_single_line() -> None:
    text = "verification rewards specific, you know, specific, you know, specific, you know, end."
    out, n = collapse_repeated_phrases(text)
    # Three "specific, you know," runs collapse to one.
    assert out.count("specific, you know,") == 1
    assert "verification rewards" in out
    assert out.endswith("end.")
    assert n == 1


def test_phrase_collapse_across_newlines() -> None:
    text = (
        "are giant reinforcement learning environments,\n"
        "so they are given a very specific, you know, specific, you\n"
        "know, specific, you know, specific, you know, verification rewards.\n"
    )
    out, n = collapse_repeated_phrases(text)
    assert out.count("specific, you know,") == 1
    assert "verification rewards." in out
    assert n == 1


def test_phrase_collapse_leaves_single_word_loops_alone() -> None:
    # Default min_words=2: single-word emphasis should be preserved.
    text = "no no no no I won't"
    out, n = collapse_repeated_phrases(text)
    assert out == text
    assert n == 0


def test_phrase_collapse_catches_single_word_when_configured() -> None:
    text = "specific specific specific specific end"
    out, n = collapse_repeated_phrases(text, min_words=1)
    assert "specific specific specific" not in out
    assert "specific" in out
    assert "end" in out
    assert n == 1


def test_phrase_collapse_rejects_invalid_args() -> None:
    with pytest.raises(ValueError, match="min_words"):
        collapse_repeated_phrases("x", min_words=0)
    with pytest.raises(ValueError, match="max_words"):
        collapse_repeated_phrases("x", min_words=4, max_words=2)


# ---------------------------------------------------------------------------
# collapse_windowed_near_duplicates
# ---------------------------------------------------------------------------


def test_windowed_dedup_drops_near_duplicate_within_window() -> None:
    long_a = "I think a lot of people probably here are excited about what this looks like."
    long_b = "I think a lot of people probably are excited about what this looks like."
    lines = [long_a, "Yeah.", "Yeah.", long_b, "moving on."]
    result, dropped = collapse_windowed_near_duplicates(lines)
    assert long_a in result
    assert long_b not in result
    # Short lines preserved.
    assert result.count("Yeah.") == 2
    assert dropped == 1


def test_windowed_dedup_preserves_short_conversational_lines() -> None:
    # Several "Yeah."s in a row, none of them dropped because below
    # min_line_length.
    lines = ["Yeah."] * 6
    result, dropped = collapse_windowed_near_duplicates(lines)
    assert result == lines
    assert dropped == 0


def test_windowed_dedup_respects_window_size() -> None:
    long_a = "This is a fully formed sentence that exceeds the min length threshold."
    # Use deliberately diverse filler so the filler lines are not themselves
    # near-duplicates of one another.
    filler = [
        "We discussed the migration plan for the production database last Tuesday.",
        "Marketing wants the launch announcement to ship before the conference keynote.",
        "Andrew suggested we revisit the pricing model after the next board meeting.",
        "Engineering will absorb the on-call rotation while the new hires are ramping up.",
        "Procurement flagged a delay in the data center power upgrade timeline.",
        "Legal asked us to update the privacy policy before customers in the EU sign up.",
        "The customer success team shipped a new onboarding flow last quarter and saw lift.",
        "Ops needs additional staging headroom before Black Friday traffic ramps in earnest.",
        "Recruiting closed two senior backend roles and is interviewing for staff frontend.",
        "Finance wants us to forecast cloud spend by region for the FY27 budget cycle.",
    ]
    lines = [long_a, *filler, long_a]
    # window=3: the second long_a is far past its match, so kept.
    result, dropped = collapse_windowed_near_duplicates(lines, window=3)
    assert dropped == 0
    assert result.count(long_a) == 2
    # window=large: it's caught and dropped.
    result2, dropped2 = collapse_windowed_near_duplicates(lines, window=20)
    assert dropped2 == 1
    assert result2.count(long_a) == 1


def test_windowed_dedup_rejects_invalid_window() -> None:
    with pytest.raises(ValueError, match="window"):
        collapse_windowed_near_duplicates(["x"], window=0)


# ---------------------------------------------------------------------------
# clean_transcript wires the new passes
# ---------------------------------------------------------------------------


def test_clean_transcript_runs_phrase_and_windowed_passes() -> None:
    text = (
        "verification rewards specific, you know, specific, you know, specific, you know, end.\n"
        "I think a lot of people probably here are excited about what this looks like.\n"
        "Yeah.\n"
        "I think a lot of people probably are excited about what this looks like.\n"
        "Conclusion sentence that wraps things up nicely so well-formed."
    )
    cleaned, stats = clean_transcript(text, corrections={})
    assert stats.phrase_loops_collapsed == 1
    assert stats.windowed_duplicates_collapsed == 1
    assert cleaned.count("specific, you know,") == 1
    assert cleaned.count("I think a lot of people probably") == 1
