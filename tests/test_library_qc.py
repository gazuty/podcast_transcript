"""Tests for :mod:`podcast_transcript.library.summarise` and :mod:`.qc`.

The ``fake_anthropic`` fixture from :mod:`conftest` lets us drive both
the streamed summary call and the JSON-format QC call without going
near the real SDK or the network.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from podcast_transcript.library.qc import (
    QCResult,
    format_qc_markdown,
    qc_summary,
    run_summary_with_qc,
    sample_coverage_chunks,
)
from podcast_transcript.library.summarise import (
    SummariseInput,
    SummariserError,
    summarise_transcript,
)

if TYPE_CHECKING:
    from .conftest import FakeAnthropic


# ---------------------------------------------------------------------------
# Coverage sampling
# ---------------------------------------------------------------------------


def test_sample_coverage_chunks_is_deterministic() -> None:
    transcript = "\n".join(f"line {i}: " + "x" * 50 for i in range(200))
    a = sample_coverage_chunks(transcript, seed="ep1")
    b = sample_coverage_chunks(transcript, seed="ep1")
    assert a == b


def test_sample_coverage_chunks_varies_with_seed() -> None:
    transcript = "\n".join(f"line {i}: " + "x" * 50 for i in range(200))
    a = sample_coverage_chunks(transcript, seed="ep1")
    b = sample_coverage_chunks(transcript, seed="ep2")
    # Same length but different content with high probability
    assert len(a) == len(b) == 5
    assert a != b


def test_sample_coverage_chunks_empty_returns_empty() -> None:
    assert sample_coverage_chunks("", seed="x") == []


def test_sample_coverage_chunks_short_transcript() -> None:
    chunks = sample_coverage_chunks("only one short line", seed="x")
    assert chunks == ["only one short line"]


# ---------------------------------------------------------------------------
# summarise_transcript
# ---------------------------------------------------------------------------


SAMPLE_SUMMARY = """# Hello World

## TL;DR
A friendly test summary.

## Key points
- Razib mentioned ApoB.

## Key learnings / takeaways
- ApoB testing matters.

## Notable quotes
> "ApoB is the better marker." — Razib

## Numbers, studies, named entities
- ApoB

## Open questions / things to verify
- _None._

## Glossary (if technical)
- ApoB — apolipoprotein B.
"""


def test_summarise_transcript_returns_streamed_text(fake_anthropic: FakeAnthropic) -> None:
    fake_anthropic.enqueue_stream(SAMPLE_SUMMARY)
    inp = SummariseInput(
        transcript="Razib here talking about ApoB and lipid panels.",
        podcast="Show",
        episode_title="Hello World",
        pub_date="2026-04-17",
    )
    out = summarise_transcript(fake_anthropic, inp)
    assert out == SAMPLE_SUMMARY
    # Verify the transcript was cached as a system block
    call = fake_anthropic.stream_calls[0]
    system = call["system"]
    assert isinstance(system, list)
    transcript_block = system[1]
    assert transcript_block["cache_control"] == {"type": "ephemeral"}
    assert "Razib here talking about ApoB" in transcript_block["text"]
    # And metadata went into the user turn, not the system block
    user_msg = call["messages"][0]
    assert user_msg["role"] == "user"
    user_text = user_msg["content"][-1]["text"]
    assert "Episode metadata" in user_text
    assert "Hello World" in user_text


def test_summarise_transcript_rejects_empty(fake_anthropic: FakeAnthropic) -> None:
    inp = SummariseInput(
        transcript="   ",
        podcast="Show",
        episode_title="Hello",
        pub_date="2026-04-17",
    )
    with pytest.raises(SummariserError, match="empty transcript"):
        summarise_transcript(fake_anthropic, inp)


def test_summarise_transcript_qc_feedback_goes_in_user_turn(
    fake_anthropic: FakeAnthropic,
) -> None:
    fake_anthropic.enqueue_stream(SAMPLE_SUMMARY)
    inp = SummariseInput(
        transcript="Body of the transcript.",
        podcast="Show",
        episode_title="Hello",
        pub_date="2026-04-17",
        qc_feedback="- (HIGH faithfulness) Hallucinated a fact.",
    )
    summarise_transcript(fake_anthropic, inp)
    user_blocks = fake_anthropic.stream_calls[0]["messages"][0]["content"]
    # First user block is the QC feedback (so the cache stays warm), second is metadata
    assert "Hallucinated a fact" in user_blocks[0]["text"]
    assert "Episode metadata" in user_blocks[1]["text"]


def test_summarise_transcript_raises_on_empty_response(fake_anthropic: FakeAnthropic) -> None:
    fake_anthropic.enqueue_stream("   ")
    inp = SummariseInput(
        transcript="Body.",
        podcast="Show",
        episode_title="Hello",
        pub_date="2026-04-17",
    )
    with pytest.raises(SummariserError, match="no text content"):
        summarise_transcript(fake_anthropic, inp)


# ---------------------------------------------------------------------------
# qc_summary
# ---------------------------------------------------------------------------


def test_qc_summary_parses_passed_verdict(fake_anthropic: FakeAnthropic) -> None:
    fake_anthropic.enqueue_create(json.dumps({"verdict": "passed", "issues": []}))
    report = qc_summary(
        fake_anthropic,
        transcript="Razib talks about ApoB.",
        summary_md=SAMPLE_SUMMARY,
        seed="ep1",
    )
    assert report.verdict == "passed"
    assert report.issues == []
    # The QC call must use output_config.format with json_schema
    call = fake_anthropic.create_calls[0]
    assert call["output_config"]["format"]["type"] == "json_schema"


def test_qc_summary_parses_failed_verdict_with_issues(fake_anthropic: FakeAnthropic) -> None:
    payload = {
        "verdict": "failed",
        "issues": [
            {
                "category": "faithfulness",
                "severity": "high",
                "description": "Hallucinated a study.",
                "summary_excerpt": "claimed the JUPITER trial",
                "suggested_fix": "Remove the JUPITER reference.",
            },
        ],
    }
    fake_anthropic.enqueue_create(json.dumps(payload))
    report = qc_summary(
        fake_anthropic,
        transcript="Razib talks about ApoB.",
        summary_md=SAMPLE_SUMMARY,
        seed="ep1",
    )
    assert report.verdict == "failed"
    assert len(report.issues) == 1
    issue = report.issues[0]
    assert issue.category == "faithfulness"
    assert issue.severity == "high"
    assert issue.suggested_fix == "Remove the JUPITER reference."


def test_qc_summary_raises_on_non_json_response(fake_anthropic: FakeAnthropic) -> None:
    fake_anthropic.enqueue_create("not json at all")
    with pytest.raises(SummariserError, match="non-JSON"):
        qc_summary(
            fake_anthropic,
            transcript="x",
            summary_md=SAMPLE_SUMMARY,
            seed="ep1",
        )


# ---------------------------------------------------------------------------
# run_summary_with_qc — retry orchestration
# ---------------------------------------------------------------------------


def test_run_summary_with_qc_passes_first_try(fake_anthropic: FakeAnthropic) -> None:
    fake_anthropic.enqueue_stream(SAMPLE_SUMMARY)
    fake_anthropic.enqueue_create(json.dumps({"verdict": "passed", "issues": []}))
    result = run_summary_with_qc(
        fake_anthropic,
        SummariseInput(
            transcript="Razib talks about ApoB.",
            podcast="Show",
            episode_title="Hello",
            pub_date="2026-04-17",
        ),
        seed="ep1",
    )
    assert result.attempts == 1
    assert result.retried is False
    assert result.report.verdict == "passed"


def test_run_summary_with_qc_retries_once_on_flagged(fake_anthropic: FakeAnthropic) -> None:
    fake_anthropic.enqueue_stream(SAMPLE_SUMMARY)
    fake_anthropic.enqueue_create(
        json.dumps(
            {
                "verdict": "flagged",
                "issues": [
                    {
                        "category": "coverage",
                        "severity": "medium",
                        "description": "Missed the section on Lp(a).",
                    },
                ],
            },
        ),
    )
    # Retry produces a revised summary and a passing QC verdict
    fake_anthropic.enqueue_stream(SAMPLE_SUMMARY + "\n## Addendum\n- Lp(a).\n")
    fake_anthropic.enqueue_create(json.dumps({"verdict": "passed", "issues": []}))

    result = run_summary_with_qc(
        fake_anthropic,
        SummariseInput(
            transcript="Razib talks about ApoB and Lp(a).",
            podcast="Show",
            episode_title="Hello",
            pub_date="2026-04-17",
        ),
        seed="ep1",
    )
    assert result.attempts == 2
    assert result.retried is True
    assert result.report.verdict == "passed"
    assert "Addendum" in result.summary_md
    # Second summarise call must include the QC feedback
    retry_user_blocks = fake_anthropic.stream_calls[1]["messages"][0]["content"]
    assert "Missed the section on Lp(a)" in retry_user_blocks[0]["text"]


def test_run_summary_with_qc_preserves_failure_after_retry(
    fake_anthropic: FakeAnthropic,
) -> None:
    fake_anthropic.enqueue_stream(SAMPLE_SUMMARY)
    fake_anthropic.enqueue_create(
        json.dumps(
            {
                "verdict": "failed",
                "issues": [
                    {
                        "category": "faithfulness",
                        "severity": "high",
                        "description": "Hallucinated a fact.",
                    },
                ],
            },
        ),
    )
    fake_anthropic.enqueue_stream(SAMPLE_SUMMARY)
    fake_anthropic.enqueue_create(
        json.dumps(
            {
                "verdict": "failed",
                "issues": [
                    {
                        "category": "faithfulness",
                        "severity": "high",
                        "description": "Still hallucinating.",
                    },
                ],
            },
        ),
    )
    result = run_summary_with_qc(
        fake_anthropic,
        SummariseInput(
            transcript="Razib talks about ApoB.",
            podcast="Show",
            episode_title="Hello",
            pub_date="2026-04-17",
        ),
        seed="ep1",
    )
    assert result.attempts == 2
    assert result.report.verdict == "failed"


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def test_format_qc_markdown_includes_issues_and_chunks() -> None:
    from podcast_transcript.library.qc import QCIssue, QCReport

    report = QCReport(
        verdict="flagged",
        summary=SAMPLE_SUMMARY,
        issues=[
            QCIssue(
                category="coverage",
                severity="medium",
                description="Missed Lp(a).",
                summary_excerpt=None,
                suggested_fix="Add a bullet under Key points.",
            ),
        ],
        coverage_chunks=["chunk 1 contents", "chunk 2 contents"],
    )
    md = format_qc_markdown(report, episode_id="show__2026-04-17__hello")
    assert "show__2026-04-17__hello" in md
    assert "`flagged`" in md
    assert "coverage" in md
    assert "Missed Lp(a)" in md
    assert "Add a bullet" in md
    assert "Chunk 1" in md
    assert "chunk 1 contents" in md


def test_format_qc_markdown_no_issues_renders_clean() -> None:
    from podcast_transcript.library.qc import QCReport

    report = QCReport(verdict="passed", summary=SAMPLE_SUMMARY, issues=[])
    md = format_qc_markdown(report, episode_id="show__2026-04-17__hello")
    assert "No issues reported" in md


def test_qc_result_dataclass_carries_retry_state() -> None:
    """Belt-and-suspenders: confirm the orchestrator result type round-trips its fields."""
    from podcast_transcript.library.qc import QCReport

    qr = QCResult(
        summary_md="...",
        report=QCReport(verdict="passed", summary="..."),
        attempts=1,
        retried=False,
    )
    assert qr.summary_md == "..."
    assert qr.attempts == 1
    assert qr.retried is False
