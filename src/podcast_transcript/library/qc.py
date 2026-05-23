"""Quality-control pass over a generated summary.

The QC call is **fresh** — it does not see the summariser's reasoning,
only the transcript and the summary side by side. That separation is the
whole point: a QC pass that reuses the summariser's context would just
re-justify whatever the summariser already wrote.

What we check (per the project spec):

1. **Faithfulness** — every factual claim in the summary is supported by
   the transcript.
2. **Numbers & names** — every number, drug name, dosage, study, and
   proper noun in the summary appears (and matches) in the transcript.
3. **Quote accuracy** — quoted text is verbatim or very close; speaker
   attribution is correct.
4. **Coverage** — we sample 5 random transcript chunks (split into 10
   roughly-equal segments, seeded on ``episode_id`` so it's reproducible)
   and ask whether each is reflected somewhere in the summary.
5. **No contamination** — the summary doesn't smuggle in outside knowledge.

The QC call uses ``output_config.format`` with a JSON schema so the
verdict is structured and parseable. Issues are rendered into a
human-readable ``<id>.qc.md`` file by :func:`format_qc_markdown`.

Retry orchestration: :func:`run_summary_with_qc` calls the summariser,
runs QC, and on ``flagged``/``failed`` it regenerates the summary once
with the QC notes attached, then re-QCs. If the retry still fails, the
**broken summary is preserved** (not overwritten) and ``qc_status`` is
recorded as ``failed`` — surfaced via ``pending-vocab.md``.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Literal

from .summarise import (
    DEFAULT_SUMMARISE_MODEL,
    AnthropicClientLike,
    SummariseInput,
    SummariserError,
    summarise_transcript,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "COVERAGE_CHUNK_COUNT",
    "COVERAGE_SAMPLE_SIZE",
    "DEFAULT_QC_MODEL",
    "QCIssue",
    "QCReport",
    "QCResult",
    "format_qc_markdown",
    "qc_summary",
    "run_summary_with_qc",
    "sample_coverage_chunks",
]


DEFAULT_QC_MODEL: Final[str] = "claude-opus-4-7"
COVERAGE_CHUNK_COUNT: Final[int] = 10
COVERAGE_SAMPLE_SIZE: Final[int] = 5
_MIN_COVERAGE_CHUNK_CHARS: Final[int] = 50
_QC_MAX_TOKENS: Final[int] = 4000


Verdict = Literal["passed", "flagged", "failed"]
IssueCategory = Literal[
    "faithfulness",
    "numbers_and_names",
    "quote_accuracy",
    "coverage",
    "contamination",
]


@dataclass
class QCIssue:
    """One concrete problem the QC pass found in the summary."""

    category: IssueCategory
    severity: Literal["low", "medium", "high"]
    description: str
    summary_excerpt: str | None = None
    suggested_fix: str | None = None


@dataclass
class QCReport:
    """Full QC verdict for one summary."""

    verdict: Verdict
    summary: str
    issues: list[QCIssue] = field(default_factory=list)
    coverage_chunks: list[str] = field(default_factory=list)
    raw_json: dict[str, Any] = field(default_factory=dict)


@dataclass
class QCResult:
    """End-to-end orchestrator result: summary + QC report + retry trace."""

    summary_md: str
    report: QCReport
    attempts: int  # 1 on first-pass pass; 2 if retried
    retried: bool


# ---------------------------------------------------------------------------
# Coverage chunk sampling
# ---------------------------------------------------------------------------


def sample_coverage_chunks(
    transcript: str,
    *,
    seed: str,
    chunk_count: int = COVERAGE_CHUNK_COUNT,
    sample_size: int = COVERAGE_SAMPLE_SIZE,
) -> list[str]:
    """Split *transcript* into *chunk_count* segments and sample *sample_size*.

    Sampling uses ``random.Random(seed)`` so the same episode always
    produces the same sample — that means a re-run of QC after a vocab
    edit returns the same coverage verdict the first time, instead of
    flapping.

    Whitespace is preserved within each chunk. Empty transcripts return
    an empty list.
    """
    if chunk_count < 1:
        raise ValueError(f"chunk_count must be >= 1, got {chunk_count}")
    if sample_size < 0:
        raise ValueError(f"sample_size must be >= 0, got {sample_size}")
    stripped = transcript.strip()
    if not stripped:
        return []

    # Character-based split keeps the implementation deterministic regardless
    # of transcript shape (one line per cue vs paragraphs). Anything below
    # ~50 chars per slice produces meaningless single-character chunks, so
    # fall back to per-line for short transcripts.
    total_len = len(stripped)
    if total_len < chunk_count * _MIN_COVERAGE_CHUNK_CHARS:
        candidates = [line for line in stripped.splitlines() if line.strip()]
        return candidates[:sample_size] if candidates else [stripped]

    chunk_size = total_len // chunk_count
    chunks: list[str] = []
    for i in range(chunk_count):
        start = i * chunk_size
        end = total_len if i == chunk_count - 1 else (i + 1) * chunk_size
        chunks.append(stripped[start:end].strip())

    rng = random.Random(seed)
    picked = rng.sample(range(len(chunks)), min(sample_size, len(chunks)))
    picked.sort()
    return [chunks[i] for i in picked]


# ---------------------------------------------------------------------------
# QC call
# ---------------------------------------------------------------------------


_QC_SYSTEM_INSTRUCTIONS = """You are a strict quality-control reviewer for
AI-generated podcast summaries. Compare the SUMMARY against the TRANSCRIPT
and return a structured verdict.

Rules:
- Faithfulness: every factual claim in the summary must be supported by the transcript.
- Numbers & names: every number, drug name, dosage, study citation, and proper noun in
  the summary must appear in the transcript and match exactly.
- Quote accuracy: text inside `> "..."` quote blocks must be verbatim (or very close);
  speaker attribution must match the transcript.
- Coverage: each provided COVERAGE CHUNK must be reflected somewhere in the summary
  (TL;DR, key points, takeaways, quotes, numbers, glossary — any of those count).
  Coverage failures are usually MEDIUM severity, not HIGH.
- No contamination: the summary must not introduce outside knowledge the transcript
  doesn't discuss.

Verdict scale:
- "passed" — no issues, or only LOW-severity nits a reader would shrug at.
- "flagged" — one or more MEDIUM-severity issues; the summary needs revision but is
  not actively misleading.
- "failed" — at least one HIGH-severity issue: a hallucinated fact, a wrong number,
  a fabricated quote, or smuggled-in outside knowledge.

Return a JSON object matching the provided schema. Do not include any prose outside
the JSON.
"""


_QC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["passed", "flagged", "failed"]},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": [
                            "faithfulness",
                            "numbers_and_names",
                            "quote_accuracy",
                            "coverage",
                            "contamination",
                        ],
                    },
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "description": {"type": "string"},
                    "summary_excerpt": {"type": "string"},
                    "suggested_fix": {"type": "string"},
                },
                "required": ["category", "severity", "description"],
            },
        },
        "coverage_notes": {"type": "string"},
    },
    "required": ["verdict", "issues"],
}


def _render_qc_user_block(summary_md: str, coverage_chunks: list[str]) -> str:
    chunk_block = (
        "\n\n".join(f"--- CHUNK {i + 1} ---\n{chunk}" for i, chunk in enumerate(coverage_chunks))
        if coverage_chunks
        else "(transcript too short to sample meaningful chunks)"
    )
    return (
        "SUMMARY (to audit):\n"
        "----------\n"
        f"{summary_md}\n"
        "----------\n\n"
        "COVERAGE CHUNKS (each one should be reflected somewhere in the summary):\n"
        f"{chunk_block}\n\n"
        "Return the JSON verdict now."
    )


def qc_summary(
    client: AnthropicClientLike,
    *,
    transcript: str,
    summary_md: str,
    seed: str,
    model: str = DEFAULT_QC_MODEL,
) -> QCReport:
    """Run the QC pass and return a structured :class:`QCReport`.

    *seed* is used for deterministic coverage sampling — pass the
    ``episode_id`` so the same episode always gets the same coverage
    selection.
    """
    if not transcript.strip():
        raise ValueError("cannot QC against an empty transcript")
    if not summary_md.strip():
        raise ValueError("cannot QC an empty summary")

    coverage_chunks = sample_coverage_chunks(transcript, seed=seed)
    user_text = _render_qc_user_block(summary_md, coverage_chunks)

    response = client.messages.create(
        model=model,
        max_tokens=_QC_MAX_TOKENS,
        thinking={"type": "adaptive"},
        system=[
            {"type": "text", "text": _QC_SYSTEM_INSTRUCTIONS},
            {
                "type": "text",
                "text": f"<transcript>\n{transcript}\n</transcript>",
                "cache_control": {"type": "ephemeral"},
            },
        ],
        messages=[{"role": "user", "content": user_text}],
        output_config={"format": {"type": "json_schema", "schema": _QC_SCHEMA}},
    )

    raw = _extract_first_text(response.content)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SummariserError(
            f"QC pass returned non-JSON content: {raw[:200]!r}",
        ) from exc

    issues = [
        QCIssue(
            category=issue["category"],
            severity=issue["severity"],
            description=issue["description"],
            summary_excerpt=issue.get("summary_excerpt"),
            suggested_fix=issue.get("suggested_fix"),
        )
        for issue in data.get("issues", [])
    ]
    return QCReport(
        verdict=data["verdict"],
        summary=summary_md,
        issues=issues,
        coverage_chunks=coverage_chunks,
        raw_json=data,
    )


def _extract_first_text(blocks: Iterable[Any]) -> str:
    for block in blocks:
        block_type = getattr(block, "type", None)
        if block_type is None and isinstance(block, dict):
            block_type = block.get("type")
            text = block.get("text", "")
        else:
            text = getattr(block, "text", "")
        if block_type == "text" and text:
            return str(text)
    raise SummariserError("QC pass returned no text block")


# ---------------------------------------------------------------------------
# Orchestrator (summary -> qc -> retry once)
# ---------------------------------------------------------------------------


def run_summary_with_qc(
    client: AnthropicClientLike,
    inp: SummariseInput,
    *,
    seed: str,
    summarise_model: str = DEFAULT_SUMMARISE_MODEL,
    qc_model: str = DEFAULT_QC_MODEL,
) -> QCResult:
    """End-to-end: summarise → QC → optionally regenerate once.

    Behaviour matches the project spec:

    - First pass ``passed`` → return that summary, ``attempts=1``.
    - First pass ``flagged`` or ``failed`` → regenerate the summary with
      the QC notes appended, then re-QC. Return whichever the second
      pass produced. ``attempts=2``, ``retried=True``.
    - If the second pass also fails, the broken summary is **kept** (not
      overwritten) and the caller is expected to mark
      ``qc_status="failed"`` on the JSONL record.
    """
    summary_md = summarise_transcript(client, inp, model=summarise_model)
    report = qc_summary(
        client,
        transcript=inp.transcript,
        summary_md=summary_md,
        seed=seed,
        model=qc_model,
    )
    if report.verdict == "passed":
        return QCResult(summary_md=summary_md, report=report, attempts=1, retried=False)

    # Retry once with QC findings attached.
    retry_input = SummariseInput(
        transcript=inp.transcript,
        podcast=inp.podcast,
        episode_title=inp.episode_title,
        pub_date=inp.pub_date,
        host=inp.host,
        guests=inp.guests,
        series=inp.series,
        series_part=inp.series_part,
        source_label=inp.source_label,
        qc_feedback=_render_issues_for_retry(report.issues),
    )
    retry_summary = summarise_transcript(client, retry_input, model=summarise_model)
    retry_report = qc_summary(
        client,
        transcript=inp.transcript,
        summary_md=retry_summary,
        seed=seed,
        model=qc_model,
    )
    return QCResult(
        summary_md=retry_summary,
        report=retry_report,
        attempts=2,
        retried=True,
    )


def _render_issues_for_retry(issues: list[QCIssue]) -> str:
    if not issues:
        return "No specific issues were listed, but the previous summary was flagged or failed."
    parts: list[str] = []
    for issue in issues:
        line = f"- ({issue.severity.upper()} {issue.category}) {issue.description}"
        if issue.summary_excerpt:
            line += f"\n  In: {issue.summary_excerpt!r}"
        if issue.suggested_fix:
            line += f"\n  Suggested fix: {issue.suggested_fix}"
        parts.append(line)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Markdown report rendering
# ---------------------------------------------------------------------------


def format_qc_markdown(report: QCReport, *, episode_id: str) -> str:
    """Render a :class:`QCReport` as the ``<id>.qc.md`` file body."""
    lines: list[str] = [
        f"# QC report — {episode_id}\n\n",
        f"**Verdict:** `{report.verdict}`  \n",
        f"**Issues:** {len(report.issues)}  \n",
        f"**Coverage chunks sampled:** {len(report.coverage_chunks)}\n\n",
    ]
    if not report.issues:
        lines.append("_No issues reported._\n\n")
    else:
        lines.append("## Issues\n\n")
        for i, issue in enumerate(report.issues, start=1):
            lines.append(
                f"### {i}. {issue.category} — {issue.severity}\n\n{issue.description}\n\n",
            )
            if issue.summary_excerpt:
                lines.append(f"> {issue.summary_excerpt}\n\n")
            if issue.suggested_fix:
                lines.append(f"**Suggested fix:** {issue.suggested_fix}\n\n")
    if report.coverage_chunks:
        lines.append("## Coverage chunks sampled\n\n")
        for i, chunk in enumerate(report.coverage_chunks, start=1):
            preview = chunk if len(chunk) <= 400 else chunk[:400] + "…"
            lines.append(f"**Chunk {i}:** {preview}\n\n")
    return "".join(lines)
