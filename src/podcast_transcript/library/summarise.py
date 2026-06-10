"""Summariser: transcript → structured Markdown summary via the Claude API.

The :mod:`anthropic` SDK is imported lazily so the base package stays
zero-dep. Install with::

    pip install -e '.[summarise]'

Design notes:

- **One call per summary.** Streamed so we don't trip the SDK's
  non-streaming timeout for long outputs.
- **Transcript cached as a ``system`` block.** Both the summary and the
  separate QC call (see :mod:`.qc`) re-use the same prefix; with the 5-min
  cache TTL the second call lands as a cache read at ~0.1x input price.
- **Adaptive thinking enabled.** Opus 4.7 only supports
  ``thinking={"type": "adaptive"}``; ``budget_tokens`` is gone and sampling
  params (``temperature``/``top_p``/``top_k``) 400 the request.
- **Episode metadata is rendered into the user turn**, not the system
  block, so the cached transcript prefix doesn't get invalidated by
  per-episode strings (title, slug, etc.). This matters because the QC
  call also caches the same transcript.
- **The output format is fixed Markdown** (the per-episode template in
  ``podcast-library/README.md``). We do *not* use ``output_config`` JSON
  schema here — the summary is human-read prose, and Markdown enums
  better than a forced JSON wrapper.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_SUMMARISE_MODEL",
    "AnthropicClientLike",
    "SummariseInput",
    "SummariserError",
    "summarise_transcript",
    "wrap_api_errors",
]


DEFAULT_SUMMARISE_MODEL: Final[str] = "claude-opus-4-7"
# Generous ceiling — summaries are typically <2k tokens, but we want
# headroom for very long episodes plus the optional QC retry's extra
# context. Streaming is forced (see below) so timeout isn't a concern.
DEFAULT_MAX_TOKENS: Final[int] = 8000


class SummariserError(RuntimeError):
    """Raised when the summariser can't produce a usable summary."""


@contextmanager
def wrap_api_errors(what: str) -> Iterator[None]:
    """Translate Anthropic SDK exceptions into :class:`SummariserError`.

    The SDK is an optional dependency, so its exception types can't be
    imported here without breaking the zero-dep base package; matching on
    the exception's module keeps the "library code raises typed errors"
    convention without the import. Anything not from the SDK re-raises
    untouched — a bug in our code shouldn't be dressed up as an API failure.
    """
    try:
        yield
    except Exception as exc:
        module = type(exc).__module__ or ""
        if module == "anthropic" or module.startswith("anthropic."):
            raise SummariserError(f"{what} API call failed: {exc}") from exc
        raise


# A structural Protocol so tests can inject a fake client without
# depending on the real ``anthropic`` package being installed. Any
# object that exposes a ``messages`` attribute conforms — both the real
# ``anthropic.Anthropic`` and the test fake satisfy this trivially.
class AnthropicClientLike(Protocol):
    """Minimum surface area the summariser needs from an Anthropic client."""

    messages: Any


@dataclass
class SummariseInput:
    """Bundle of context the summariser needs for one episode."""

    transcript: str
    podcast: str
    episode_title: str
    pub_date: str
    host: str | None = None
    guests: tuple[str, ...] = ()
    series: str | None = None
    series_part: int | None = None
    source_label: str = "whisper"
    qc_feedback: str | None = None  # populated on a retry — see :mod:`.qc`


# The system prompt is intentionally static so the prefix stays cache-stable.
# Episode-specific facts go in the user turn.
_SYSTEM_INSTRUCTIONS = """You are a senior podcast summariser writing for a private,
durable library of episode summaries. Your readers know the subject area; they
want signal density, not marketing copy.

Output STRICT Markdown matching the template below. Use the exact section
headings, in the exact order, with no extra preamble or postamble. Quote
spoken material verbatim; never paraphrase inside a quote block. If a
section has nothing substantive to report, write `_None._` rather than
omitting the section.

Hard constraints:
- Every factual claim must be supported by the transcript provided.
- Names, drug dosages, study citations, and numbers must match the
  transcript exactly. If the transcript is ambiguous, mark the entry in
  "Open questions / things to verify" rather than guessing.
- Do not introduce outside knowledge the episode did not actually discuss.
- Speaker attribution on quotes must match the transcript.

Template:

# <Episode title>

## TL;DR
2-3 sentences.

## Key points
- Bullet points, each a discrete claim or framework.

## Key learnings / takeaways
- What a listener should walk away knowing or doing differently.

## Notable quotes
> "..." — Speaker, [hh:mm:ss] (only if the timestamp is present in the transcript)

## Numbers, studies, named entities
- Drugs, dosages, studies cited, papers, people, organisations — verbatim.

## Open questions / things to verify
- Anything the episode asserts strongly but does not substantiate.

## Glossary (if technical)
- Term — short definition as used in this episode.
"""


def _render_metadata_block(inp: SummariseInput) -> str:
    """Render episode metadata into the user turn (NOT the cached system block)."""
    guests = ", ".join(inp.guests) if inp.guests else "—"
    series_line = (
        f"\n- Series: {inp.series} (part {inp.series_part})"
        if inp.series and inp.series_part is not None
        else (f"\n- Series: {inp.series}" if inp.series else "")
    )
    return (
        "Episode metadata:\n"
        f"- Podcast: {inp.podcast}\n"
        f"- Title: {inp.episode_title}\n"
        f"- Date: {inp.pub_date}\n"
        f"- Host: {inp.host or '—'}\n"
        f"- Guests: {guests}\n"
        f"- Transcript source: {inp.source_label}"
        f"{series_line}\n\n"
        "Write the summary now, using the template from the system prompt. "
        "Make the H1 match the episode title exactly."
    )


def _build_user_blocks(inp: SummariseInput) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if inp.qc_feedback:
        # On a retry, prepend the QC findings so the model can see what
        # the previous summary got wrong. We keep this OUTSIDE the cached
        # transcript block so the cache prefix stays warm.
        blocks.append(
            {
                "type": "text",
                "text": (
                    "The previous summary failed QC with the following issues. "
                    "Address each one in this revision:\n\n"
                    f"{inp.qc_feedback.strip()}"
                ),
            },
        )
    blocks.append({"type": "text", "text": _render_metadata_block(inp)})
    return blocks


def _build_system_blocks(transcript: str) -> list[dict[str, Any]]:
    """System turn = instructions + the cacheable transcript block.

    The transcript is the longest piece of content in the request and the
    one that's stable across the summarise + QC call pair, so we mark it
    with ``cache_control`` for prefix caching. Min cacheable prefix on
    Opus 4.7 is 4096 tokens — short transcripts won't cache, which is
    fine (no error, just no speedup).
    """
    return [
        {"type": "text", "text": _SYSTEM_INSTRUCTIONS},
        {
            "type": "text",
            "text": f"<transcript>\n{transcript}\n</transcript>",
            "cache_control": {"type": "ephemeral"},
        },
    ]


def summarise_transcript(
    client: AnthropicClientLike,
    inp: SummariseInput,
    *,
    model: str = DEFAULT_SUMMARISE_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Generate the Markdown summary for one episode.

    Streams the response so we don't trip the SDK's ~10-minute non-stream
    timeout guard on long outputs. Returns the full text body.

    Raises :class:`SummariserError` if the model returns no text content,
    or — via :func:`wrap_api_errors` — if the Anthropic SDK raises (rate
    limit, timeout, connection failure).
    """
    if not inp.transcript.strip():
        raise SummariserError("cannot summarise an empty transcript")

    system_blocks = _build_system_blocks(inp.transcript)
    user_blocks = _build_user_blocks(inp)

    with (
        wrap_api_errors("summarise"),
        client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            system=system_blocks,
            messages=[{"role": "user", "content": user_blocks}],
        ) as stream,
    ):
        final = stream.get_final_message()

    text = _concat_text_blocks(final.content)
    if not text.strip():
        raise SummariserError(
            "summariser returned no text content "
            f"(stop_reason={getattr(final, 'stop_reason', None)!r})",
        )
    return text


def _concat_text_blocks(blocks: Iterable[Any]) -> str:
    """Join every ``type=="text"`` block on a Message response into one string."""
    parts: list[str] = []
    for block in blocks:
        # Real anthropic blocks expose ``.type`` and ``.text``; fake test
        # blocks may use a dict shape — handle both.
        block_type = getattr(block, "type", None)
        if block_type is None and isinstance(block, dict):
            block_type = block.get("type")
            text = block.get("text", "")
        else:
            text = getattr(block, "text", "")
        if block_type == "text" and text:
            parts.append(text)
    return "".join(parts)
