#!/usr/bin/env python3
"""End-to-end ingest of one podcast episode into the library.

Wires source resolution → transcribe (or accept on-disk transcript) →
summarise → QC → JSONL upsert → index rebuild. The summariser and QC
both require the `[summarise]` extra (`pip install -e '.[summarise]'`).

Examples::

    # Discover via RSS
    python podcast-library/scripts/ingest.py \\
        --rss "https://feeds.example.com/show.xml" \\
        --episode-index 0 \\
        --podcast "Show Title" --title "Episode One" --pub-date 2026-04-17

    # Already have a transcript on disk
    python podcast-library/scripts/ingest.py \\
        --transcript transcripts/show/2026-04-17_episode-one.txt \\
        --podcast "Show Title" --title "Episode One" --pub-date 2026-04-17
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from podcast_transcript.library.ingest import (  # noqa: E402
    IngestError,
    IngestPaths,
    IngestRequest,
    ingest_episode,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-root", type=Path, default=_REPO_ROOT / "podcast-library")
    parser.add_argument("--podcast", required=True, help="Podcast name as it should appear.")
    parser.add_argument("--title", dest="episode_title", required=True)
    parser.add_argument("--pub-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--host")
    parser.add_argument("--guests", nargs="*", default=[])
    parser.add_argument("--topics", nargs="*", default=[])
    parser.add_argument("--tags", nargs="*", default=[])
    parser.add_argument("--series")
    parser.add_argument("--series-part", type=int)
    parser.add_argument("--episode-number", type=int)
    parser.add_argument("--duration-seconds", type=int)

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--transcript", type=Path, help="Path to an already-prepared transcript.")
    src.add_argument("--url", help="Direct http(s) URL to the audio file.")
    src.add_argument("--rss", help="RSS feed URL.")
    src.add_argument("--page", help="Episode page URL.")

    parser.add_argument("--episode-regex")
    parser.add_argument("--episode-index", type=int)
    parser.add_argument("--strip-before", action="append", default=[])
    parser.add_argument("--strip-after", action="append", default=[])
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Lazy import — the anthropic SDK only loads when ingest actually runs.
    try:
        import anthropic
    except ImportError:
        print(
            "anthropic SDK not installed. Run: pip install -e '.[summarise]'",
            file=sys.stderr,
        )
        return 2
    client = anthropic.Anthropic()

    request = IngestRequest(
        podcast=args.podcast,
        episode_title=args.episode_title,
        pub_date=args.pub_date,
        transcript_path=args.transcript,
        url=args.url,
        rss_url=args.rss,
        page_url=args.page,
        episode_regex=args.episode_regex,
        episode_index=args.episode_index,
        host=args.host,
        guests=list(args.guests),
        proposed_topics=list(args.topics),
        tags=list(args.tags),
        series=args.series,
        series_part=args.series_part,
        episode_number=args.episode_number,
        duration_seconds=args.duration_seconds,
        strip_before=list(args.strip_before),
        strip_after=list(args.strip_after),
        timeout=args.timeout,
    )
    paths = IngestPaths(library_root=args.library_root)

    try:
        result = ingest_episode(request, paths=paths, client=client)
    except IngestError as exc:
        print(f"ingest failed: {exc}", file=sys.stderr)
        return 2

    print(f"episode:   {result.episode.id}")
    print(f"summary:   {result.summary_path}")
    print(f"qc:        {result.qc_path} (verdict: {result.qc_result.report.verdict})")
    print(f"transcript: {result.transcript_path}")
    if result.pending_topics:
        print(f"pending topics:   {', '.join(result.pending_topics)}")
    if result.pending_speakers:
        print(f"pending speakers: {', '.join(result.pending_speakers)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
