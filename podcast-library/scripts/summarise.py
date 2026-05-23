#!/usr/bin/env python3
"""Standalone summariser: takes a transcript file, writes a Markdown summary.

Useful when you already have a transcript in the library and want to
regenerate just the summary without re-running ingest. Does NOT run QC
— see ``qc_summary.py`` for that. Does NOT update ``episodes.jsonl``.
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

from podcast_transcript.library.summarise import (  # noqa: E402
    SummariseInput,
    SummariserError,
    summarise_transcript,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", type=Path, help="Path to the transcript .txt")
    parser.add_argument("output", type=Path, help="Where to write the .md summary")
    parser.add_argument("--podcast", required=True)
    parser.add_argument("--title", dest="episode_title", required=True)
    parser.add_argument("--pub-date", required=True)
    parser.add_argument("--host")
    parser.add_argument("--guests", nargs="*", default=[])
    parser.add_argument("--source-label", default="whisper")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        import anthropic
    except ImportError:
        print(
            "anthropic SDK not installed. Run: pip install -e '.[summarise]'",
            file=sys.stderr,
        )
        return 2
    client = anthropic.Anthropic()

    transcript = args.transcript.read_text(encoding="utf-8")
    try:
        summary = summarise_transcript(
            client,
            SummariseInput(
                transcript=transcript,
                podcast=args.podcast,
                episode_title=args.episode_title,
                pub_date=args.pub_date,
                host=args.host,
                guests=tuple(args.guests),
                source_label=args.source_label,
            ),
        )
    except SummariserError as exc:
        print(f"summariser failed: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(summary, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
