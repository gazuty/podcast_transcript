#!/usr/bin/env python3
"""Run the QC pass on an existing transcript + summary pair.

Useful when you want to re-grade a summary without regenerating it
(e.g. after tweaking the QC prompt). Writes the QC report next to the
summary as ``<summary>.qc.md`` and prints the verdict.

Does NOT update ``episodes.jsonl`` — re-run ingest for that.
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

from podcast_transcript.library.qc import format_qc_markdown, qc_summary  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", type=Path, help="Path to the transcript .txt")
    parser.add_argument("summary", type=Path, help="Path to the summary .md")
    parser.add_argument(
        "--episode-id",
        required=True,
        help="Used as the deterministic seed for coverage sampling.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Where to write the QC report (default: <summary>.qc.md).",
    )
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
    summary = args.summary.read_text(encoding="utf-8")
    report = qc_summary(
        client,
        transcript=transcript,
        summary_md=summary,
        seed=args.episode_id,
    )

    output = args.output or args.summary.with_suffix(".qc.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(format_qc_markdown(report, episode_id=args.episode_id), encoding="utf-8")
    print(f"verdict: {report.verdict}  ({len(report.issues)} issue(s))")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
