#!/usr/bin/env python3
"""Regenerate the four `by-*.md` indexes (plus `pending-vocab.md`) from `episodes.jsonl`.

Idempotent and side-effect-free apart from those five files. Safe to run
anytime — the JSONL is the source of truth.

Usage::

    python podcast-library/scripts/rebuild_indexes.py
    python podcast-library/scripts/rebuild_indexes.py --library-root /path/to/library
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running the script straight from the repo without installing.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from podcast_transcript.library.indexes import rebuild_all  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library-root",
        type=Path,
        default=_REPO_ROOT / "podcast-library",
        help="Path to the podcast-library directory (default: %(default)s).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    index_dir = args.library_root / "index"
    if not index_dir.is_dir():
        print(f"index directory not found: {index_dir}", file=sys.stderr)
        return 2

    written = rebuild_all(index_dir=index_dir)
    for path in written.values():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
