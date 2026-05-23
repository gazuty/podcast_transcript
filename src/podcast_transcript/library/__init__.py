"""Podcast library: structured episode records, summaries, and indexes.

The library lives logically inside this Python package so it can reuse
:mod:`podcast_transcript.pipeline` (download/transcribe), :mod:`clean`
(transcript cleanup), and :mod:`feed` / :mod:`page_scrape` (source
resolution). The on-disk artifacts live under ``podcast-library/`` at the
repo root; this module is just the code that reads, writes, and
regenerates them.

See ``podcast-library/README.md`` for the data layout and CLI workflows.
"""

from __future__ import annotations
