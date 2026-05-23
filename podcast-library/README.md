# Podcast library

A structured, searchable archive of podcast transcripts and AI-generated
summaries. The on-disk layout below is the canonical state — every index
file is regenerated from `index/episodes.jsonl`, which is itself the
source of truth.

## Layout

```
podcast-library/
├── audio/                          # raw downloads (gitignored)
├── transcripts/<slug>/<id>.txt     # raw or whisper-cleaned transcript
│                       └── <id>.vtt│srt   # caption form when fetched from publisher
├── summaries/<slug>/<id>.md        # AI-generated summary (canonical format)
│                       └── <id>.qc.md      # QC report (verdict + issues)
├── index/
│   ├── episodes.jsonl              # one JSON record per episode — source of truth
│   ├── by-speaker.md               # generated: grouped by canonical speaker
│   ├── by-topic.md                 # generated: grouped by canonical topic
│   ├── by-date.md                  # generated: reverse-chronological
│   ├── by-podcast.md               # generated: grouped by show, then series
│   ├── pending-vocab.md            # generated: unknown topics/speakers awaiting review
│   └── vocab/
│       ├── topics.json             # controlled vocabulary
│       └── speakers.json           # controlled vocabulary
└── scripts/                        # thin shims → src/podcast_transcript/library
    ├── ingest.py
    ├── summarise.py
    ├── qc_summary.py
    └── rebuild_indexes.py
```

## Adding an episode

Three equivalent entry points; all flow through the same code path
(`podcast_transcript.library.ingest.ingest_episode`):

```bash
# Full pipeline: discover transcript → summarise → QC → index
python podcast-library/scripts/ingest.py \
  --rss "https://feeds.example.com/show.xml" \
  --episode-index 0 \
  --podcast "The Longevity Show with Dr. Hillary Lin"

# Page-URL source
python podcast-library/scripts/ingest.py \
  --page "https://example.com/episodes/42" \
  --podcast "The Longevity Show with Dr. Hillary Lin"

# If you already have a transcript on disk, skip download/transcribe
python podcast-library/scripts/ingest.py \
  --transcript transcripts/longevity-show-lin/2026-04-17_modern-lipid-playbook-part-2.txt \
  --podcast "The Longevity Show with Dr. Hillary Lin" \
  --title "The Modern Lipid Playbook Part 2"
```

Ingest is idempotent: re-running on the same source updates the existing
record in `episodes.jsonl` rather than appending a duplicate (keyed on
`id` — see [Episode ID format](#episode-id-format) below).

Every ingest:

1. Resolves the episode and produces a transcript (reusing `run_pipeline`
   from the main package, so publisher SRT/VTT is preferred over Whisper).
2. Sends the transcript to Claude Opus 4.7 for a structured summary
   (transcript prompt-cached as a system block).
3. Runs a separate QC pass — a fresh Opus 4.7 call that sees only the
   transcript and the summary, and produces a verdict (`passed` /
   `flagged` / `failed`) with specific issues.
4. On `flagged` or `failed`, regenerates the summary once with the QC
   notes attached. If the retry still fails, the broken summary is
   **preserved** (not overwritten), `qc_status: failed` is recorded on
   the JSONL record, and the episode surfaces in `pending-vocab.md`.
5. Normalises every topic and speaker against `vocab/{topics,speakers}.json`.
   Unknown entries are added with `pending: true`; you review them in
   `pending-vocab.md` and promote or alias them by editing the vocab file.
6. Regenerates the four `by-*.md` indexes.

## Episode ID format

`<podcast_slug>__<YYYY-MM-DD>__<title_slug>`. Both slugs are
lowercase-kebab-case with non-alphanumerics collapsed. Example:

```
longevity-show-lin__2026-04-17__modern-lipid-playbook-part-2
```

The double underscore makes the three parts trivially splittable.

## Searching

The first-cut search surface is plain text:

```bash
# By keyword across all transcripts
grep -rli "ApoB" podcast-library/transcripts/

# By keyword across summaries only
grep -rli "ApoB" podcast-library/summaries/

# Browse by speaker / topic / date / podcast
$EDITOR podcast-library/index/by-{speaker,topic,date,podcast}.md
```

Structured queries against `episodes.jsonl`:

```bash
# All episodes for a given podcast
jq -c 'select(.podcast_slug == "longevity-show-lin")' \
  podcast-library/index/episodes.jsonl

# All episodes with a given topic
jq -c 'select(.topics | index("ApoB"))' \
  podcast-library/index/episodes.jsonl

# All episodes whose QC failed
jq -c 'select(.summary.qc_status == "failed")' \
  podcast-library/index/episodes.jsonl
```

## Controlled vocabulary

`vocab/topics.json` and `vocab/speakers.json` have the same shape:

```json
{
  "canonical": {
    "Hillary Lin": {"added": "2026-04-17"}
  },
  "aliases": {
    "Dr. Hillary Lin": "Hillary Lin",
    "Hillary Lin, MD": "Hillary Lin"
  }
}
```

When the summariser proposes a name that's neither a canonical entry nor
an alias, it lands in `pending-vocab.md` with `pending: true` on the JSONL
record. You either:

- promote it: move the entry into `canonical`, drop the `pending` flag
  next time you ingest, OR
- alias it: add a row under `aliases` mapping it to an existing canonical
  name.

After editing the vocab, re-run `python podcast-library/scripts/rebuild_indexes.py`
to re-normalise the JSONL records and regenerate the by-* indexes.

## Regenerating indexes manually

```bash
python podcast-library/scripts/rebuild_indexes.py
```

Idempotent and side-effect-free apart from the four `by-*.md` files and
`pending-vocab.md`.

## Architecture

All logic lives in `src/podcast_transcript/library/`; the `scripts/*.py`
files are thin argparse shims. Tests live under `tests/test_library_*.py`
and use a fake `anthropic` client fixture so QC and summarisation are
exercised without making real API calls.

| Module | Responsibility |
|---|---|
| `episode.py` | `Episode` dataclass + `validate()` |
| `store.py` | JSONL read/write/upsert (atomic) |
| `vocab.py` | Canonical-name resolution + alias management |
| `indexes.py` | Regenerate the four `by-*.md` files |
| `summarise.py` | Anthropic SDK wrapper with prompt caching |
| `qc.py` | Faithfulness/coverage QC pass + retry orchestration |
| `ingest.py` | End-to-end orchestration |

The `anthropic` SDK is an *optional* dependency, installed via
`pip install -e '.[summarise]'`. The base package stays zero-dep at
runtime; CI continues to lint/type-check/test without it.
