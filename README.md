# podcast_transcript

Local-only podcast download, transcription, and transcript cleanup for Apple
Silicon Macs, using [OpenAI Whisper](https://github.com/openai/whisper).

Audio processing runs on your Mac. CI is for lint/type-check/tests only — no
audio touches GitHub Actions.

## Requirements

- macOS on Apple Silicon
- [Homebrew](https://brew.sh)
- The setup script installs `ffmpeg` and `python@3.11`.

## Quick start

```bash
./scripts/setup.sh

# Download an episode
podcast-transcript download \
  "https://traffic.libsyn.com/secure/unsupervisedlearning/directionalselection_ungated.mp3" \
  razib_directional_selection

# Transcribe it
podcast-transcript transcribe razib_directional_selection.mp3

# (Optional) Clean up the transcript
podcast-transcript clean transcripts/razib_directional_selection.txt \
  --corrections-pack razib_khan
```

Or all of the above in one call:

```bash
podcast-transcript run \
  --url "https://traffic.libsyn.com/secure/unsupervisedlearning/directionalselection_ungated.mp3" \
  --slug razib_directional_selection \
  --corrections-pack razib_khan
```

`run` can also nominate an episode by RSS feed or by web-page URL, and will
prefer a publisher-hosted SRT/VTT transcript when one is declared (skipping
the Whisper step entirely):

```bash
# RSS — picks up <podcast:transcript> if the publisher declares one.
podcast-transcript run \
  --rss "https://feeds.example.com/show.xml" \
  --episode-index 0 \
  --slug latest_episode

# Episode page — scrapes the page for an SRT/VTT link, falls back to the audio.
podcast-transcript run \
  --page "https://example.com/episodes/42" \
  --slug ep42
```

Pass `--no-discover-transcript` to skip the publisher-transcript step and
always run Whisper (useful when the publisher's transcript is known to be
worse than what local Whisper produces).

Outputs are written to `./transcripts/` in five formats: `.txt`, `.srt`,
`.vtt`, `.tsv`, `.json` when Whisper runs. `clean` and `run` additionally
write `<slug>_clean.txt`. When a publisher transcript is used, the raw
caption text is written to `<slug>.txt` (one line per cue) and the audio
download / other Whisper outputs are skipped.

## Podcast library

There's a separate, structured archive under `podcast-library/` for
nominated episodes. It builds on top of `run` and adds AI-generated
summaries (with a QC pass), a controlled vocabulary for speakers and
topics, and four grep-friendly markdown indexes — all rebuilt from a
single `episodes.jsonl` source of truth.

```bash
# Install the optional Claude API dependency
pip install -e '.[summarise]'

# Ingest one episode end-to-end (RSS, page, direct URL, or pre-made transcript)
python podcast-library/scripts/ingest.py \
  --rss "https://feeds.example.com/show.xml" --episode-index 0 \
  --podcast "Show Title" --title "Episode One" --pub-date 2026-04-17

# Re-regenerate the by-* indexes from episodes.jsonl
python podcast-library/scripts/rebuild_indexes.py
```

See `podcast-library/README.md` for the full layout, the episode record
schema, the summary template, the vocab-promotion workflow, and the QC
retry rules.

## CLI reference

```text
podcast-transcript download URL STEM [--output-dir DIR] [--timeout SECONDS]
podcast-transcript transcribe AUDIO_FILE [--model MODEL] [--language LANG] [--output-dir DIR]
podcast-transcript clean INPUT [--output OUT | --in-place]
                                [--corrections FILE]... [--corrections-pack NAME]...
                                [--no-default-corrections] [--no-user-corrections]
                                [--reflow] [--sentences-per-paragraph N] [--quiet]
podcast-transcript add-correction WRONG [RIGHT] [--uncertain] [--dict PATH]
podcast-transcript run (--url URL
                        | --rss FEED [--episode-regex RE | --episode-index N]
                        | --page PAGE_URL)
                       --slug SLUG
                       [--audio-dir DIR] [--output-dir DIR]
                       [--model MODEL] [--language LANG]
                       [--strip-before REGEX]... [--strip-after REGEX]...
                       [--corrections FILE]... [--corrections-pack NAME]...
                       [--no-default-corrections] [--no-user-corrections]
                       [--no-discover-transcript]
                       [--reflow] [--sentences-per-paragraph N] [--timeout SECONDS]
```

Defaults: `--model large-v3`, `--language en`, `--output-dir transcripts`.

For autodetect language, pass `--language ""`. For a faster (less accurate)
model, pass `--model turbo`.

### Cleanup

`podcast-transcript clean` runs deterministic, dependency-free fixes for
common Whisper failure modes:

- Collapses runs of near-identical adjacent lines (looping hallucinations).
- Strips trailing outro garbage (non-Latin script, short fragments after the
  last well-formed English line).
- Detects paywall/preview cuts in the tail and warns when found (e.g. an
  episode that ends with "to hear the rest, subscribe at …").
- Applies word-bounded corrections from `data/corrections.toml` and any
  bundled packs you pass with `--corrections-pack` (currently:
  `razib_khan`).
- Annotates *uncertain* candidate corrections inline as
  `[?: original → suggested]` so you can grep, accept, or override later.
- Optional `--reflow` joins per-segment lines into prose paragraphs.

### Corrections layering

`clean` and `run` merge corrections from multiple sources in this order
(later layers override on conflicts):

1. Bundled defaults (`data/corrections.toml`) — skip with
   `--no-default-corrections`.
2. Bundled packs (`--corrections-pack NAME`, repeatable).
3. Per-user file at `~/.config/podcast_transcript/corrections.toml` — skip
   with `--no-user-corrections`.
4. Explicit `--corrections PATH` files (repeatable).

Add to the user file as you discover new mistranscriptions:

```bash
podcast-transcript add-correction "Razeeb" "Razib"            # confident
podcast-transcript add-correction "benorephora" --uncertain   # flag-only
```

The user file is rewritten in place by `add-correction` and is not intended
for hand-edited comments — keep prose docs in the bundled packs.

## How it works

- `download_podcast` streams the response to `<file>.part` with `Content-Type`
  validation and atomic rename, so you never end up with an HTML error page
  saved as `.mp3`.
- `transcribe_audio` lazy-imports `whisper`, loads the requested model, and
  writes all five output formats via `whisper.utils.get_writer`.
- `clean_transcript` runs five composable rule-based passes (loop collapser,
  outro stripper, preview-cut detector, corrections + uncertain annotator,
  optional reflow) — see `clean.py`.
- `run_pipeline` (in `pipeline.py`) chains source-resolution →
  (publisher-transcript fetch *or* download + transcribe) → ad-strip →
  clean. RSS feeds are parsed with stdlib `xml.etree.ElementTree` in
  `feed.py` (no `feedparser` dep) and pick up the Podcasting 2.0
  `<podcast:transcript>` element when present. Episode web pages are
  scraped with stdlib `html.parser` (`page_scrape.py`) for SRT/VTT and
  audio links. SRT/VTT caption text is converted to one prose line per
  cue in `transcript_fetch.py` so the same `clean.py` passes run
  unchanged.

Publisher transcripts are accepted only as SRT or VTT (HTML and JSON are
ignored). When both formats are offered, SRT wins — the cue grammar is
simpler so text extraction is more robust. `application/x-subrip` and
`text/srt` are treated as synonyms for `application/srt`.

First run downloads Whisper model weights to `~/.cache/whisper/`.

## Development

```bash
source venv/bin/activate
pip install -e ".[dev]"

ruff check .
ruff format --check .
mypy
pytest
```

The test suite mocks `whisper` (via `fake_whisper`) and the `anthropic`
client (via `fake_anthropic`), so it requires neither torch nor the Claude
API. CI never makes a real model call.

Optional extras:

| Extra        | Pulls in                | When you need it                          |
|--------------|-------------------------|-------------------------------------------|
| `whisper`    | `openai-whisper`, torch | To actually transcribe on your Mac        |
| `summarise`  | `anthropic`             | To run the podcast-library ingest         |
| `dev`        | pytest, ruff, mypy      | To work on the codebase                   |

## Project layout

```
.
├── pyproject.toml            # Project metadata, deps, tooling config
├── src/
│   └── podcast_transcript/
│       ├── cli.py                    # argparse entry point
│       ├── clean.py                  # rule-based transcript cleanup
│       ├── corrections_user.py       # per-user corrections file + bundled packs
│       ├── download.py               # stdlib-only HTTP download with validation
│       ├── feed.py                   # stdlib-only RSS-2.0 parser (incl. podcast:transcript)
│       ├── page_scrape.py            # stdlib HTML scraper for `--page` episode URLs
│       ├── pipeline.py               # end-to-end `run` orchestration
│       ├── transcribe.py             # lazy-imported whisper wrapper
│       ├── transcript_fetch.py      # fetch + SRT/VTT → text for publisher transcripts
│       ├── library/
│       │   ├── episode.py            # Episode dataclass + validator
│       │   ├── store.py              # JSONL read/write/upsert
│       │   ├── vocab.py              # canonical names + alias resolution
│       │   ├── indexes.py            # 4 markdown index regenerators + pending-vocab
│       │   ├── summarise.py          # anthropic SDK wrapper (lazy import)
│       │   ├── qc.py                 # faithfulness/coverage QC + retry orchestration
│       │   └── ingest.py             # end-to-end orchestration
│       └── data/
│           ├── corrections.toml             # general defaults
│           └── corrections.razib_khan.toml  # podcast-specific pack
├── podcast-library/          # structured archive (see its own README)
│   ├── audio/                # gitignored
│   ├── transcripts/<slug>/<id>.txt
│   ├── summaries/<slug>/<id>.md  (+ <id>.qc.md)
│   ├── index/                # episodes.jsonl + by-*.md + vocab/
│   └── scripts/              # ingest.py, summarise.py, qc_summary.py, rebuild_indexes.py
├── tests/                    # pytest unit tests (mock whisper + anthropic, in-process HTTP server)
├── scripts/
│   └── setup.sh              # brew + venv bootstrap
└── .github/workflows/ci.yml  # lint, type-check, test, shellcheck
```
