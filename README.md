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

Outputs are written to `./transcripts/` in five formats: `.txt`, `.srt`,
`.vtt`, `.tsv`, `.json`. `clean` and `run` additionally write
`<slug>_clean.txt`.

## CLI reference

```text
podcast-transcript download URL STEM [--output-dir DIR] [--timeout SECONDS]
podcast-transcript transcribe AUDIO_FILE [--model MODEL] [--language LANG] [--output-dir DIR]
podcast-transcript clean INPUT [--output OUT | --in-place]
                                [--corrections FILE]... [--corrections-pack NAME]...
                                [--no-default-corrections] [--no-user-corrections]
                                [--reflow] [--sentences-per-paragraph N] [--quiet]
podcast-transcript add-correction WRONG [RIGHT] [--uncertain] [--dict PATH]
podcast-transcript run (--url URL | --rss FEED [--episode-regex RE | --episode-index N])
                       --slug SLUG
                       [--audio-dir DIR] [--output-dir DIR]
                       [--model MODEL] [--language LANG]
                       [--strip-before REGEX]... [--strip-after REGEX]...
                       [--corrections FILE]... [--corrections-pack NAME]...
                       [--no-default-corrections] [--no-user-corrections]
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
- `run_pipeline` (in `pipeline.py`) chains download → transcribe →
  ad-strip → clean. RSS feeds are parsed with stdlib
  `xml.etree.ElementTree` in `feed.py` (no `feedparser` dep).

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

The test suite mocks `whisper`, so it does not require torch and does not
download model weights.

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
│       ├── feed.py                   # stdlib-only RSS-2.0 parser
│       ├── pipeline.py               # end-to-end `run` orchestration
│       ├── transcribe.py             # lazy-imported whisper wrapper
│       └── data/
│           ├── corrections.toml             # general defaults
│           └── corrections.razib_khan.toml  # podcast-specific pack
├── tests/                    # pytest unit tests (mock whisper, in-process HTTP server)
├── scripts/
│   └── setup.sh              # brew + venv bootstrap
└── .github/workflows/ci.yml  # lint, type-check, test, shellcheck
```
