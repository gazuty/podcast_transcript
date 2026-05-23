# CLAUDE.md

Guidance for Claude (and other AI assistants) working in this repository.

## What this repo is

A small, local-only podcast download, transcription, and transcript-cleanup
tool for Apple Silicon Macs. It exposes a `podcast-transcript` CLI with five
subcommands:

- `download` — fetch a podcast MP3 from a direct URL.
- `transcribe` — run OpenAI Whisper on a local audio file and write outputs
  (`.txt`, `.srt`, `.vtt`, `.tsv`, `.json`) to `./transcripts/`.
- `clean` — apply rule-based fixes to a Whisper transcript (loop collapse,
  outro stripping, preview-cut detection, term corrections + inline
  uncertainty annotations, optional paragraph reflow).
- `add-correction` — append/update an entry in the per-user corrections
  TOML at `~/.config/podcast_transcript/corrections.toml`.
- `run` — end-to-end pipeline. Nominate via `--url`, `--rss` (with
  `--episode-regex` / `--episode-index`), or `--page` (episode web URL).
  By default `run` checks for a publisher-hosted SRT/VTT transcript
  (Podcasting 2.0 `<podcast:transcript>` tag on RSS, or an `.srt`/`.vtt`
  `<a href>` on a page) and fetches that instead of transcribing — only
  falling back to download + Whisper when no usable transcript is
  declared. Always writes `<slug>_clean.txt` after cleanup.

On top of the CLI there is a **podcast library** under
`podcast-library/`: a structured archive of nominated episodes with
AI-generated summaries (with a QC pass), a controlled vocabulary for
speakers/topics, and four grep-friendly markdown indexes, all
regenerated from a single `episodes.jsonl` source of truth. The
orchestration lives in `src/podcast_transcript/library/`; see the
*Podcast library* section below.

Audio processing is intended to run on the user's Mac, **never** in GitHub
Actions. CI exists only for lint/type-check/tests/shellcheck.

## Layout

```
.
├── pyproject.toml            # PEP 621 metadata; ruff/mypy/pytest config
├── src/
│   └── podcast_transcript/
│       ├── __init__.py
│       ├── __main__.py              # `python -m podcast_transcript`
│       ├── cli.py                   # argparse entry point (also wired to console_scripts)
│       ├── clean.py                 # rule-based transcript cleanup
│       ├── corrections_user.py      # per-user corrections file + bundled packs
│       ├── download.py              # stdlib-only HTTP download with validation
│       ├── feed.py                  # stdlib RSS-2.0 parser + <podcast:transcript> reader
│       ├── page_scrape.py           # stdlib HTML scrape for `--page` episode URLs
│       ├── pipeline.py              # end-to-end `run` orchestration
│       ├── transcribe.py            # lazy-imported whisper wrapper
│       ├── transcript_fetch.py     # fetch + SRT/VTT → text for publisher transcripts
│       ├── library/                 # podcast-library orchestration
│       │   ├── episode.py           # Episode dataclass + validator
│       │   ├── store.py             # JSONL read/write/upsert
│       │   ├── vocab.py             # canonical-name + alias resolution
│       │   ├── indexes.py           # 4 markdown indexes + pending-vocab
│       │   ├── summarise.py         # anthropic SDK wrapper (lazy import)
│       │   ├── qc.py                # QC pass + retry orchestration
│       │   └── ingest.py            # end-to-end orchestration
│       ├── py.typed                 # PEP 561 marker
│       └── data/
│           ├── corrections.toml             # general defaults
│           └── corrections.razib_khan.toml  # podcast-specific pack
├── podcast-library/                 # data root (see podcast-library/README.md)
│   ├── audio/                       # gitignored
│   ├── transcripts/<slug>/<id>.txt
│   ├── summaries/<slug>/<id>.md     # plus <id>.qc.md
│   ├── index/                       # episodes.jsonl + by-*.md + vocab/{topics,speakers}.json
│   └── scripts/                     # thin argparse shims → library/* code
├── tests/
│   ├── conftest.py                  # fake_whisper, fake_anthropic, http_server, autouse user-corrections isolation
│   ├── test_clean.py
│   ├── test_cli.py
│   ├── test_download.py
│   ├── test_feed.py
│   ├── test_library_episode.py
│   ├── test_library_indexes.py
│   ├── test_library_ingest.py
│   ├── test_library_qc.py
│   ├── test_library_store.py
│   ├── test_library_vocab.py
│   ├── test_page_scrape.py
│   ├── test_pipeline.py
│   ├── test_transcribe.py
│   └── test_transcript_fetch.py
├── scripts/
│   └── setup.sh              # brew install + venv + `pip install -e .[whisper,dev]`
└── .github/workflows/
    └── ci.yml                # ubuntu: ruff + mypy + pytest, separate shellcheck job
```

## Common commands

```bash
# One-time bootstrap on a fresh Mac
./scripts/setup.sh

# Activate the venv each shell session
source venv/bin/activate

# Use the CLI
podcast-transcript download <url> <stem>
podcast-transcript transcribe <file.mp3> [--model turbo]
podcast-transcript clean <transcript.txt> [--corrections-pack razib_khan] [--reflow]
podcast-transcript add-correction <wrong> <right> [--uncertain]
podcast-transcript run --url <url> --slug <stem> [--corrections-pack razib_khan]
podcast-transcript run --rss <feed> --episode-regex <re> --slug <stem>     # uses <podcast:transcript> when present
podcast-transcript run --page <url> --slug <stem>                          # scrapes SRT/VTT or audio link from page
podcast-transcript run --rss <feed> ... --no-discover-transcript           # force Whisper even if a transcript is declared

# Podcast library (optional `[summarise]` extra required for ingest)
python podcast-library/scripts/ingest.py --rss <feed> --episode-index 0 \
  --podcast "Show Title" --title "Episode One" --pub-date 2026-04-17
python podcast-library/scripts/rebuild_indexes.py   # regenerate by-*.md from episodes.jsonl
python podcast-library/scripts/qc_summary.py <transcript.txt> <summary.md> --episode-id <id>

# Dev loop
ruff check .                 # lint
ruff format --check .        # formatting
mypy                         # type-check (config in pyproject.toml)
pytest -v                    # unit tests (mocks whisper, no torch needed)
```

## Architectural decisions worth knowing

### `openai-whisper` is an *optional* dependency.

The package itself has zero required runtime deps. `openai-whisper` (and its
torch transitive) is installed via the `whisper` extra:

```bash
pip install -e ".[whisper]"   # for actual transcription
pip install -e ".[dev]"       # CI / dev — fast install, no torch
```

`transcribe.py` imports `whisper` lazily inside `transcribe_audio` and raises
a `TranscriptionError` with a useful install hint if it's missing. This is
why CI can lint/type-check/test in seconds instead of minutes.

### Tests never import the real whisper.

`tests/conftest.py` provides a `fake_whisper` fixture that injects a
`MagicMock` into `sys.modules["whisper"]` and `sys.modules["whisper.utils"]`.
Any test that exercises `transcribe_audio` must take this fixture (otherwise
the lazy import would fail in CI where whisper isn't installed).

### Download tests use an in-process HTTP server, not the real network.

`tests/conftest.py` exposes an `http_server` fixture that starts a
`http.server.HTTPServer` on `127.0.0.1` with a per-test responder callable.
Tests are fully hermetic.

### `download.py` uses `urllib`, not `requests`/`httpx`.

Deliberate: keeps required runtime deps at zero. The function streams to
`<output>.part` and renames atomically on success. Validates the response
`Content-Type` is `audio/*` or `application/octet-stream` to avoid silently
saving an HTML error page as `.mp3`. Restricts URL schemes to `http`/`https`
so `urlopen` can't be coerced into reading local files.

### `clean.py` is intentionally rule-based, no LLM.

Five composable passes — each is a pure function, easy to read, easy to
test, free, and deterministic:

1. **Loop collapser** — `difflib.SequenceMatcher` ratio against a run-leader;
   collapse runs of `min_run` (default 3) similar adjacent lines.
2. **Outro stripper** — find the *last* well-formed English line (pure ASCII,
   sentence-final punctuation OR ≥30 chars) and drop everything after it.
   Walking from the end gives up too early when a fragment like `"you"` sits
   between real content and script-mismatch outro junk.
3. **Preview-cut detector** — scan the last 5 % of non-empty lines for a
   small set of paywall phrases (`subscribe to hear`, `to hear the rest`,
   `head over to … .substack.com`, etc.). Emits a `WARNING` log line when
   matched but does not modify the transcript.
4. **Corrections + uncertain annotator** — word-bounded regex substitutions
   from one or more TOML files. Each file may contribute `[corrections]`
   (silent replacements) and `[uncertain]` (annotated inline as
   `[?: original → suggested]`). Layering order: bundled defaults →
   `--corrections-pack` → per-user file → explicit `--corrections`.
5. **Paragraph reflow** (opt-in) — collapse Whisper's per-segment lines into
   prose paragraphs of N sentences each.

If you reach for an LLM-based polish pass later, add it as an *optional*
extra (mirror the `whisper` extra pattern) so the rule-based pipeline stays
zero-dep.

### `pipeline.py` orchestrates the end-to-end `run` subcommand.

`run_pipeline` resolves the source (`--url`, `--rss`, or `--page`), then
either fetches a publisher SRT/VTT transcript or runs download +
transcribe. Both paths then go through ad-strip → clean and write
`<transcripts_dir>/<slug>_clean.txt`.

- Publisher transcript discovery is on by default. With `--rss`, we read
  the Podcasting 2.0 `<podcast:transcript>` element from the selected
  item. With `--page`, `page_scrape.py` HTML-parses the page for `<a>` /
  `<link>` / `<audio>` / `<source>` elements pointing at an SRT/VTT (and
  also the audio fallback). `--no-discover-transcript` disables this
  step entirely.
- When a publisher transcript is used, the audio is **not** downloaded
  and Whisper does **not** run; `PipelineResult.audio_path` is `None`
  and `transcript_source` is `"rss"` or `"page"` instead of `"whisper"`.
- The ad-strip step uses repeatable `--strip-before` / `--strip-after`
  regex flags; `--strip-after` only fires when the match lands in the
  tail half of the transcript, so an outro phrase that happens to occur
  naturally mid-show doesn't chop the body. It runs on whichever raw
  text source produced the transcript (Whisper or fetched).

### `transcript_fetch.py` converts SRT/VTT to one line per cue.

The pipeline accepts publisher transcripts only as SRT (`application/srt`,
`application/x-subrip`, `text/srt`) or VTT (`text/vtt`, `application/vtt`).
HTML and JSON are deliberately ignored — HTML stripping is brittle, and
JSON adoption is too low to be worth a parser. SRT wins ties over VTT
because the cue grammar is simpler (no `WEBVTT`/`NOTE`/`STYLE`/`REGION`
blocks, no cue settings) so text extraction is more robust.

Converted output is one line per cue, which matches Whisper's `.txt`
shape so `clean.py` runs unchanged.

### `page_scrape.py` is the `--page` source.

Uses stdlib `html.parser` (no `beautifulsoup4` dep). Heuristics are
deliberately conservative — first matching `<a href>` / `<link href>` /
`<source src>` wins per category. URLs are matched on the path portion
only so query strings (`?token=…`) don't defeat the extension check.
Relative hrefs are resolved against the page URL with
`urllib.parse.urljoin`.

### `feed.py` is a deliberately narrow RSS-2.0 parser.

Stdlib `xml.etree.ElementTree`, no `feedparser` dep. Only reads `<title>`,
`<enclosure url=…>`, and `<pubDate>`. Atom feeds raise `FeedParseError`.
Items missing an enclosure are skipped silently. Capped at 10 MiB to avoid
pathological responses chewing memory.

### `library/` builds the podcast archive on top of `pipeline.py`.

Logic in `src/podcast_transcript/library/`; on-disk artifacts under
`podcast-library/`. The two are connected only via `IngestPaths`, so the
library root can be relocated per-environment without touching code.

- **`episode.py`** — the schema spine. `Episode` is a `dataclass` with a
  hand-rolled `validate()` (no pydantic — base package stays zero-dep).
  Empty optionals are dropped on serialise so JSONL diffs stay tight.
  `id` follows `<podcast-slug>__<YYYY-MM-DD>__<title-slug>` so ingest is
  idempotent and by-date sorting is a string sort.
- **`store.py`** — atomic JSONL read/write/upsert. The whole file is
  rewritten on every upsert (fine at the scale of a personal library;
  avoids any concurrency story). Sorted by id for stable diffs.
- **`vocab.py`** — canonical names + alias resolution from
  `vocab/{topics,speakers}.json`. Unknown names auto-add via
  `add_pending()` with `pending: true`; the resolved-but-pending names
  also land on `Episode.pending_topics` / `pending_speakers` so
  `pending-vocab.md` can surface them. Aliases are never auto-created.
- **`indexes.py`** — pure builders for `by-{speaker,topic,date,podcast}.md`
  plus `pending-vocab.md`. `rebuild_all()` is idempotent (re-running on
  an unchanged JSONL produces byte-identical output apart from the
  "Generated at" header line).
- **`summarise.py`** — lazy `import anthropic` so the base package stays
  zero-dep. Uses Opus 4.7 with **adaptive thinking** and streams the
  response. The transcript is sent as a **cached system block**
  (`cache_control: ephemeral`) so the separate QC call lands as a cache
  read at ~0.1× input price. Per-episode metadata (title, host, guests)
  goes in the user turn, not the system block, so the cached prefix
  stays stable across ingests.
- **`qc.py`** — fresh `messages.create()` call (no shared context with
  the summariser) using `output_config.format` with a JSON schema for
  structured `verdict` / `issues`. Coverage sampling splits the
  transcript into 10 segments, samples 5 via
  `random.Random(episode_id)` so the same episode always gets the same
  chunks. `run_summary_with_qc()` orchestrates one retry on
  `flagged`/`failed`, attaching QC notes to the regen prompt. On a
  second failure, the broken summary is **preserved** (never silently
  overwritten) — if a previous good summary exists at the path, the
  failed retry is written to `<id>.failed.md` next to it.
- **`ingest.py`** — single entry point `ingest_episode()` that wires
  source resolution (reusing `run_pipeline` when not given a transcript
  path directly) → summarise → QC → vocab → JSONL upsert → rebuild
  indexes. `IngestPaths` is the only thing that knows where things live.

The `anthropic` SDK is an *optional* dependency under the `summarise`
extra (`pip install -e '.[summarise]'`). Tests use the `fake_anthropic`
fixture in `tests/conftest.py`, which mirrors the `fake_whisper`
pattern — it programs streamed and `create()` responses ahead of time
and records the call kwargs for assertions, so CI never makes an API
call.

When extending the library:

- New JSONL fields: extend `Episode.to_dict` / `from_dict` /
  `validate`, then write a test in `test_library_episode.py` that
  covers the round-trip *and* a validation failure case.
- New index file: add a `build_*` function in `indexes.py` and include
  it in the `rebuild_all` outputs dict; document the file in
  `podcast-library/README.md`.
- New QC heuristic: extend the `_QC_SCHEMA` JSON schema and the
  prompt; the orchestrator handles the rest.

## Conventions

- **Local-only audio.** Don't add CI steps that download or transcribe audio.
  CI runs on Ubuntu and is for static checks only.
- **`from __future__ import annotations`** at the top of every module.
- **All public functions are typed.** mypy is configured `strict = true`.
- **Errors are raised, not printed.** The CLI layer translates exceptions into
  log lines + non-zero exit codes; library code raises typed exceptions.
- **Ruff is the source of truth for style.** Don't hand-format; run
  `ruff format`.
- **Keep the dependency surface tiny.** Anything new in `dependencies =` needs
  a real reason. Optional extras are fine.

## CI

Two jobs, both on `ubuntu-latest`:

1. **lint-and-test**
   - `ruff check .`
   - `ruff format --check .`
   - `mypy`
   - `pytest -v`
2. **shellcheck** — runs `shellcheck` on `./scripts`.

If you add a new shell script under `scripts/`, the shellcheck job picks it up
automatically.

## When making changes

- README, CLAUDE.md, `podcast-library/README.md`, and `pyproject.toml`
  should stay in sync on user-visible changes (commands, output paths,
  supported models, deps).
- Don't introduce a step that requires network access during transcription
  or summarisation in CI — the design assumes audio is already on disk and
  any Anthropic API call only happens locally.
- New tests go under `tests/`. Use the existing `fake_whisper`,
  `fake_anthropic`, and `http_server` fixtures rather than reaching for
  the real network, torch, or the Claude API.
- New CLI subcommands: add the parser in `cli.py`, a `_run_<name>` handler,
  and a corresponding test in `tests/test_cli.py`.
- New `podcast-library/scripts/*.py` shims must stay thin — argparse +
  call into `podcast_transcript.library.*`. Logic and tests live in the
  Python package, not the script.

## Known not-done items

- No `LICENSE` file (repo is private).
- No release/publishing pipeline. The package isn't on PyPI; install is local
  via `pip install -e`.
