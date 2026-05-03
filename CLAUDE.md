# CLAUDE.md

Guidance for Claude (and other AI assistants) working in this repository.

## What this repo is

A small, local-only podcast download, transcription, and transcript-cleanup
tool for Apple Silicon Macs. It exposes a `podcast-transcript` CLI with three
subcommands:

- `download` — fetch a podcast MP3 from a direct URL.
- `transcribe` — run OpenAI Whisper on a local audio file and write outputs
  (`.txt`, `.srt`, `.vtt`, `.tsv`, `.json`) to `./transcripts/`.
- `clean` — apply rule-based fixes to a Whisper transcript (loop collapse,
  outro stripping, term corrections, optional paragraph reflow).

Audio processing is intended to run on the user's Mac, **never** in GitHub
Actions. CI exists only for lint/type-check/tests/shellcheck.

## Layout

```
.
├── pyproject.toml            # PEP 621 metadata; ruff/mypy/pytest config
├── src/
│   └── podcast_transcript/
│       ├── __init__.py
│       ├── __main__.py       # `python -m podcast_transcript`
│       ├── cli.py            # argparse entry point (also wired to console_scripts)
│       ├── clean.py          # rule-based transcript cleanup
│       ├── download.py       # stdlib-only HTTP download with validation
│       ├── transcribe.py     # Lazy-imported whisper wrapper
│       ├── py.typed          # PEP 561 marker
│       └── data/
│           └── corrections.toml   # bundled corrections dictionary
├── tests/
│   ├── conftest.py           # fake_whisper fixture, in-process http_server fixture
│   ├── test_clean.py
│   ├── test_cli.py
│   ├── test_download.py
│   └── test_transcribe.py
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
podcast-transcript clean <transcript.txt> [--reflow]

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

Four composable passes — each is a pure function, easy to read, easy to
test, free, and deterministic:

1. **Loop collapser** — `difflib.SequenceMatcher` ratio against a run-leader;
   collapse runs of `min_run` (default 3) similar adjacent lines.
2. **Outro stripper** — find the *last* well-formed English line (pure ASCII,
   sentence-final punctuation OR ≥30 chars) and drop everything after it.
   Walking from the end gives up too early when a fragment like `"you"` sits
   between real content and script-mismatch outro junk.
3. **Corrections** — word-bounded regex substitutions from a TOML file.
   Defaults ship at `data/corrections.toml`; users extend with
   `--corrections my.toml`. Bundled via hatchling and loaded via
   `importlib.resources`.
4. **Paragraph reflow** (opt-in) — collapse Whisper's per-segment lines into
   prose paragraphs of N sentences each.

If you reach for an LLM-based polish pass later, add it as an *optional*
extra (mirror the `whisper` extra pattern) so the rule-based pipeline stays
zero-dep.

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

- README, CLAUDE.md, and `pyproject.toml` should stay in sync on user-visible
  changes (commands, output paths, supported models, deps).
- Don't introduce a step that requires network access during transcription —
  the design assumes audio is already downloaded.
- New tests go under `tests/`. Use the existing `fake_whisper` and
  `http_server` fixtures rather than reaching for the real network or torch.
- New CLI subcommands: add the parser in `cli.py`, a `_run_<name>` handler,
  and a corresponding test in `tests/test_cli.py`.

## Known not-done items

- No `LICENSE` file (repo is private).
- No release/publishing pipeline. The package isn't on PyPI; install is local
  via `pip install -e`.
