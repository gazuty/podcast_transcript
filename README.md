# podcast_transcript

Local-only podcast download and transcription for Apple Silicon Macs, using
[OpenAI Whisper](https://github.com/openai/whisper).

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
```

Outputs are written to `./transcripts/` in five formats: `.txt`, `.srt`,
`.vtt`, `.tsv`, `.json`.

## CLI reference

```text
podcast-transcript download URL STEM [--output-dir DIR] [--timeout SECONDS]
podcast-transcript transcribe AUDIO_FILE [--model MODEL] [--language LANG] [--output-dir DIR]
```

Defaults: `--model large-v3`, `--language en`, `--output-dir transcripts`.

For autodetect language, pass `--language ""`. For a faster (less accurate)
model, pass `--model turbo`.

## How it works

- `download_podcast` streams the response to `<file>.part` with `Content-Type`
  validation and atomic rename, so you never end up with an HTML error page
  saved as `.mp3`.
- `transcribe_audio` lazy-imports `whisper`, loads the requested model, and
  writes all five output formats via `whisper.utils.get_writer`.

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
│       ├── cli.py            # argparse entry point
│       ├── download.py       # stdlib-only HTTP download with validation
│       └── transcribe.py     # Lazy-imported whisper wrapper
├── tests/                    # pytest unit tests (mock whisper, in-process HTTP server)
├── scripts/
│   └── setup.sh              # brew + venv bootstrap
└── .github/workflows/ci.yml  # lint, type-check, test, shellcheck
```
