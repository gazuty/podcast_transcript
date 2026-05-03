# podcast_transcript

Local-only podcast transcription workflow for Apple Silicon Macs using OpenAI Whisper.

## Purpose

This repository is designed to:
- download podcast audio from a direct MP3 URL
- transcribe it locally on an Apple Silicon Mac
- keep the code and workflow backed up on GitHub

Audio processing is intended to run locally, not in GitHub Actions.

## Requirements

- macOS on Apple Silicon
- Homebrew
- `ffmpeg`
- `python@3.11`

## Quick start

```bash
./scripts/setup.sh
./scripts/download.sh "https://traffic.libsyn.com/secure/unsupervisedlearning/directionalselection_ungated.mp3" razib_directional_selection
./scripts/transcribe.sh razib_directional_selection.mp3
```

Outputs are written to `./transcripts/`.

## Workflow

1. Set up a Python virtual environment
2. Install Whisper
3. Download a podcast MP3
4. Transcribe locally with Whisper `large-v3`
5. Read outputs from `transcripts/`

## Output formats

The transcription script produces:
- `.txt`
- `.srt`
- `.vtt`
- `.tsv`
- `.json`

## Notes

- First run downloads Whisper model weights to `~/.cache/whisper/`
- `large-v3` is slower but better for technical speech
- For faster runs, you can switch to `turbo`
- GitHub Actions in this repo are only for validation and repository hygiene

## Example

```bash
open ./transcripts/razib_directional_selection.txt
```
