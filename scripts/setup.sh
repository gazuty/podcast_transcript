#!/usr/bin/env bash
#
# One-time bootstrap for an Apple Silicon Mac:
#   - install ffmpeg + python@3.11 via Homebrew
#   - create a venv in ./venv
#   - install this package and its optional whisper + dev extras
#
set -euo pipefail

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required but not installed. See https://brew.sh" >&2
  exit 1
fi

brew install ffmpeg python@3.11

PYTHON_BIN="$(brew --prefix python@3.11)/bin/python3.11"

"$PYTHON_BIN" -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate

pip install --upgrade pip
pip install -e ".[whisper,dev]"

echo
echo "Setup complete. Activate later with: source venv/bin/activate"
echo "Then run: podcast-transcript --help"
