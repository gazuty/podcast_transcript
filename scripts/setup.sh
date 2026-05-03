#!/usr/bin/env bash
set -euo pipefail

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required but not installed."
  exit 1
fi

brew install ffmpeg python@3.11

mkdir -p .
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete. Activate later with: source venv/bin/activate"
