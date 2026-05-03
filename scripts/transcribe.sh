#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <audio-file> [model]"
  exit 1
fi

AUDIO_FILE="$1"
MODEL="${2:-large-v3}"

if [ ! -f "$AUDIO_FILE" ]; then
  echo "Audio file not found: $AUDIO_FILE"
  exit 1
fi

mkdir -p transcripts
source venv/bin/activate
whisper "$AUDIO_FILE" \
  --model "$MODEL" \
  --language en \
  --output_format all \
  --output_dir ./transcripts

echo "Transcript outputs written to ./transcripts"
