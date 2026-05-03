#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <mp3-url> <output-stem>"
  exit 1
fi

URL="$1"
STEM="$2"

curl -L "$URL" -o "${STEM}.mp3"

echo "Downloaded to ${STEM}.mp3"
