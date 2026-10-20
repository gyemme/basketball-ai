#!/usr/bin/env bash
# One-time setup on a Mac:  ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v ffmpeg >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    echo "Installing ffmpeg with Homebrew ..."
    brew install ffmpeg
  else
    echo "ffmpeg is required. Install Homebrew (https://brew.sh) then: brew install ffmpeg"
    exit 1
  fi
fi

PY=${PYTHON:-python3}
"$PY" -c 'import sys; assert sys.version_info >= (3, 9), "Python 3.9+ needed"'
if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip wheel >/dev/null
EXTRAS="yolo,dev"
if [ "$(uname)" = "Darwin" ]; then EXTRAS="$EXTRAS,mac"; else EXTRAS="$EXTRAS,ocr"; fi
pip install -e ".[${EXTRAS}]"

echo
echo "Done. Activate the environment in each new terminal with:"
echo "  source .venv/bin/activate"
echo "Then try:"
echo "  hoopstats score \"videos/18:00:21.MP4\""
echo "  hoopstats analyze \"videos/18:00:21.MP4\" --open"
