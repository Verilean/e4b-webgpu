#!/bin/bash
# Run the DG trace replayer in headless Chrome. Kills the A4B resident tab
# first (both engines don't fit in GPU memory together). Usage:
#   ./harness/replay-dg.sh [timeout_s]
set -u
TIMEOUT="${1:-600}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PROFILE="${E4B_CHROME_PROFILE:-$HOME/.cache/e4b-chrome-a4b}"
pkill -f "user-data-dir=$PROFILE" 2>/dev/null; sleep 1
: > "$ROOT/harness/run.log"
nohup "$CHROME" --headless=new --user-data-dir="$PROFILE" --no-first-run \
  --enable-unsafe-webgpu --use-angle=metal \
  --disk-cache-size=1 --media-cache-size=1 \
  "http://127.0.0.1:8877/engine-dg.html" >/dev/null 2>&1 &
SECS=0
until grep -qE "ENGINE (PASS|FAIL|EXCEPTION)" "$ROOT/harness/run.log" 2>/dev/null; do
  sleep 2; SECS=$((SECS+2))
  if [ "$SECS" -ge "$TIMEOUT" ]; then echo "TIMEOUT after ${TIMEOUT}s"; break; fi
done
cat "$ROOT/harness/run.log"
