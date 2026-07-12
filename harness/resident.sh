#!/bin/bash
# Start (or restart) the RESIDENT A4B tab: loads 14.4GB once and then serves
# /cmd requests. Usage: ./harness/resident.sh [timeout_s]   (then use cmd.sh)
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
  "http://127.0.0.1:8877/resident-a4b.html" >/dev/null 2>&1 &
SECS=0
until grep -qE "RESIDENT READY|ERROR" "$ROOT/harness/run.log" 2>/dev/null; do
  sleep 2; SECS=$((SECS+2))
  if [ "$SECS" -ge "$TIMEOUT" ]; then echo "TIMEOUT after ${TIMEOUT}s"; break; fi
done
tail -5 "$ROOT/harness/run.log"
