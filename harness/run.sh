#!/bin/bash
# One-shot: run PAGE (default index.html) in headless Chrome against the dev server,
# tail harness/run.log until DONE/ERROR. The server must already be running
# (python3 harness/serve.py &). Usage: harness/run.sh [page] [timeout_s]
set -u
PAGE="${1:-index.html}"
TIMEOUT="${2:-300}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PROFILE="${E4B_CHROME_PROFILE:-$HOME/.cache/e4b-chrome-profile}"
: > "$ROOT/harness/run.log"
"$CHROME" --headless=new --user-data-dir="$PROFILE" --no-first-run \
  --enable-unsafe-webgpu --use-angle=metal \
  --disk-cache-size=1 --media-cache-size=1 \
  "http://127.0.0.1:8877/$PAGE" >/dev/null 2>&1 &
CPID=$!
SECS=0
until grep -qE "DONE|ERROR|FATAL" "$ROOT/harness/run.log" 2>/dev/null; do
  sleep 1; SECS=$((SECS+1))
  if [ "$SECS" -ge "$TIMEOUT" ]; then echo "TIMEOUT after ${TIMEOUT}s"; break; fi
done
kill $CPID 2>/dev/null
cat "$ROOT/harness/run.log"
