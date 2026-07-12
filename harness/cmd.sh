#!/bin/bash
# Send a command to the RESIDENT tab (weights stay on GPU) and tail the log.
# Usage: ./harness/cmd.sh '{"mode":"gate","n":24}' [timeout_s]
set -e
CMD="$1"; TIMEOUT="${2:-300}"
LOG="$(dirname "$0")/run.log"
: > "$LOG"
curl -s -XPOST --data "$CMD" http://127.0.0.1:8877/cmd > /dev/null
for ((i=0; i<TIMEOUT; i++)); do
  if grep -qE "^(DONE|ERROR)" "$LOG" 2>/dev/null; then break; fi
  sleep 1
done
cat "$LOG"
