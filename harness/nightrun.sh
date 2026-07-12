#!/bin/bash
# Wait for a genuinely quiet box (low load AND idle display server = idle GPU),
# then run the honest measurement protocol and append to harness/night.log.
# Usage: nohup ./harness/nightrun.sh > /dev/null 2>&1 &
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/harness/night.log"
echo "=== nightrun armed $(date) ===" >> "$LOG"
for i in $(seq 1 2880); do          # up to 24h, check every 30s
  LOAD=$(sysctl -n vm.loadavg | awk '{print $2}')
  WS=$(ps aux | awk '/WindowServer/ && !/awk/ {print int($3)}' | head -1)
  if (( $(echo "$LOAD < 0.8" | bc -l) )) && [ "${WS:-100}" -lt 5 ]; then
    echo "=== quiet window found $(date) (load=$LOAD ws=$WS%) ===" >> "$LOG"
    pgrep -f "python3 harness/serve.py" > /dev/null || (cd "$ROOT" && python3 harness/serve.py > harness/serve.log 2>&1 &)
    sleep 2
    "$ROOT/harness/resident.sh" 500 >> "$LOG" 2>&1
    for r in 1 2 3 4; do
      "$ROOT/harness/cmd.sh" '{"mode":"bench","n":64}' 240 2>/dev/null | grep decode >> "$LOG"
    done
    "$ROOT/harness/cmd.sh" '{"mode":"profile"}' 240 2>/dev/null | grep -E "profile tot|ms x" | head -12 >> "$LOG"
    "$ROOT/harness/cmd.sh" '{"mode":"gate","n":16}' 300 2>/dev/null | grep GATE >> "$LOG"
    echo "=== nightrun complete $(date) ===" >> "$LOG"
    exit 0
  fi
  sleep 30
done
echo "=== nightrun: no quiet window in 24h ===" >> "$LOG"
