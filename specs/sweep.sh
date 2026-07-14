#!/bin/bash
# TLC sweep for KVRing: establishes the tight completeness bound at small
# sizes, then checks the PRODUCTION constants (W=1024, C=512, RING=1536).
# Usage: bash specs/sweep.sh
set -u
JAVA=/opt/homebrew/opt/openjdk/bin/java
JAR=$HOME/tla2tools.jar
cd "$(dirname "$0")"

run() { # W C RING NCHUNKS GEN -> prints PASS/FAIL(invariant)
  local w=$1 c=$2 r=$3 n=$4 g=$5
  cat > KVRing.cfg <<EOF
CONSTANTS
  W = $w
  C = $c
  RING = $r
  NCHUNKS = $n
  GEN = $g
SPECIFICATION Spec
INVARIANTS SoundRec Complete
EOF
  out=$($JAVA -XX:+UseParallelGC -cp "$JAR" tlc2.TLC -deadlock -workers auto KVRing.tla 2>&1)
  if echo "$out" | grep -q "Model checking completed. No error"; then
    echo "W=$w C=$c RING=$r : PASS"
  else
    viol=$(echo "$out" | grep -o "Invariant [A-Za-z]* is violated" | head -1)
    echo "W=$w C=$c RING=$r : FAIL ($viol)"
  fi
}

echo "== small-instance bound sweep (predicted boundary RING = W+C-1) =="
for wc in "4 3" "6 5" "5 2"; do
  set -- $wc; w=$1; c=$2
  for r in $((w+c-3)) $((w+c-2)) $((w+c-1)) $((w+c)); do
    run "$w" "$c" "$r" 4 6
  done
done

echo "== production constants (W=1024 C=512, engine RING=1536) =="
run 1024 512 1536 3 8   # shipped value        (predicted PASS)
run 1024 512 1535 3 8   # provable minimum     (predicted PASS)
run 1024 512 1534 3 8   # one below            (predicted FAIL)
