#!/bin/bash
# eval8-chrome.sh — M2b: run the dg_eval 8-prompt suite through the CHROME engine.
# For each prompt: capture a hole-free trace (native, optimal traceable config),
# validate completeness, run the engine in headless Chrome, keyword-check the text,
# then delete the trace (dumps are ~21GB each; disk can't hold 8).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HESPER="$HOME/git/verilean/hesper"
MODEL="$HESPER/diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
BIN="$HESPER/.lake/build/bin/diffusiongemma-decode"
TDIR="/tmp/claude-503/dgtrace-eval"
CFG="DG_NOMSL=1 DG_NOMSLDOWN=1 DG_Q6KWARP=1"

PROMPTS=(${EVAL_PROMPTS_OVERRIDE:+dummy})
EXPECTS=()
if [ -n "${EVAL_SUBSET:-}" ]; then
  PROMPTS=("The opposite of hot is" "The first man on the moon was")
  EXPECTS=("[Cc]old" "[Aa]rmstrong")
else
  PROMPTS=(
    "The capital of France is"
    "The largest planet in our solar system is"
    "Water is made of"
    "The author of Romeo and Juliet is"
    "2+2="
    "The chemical symbol for gold is"
    "The opposite of hot is"
    "The first man on the moon was"
  )
  EXPECTS=(
    "[Pp]aris"
    "[Jj]upiter|gas giant"
    "[Hh]ydrogen|H2O|[Oo]xygen"
    "[Ss]hakespeare"
    "4|[Ff]our"
    "Au"
    "[Cc]old"
    "[Aa]rmstrong"
  )
fi

score=0
echo "=== eval8-chrome  CFG='$CFG' ==="
for i in "${!PROMPTS[@]}"; do
  p="${PROMPTS[$i]}"; ex="${EXPECTS[$i]}"
  rm -rf "$TDIR"; mkdir -p "$TDIR"
  pkill -f "user-data-dir=$HOME/.cache/e4b-chrome-a4b" 2>/dev/null; sleep 2   # free the previous engine's 20GB GPU before the native capture
  env DG_TRACE_JS="$TDIR" DG_TRACE_JS_DUMP=1 $CFG \
    timeout 700 "$BIN" "$MODEL" "$p" > "$TDIR/native.log" 2>&1
  ntext=$(awk '/TEXT\(raw/{found=1; sub(/^.*TEXT[^:]*: /,""); print; next} found{print}' "$TDIR/native.log" | tr '\n' ' ')
  holes=$(python3 "$ROOT/scripts/dgtrace-validate.py" "$TDIR" 2>/dev/null | python3 -c "
import sys, re
static = {'weights','wnorm','rw','rscale','embedding_table','scale','bias'}
dyn = {'token_ids','uin','params','scTok','scProb','scT','tok','tbuf'}
n = 0
for l in sys.stdin:
    m = re.search(r\"first as '(\w+)'\", l)
    if m and m.group(1) not in static and m.group(1) not in dyn: n += 1
print(n)")
  cp /tmp/claude-503/dgtrace/vocab.json "$TDIR/vocab.json" 2>/dev/null || cp /tmp/claude-503/dgtrace4/vocab.json "$TDIR/vocab.json"
  ln -sfn "$TDIR" "$ROOT/dgtrace"
  "$ROOT/harness/engine-dg.sh" 900 > /dev/null 2>&1
  etext=$(awk '/ENGINE (PASS|FAIL)/{found=1; sub(/^.*text: /,""); print; next} found{print}' "$ROOT/harness/run.log" | tr '\n' ' ')
  stepms=$(grep -oE "ENGINE (PASS|FAIL): [0-9]+ steps, [0-9]+ms/step" "$ROOT/harness/run.log" | grep -oE "[0-9]+ms" | grep -oE "[0-9]+")
  if echo "$etext" | grep -qE "$ex"; then v="PASS"; score=$((score+1)); else v="FAIL"; fi
  nv=$(echo "$ntext" | grep -qE "$ex" && echo P || echo F)
  printf "%s [native:%s, %sms/step] %-42s → %.90s\n" "$v" "$nv" "${stepms:-?}" "\"$p\"" "$etext"
done
rm -rf "$TDIR"
ln -sfn /tmp/claude-503/dgtrace4 "$ROOT/dgtrace"
echo "=== CHROME TOTAL: $score/${#PROMPTS[@]} ==="
