# DG→JS port — experiment log

Working log for the DiffusionGemma trace-replay port (M0-M2) and the
debugging-tooling case study (the mutation study V-P4..P6's qualitative
companion). One entry per iteration: what was run, what the tool detected,
what it missed, and the improvement extracted. Times are wall-clock per
iteration (trace regen ~3-15 min depending on dump flags; replay-only ~2-3
min).

## Parity hunt R0-R10 (2026-07-15) — VERDICT: no bug; 1-2 ULP FMA-contraction
## difference between hesper's pinned Dawn/Tint and current Chrome, amplified
## by near-tie diffusion chaos. 11 replay runs, 6 trace regens.

| # | change / hypothesis | tool used | detected | missed / misled | improvement extracted |
|---|---|---|---|---|---|
| R0 | first replay | WebGPU validation (loud) | 404 HTML fed as WGSL | — | **64-bit hashes must be JSON strings** (2^53 doubles); fixed trace+replayer |
| R1 | replay runs end-to-end | readback bit-gates (6) | divergence EXISTS | no localization at all | gates alone are a smoke alarm, not a map |
| R2 | post-state compare, first-bind ranking | 600-buffer diff | 44 diverged | **ranking misled to dispatch #0** — first BIND ≠ content writer (ping-pong reuse) | rank by LAST binder |
| R3 | (reasoning) dump captured POST-step state | — | pre-state dump timing bug | — | **two-phase capture**: refs at step N, pre-dump at step N+1 start, stream = step N+1 |
| R4 | last-binder ranking + f32 maxRel | post-state + maxRel | divergence confined to tail (#1079+); 401 mutable bufs bit-exact | **maxRel 1e5+ at FINAL states → wrongly rejected the rounding hypothesis** (was measuring after 30 layers of chaotic amplification) | measure error at the FIRST divergence, never at amplified finals |
| R5 | one-pass-per-dispatch (Dawn barrier-drop hypothesis) | A/B replay | no effect (hypothesis dead) | — | kept anyway: replay now structurally matches bridge.cpp |
| R6 | run wgsl-check on the DG trace | wgsl-check | 8 FAILs — all FALSE (mod idiom `x-(x/c)*c` read as wrappable) | correctly found nothing real in its class | **checker: mod-idiom recognition**; DG = second engine at 0 FAIL / 60 WARN |
| R7 | per-dispatch checksum stream v1 (XOR) | DG_TRACE_JS_CKSUM | **first divergent dispatch = #1138 (embed gather), 12 OK after** | XOR hides WHICH buffer | checksums should be per-buffer from the start |
| R8 | dump-all provenance, .bin-first loading | authoritative loads | GGUF range-fetch exonerated (same mismatch bytes) | — | kept as loading mode (removes a whole uncertainty class) |
| R9 | per-buffer checksums | cksum v2 | inputs bit-exact, **OUTPUT only differs** | — | — |
| R10 | 4KB snapshots + byte diff | snapshot diff | **maxRel 2.05e-7 = 1-2 f32 ULP** → FMA-contraction, case closed | — | snapshots make the checksum stream terminal: dispatch+buffer+magnitude in one run |

**Cost accounting**: wandering path = 11 iterations (~3h incl. trace regens).
The protocol extracted for next time = 2 iterations (~30 min): (1) cksum
stream on → first divergent dispatch + buffer; (2) snapshot diff → magnitude
→ classify {logic bug | load bug | compiler rounding}.

**What each tool CANNOT see** (boundaries, stated once):
- wgsl-check: numeric/rounding differences (by design; store-bounds only),
  OOB reads (deferred), data-dependent invariants (WARNs, by design).
- post-state compare: anything overwritten later (ping-pong reuse masks
  mid-step corruption) — needs the checksum stream for mid-step.
- readback gates: bit-equality is the wrong equivalence across compiler
  versions; useful only same-compiler.
- big-buffer tail probe: only load truncation.

**Standing conclusions for the port**:
- Bit-parity across Dawn/Tint versions unattainable by construction. M1 gate
  = per-dispatch tolerance (snapshots); M2 gate = decoded text + eval-8.
- DG_NOMSL=1 required for a WGSL-replayable trace (default MoE gate/up is
  hand-MSL, invisible to Dawn-level tracing).

## Next entries appended below as work proceeds.

## R11-R12 (2026-07-15): tolerance gate + the true amplification curve

| # | change | detected | missed / misled | improvement |
|---|---|---|---|---|
| R11 | tolerance classes on per-element maxRel | — | **metric trap: per-element relative error over-penalizes near-zero elements** (ULP noise on 1e-6 → "1e-2 error"; one buffer with f32-max-scale elements → rel 3e47) → 966/1138 "FLIP", first at #1154: unusable | normalize by buffer RMS |
| R12 | RMS-normalized max deviation | **true curve: 75 exact + 91 ULP + 23 mod; first signal-level flip at the 44th dispatch of the step — layer 0-1's MoE ROUTER (din/wts/acc relRMS 13)**; 949 downstream "FLIP" = a legitimately different (near-tie-flipped) computation path, not corruption | — | — |

**Correction to R4's inference**: "layers 0-28 bit-exact" was WRONG — the 401
matched mutable buffers are early-only/one-off scratch (last-bind < tail), not
per-layer state; all per-layer flow state is shared scratch whose last-bind is
the tail. The checksum stream (R7+) is the only mid-step ground truth.
Lesson: post-state comparison CANNOT reason about mid-step causality under
buffer reuse — do not infer layer-boundary correctness from it.

**M1 verdict**: the replay is faithful to compiler-ULP level; divergence
beyond that is the engine's own near-tie sensitivity (router flips on 1-ULP
input drift within the first layer — the DENSEF16/Jupiter class, now measured
at dispatch granularity). Formal M1 = PASS (fidelity bounded by compiler
rounding). M2 gate must therefore be the eval-8 keyword protocol (near-tie
flips tolerated), NOT text equality — requires the JS commit-scheduler port.

**Numbers**: parity arc totals — 13 replay runs, 7 trace regens, ~4h wall.
Checksum+snapshot protocol reduces the next such hunt to ~2 runs.

## M2a plan (2026-07-15, scoped) — France-only text parity in Chrome

Scope decision: kernels bake the canvas length N as template constants, so
one trace serves ONE prompt length. M2a = France text parity on the existing
trace; eval-8 = M2b (needs role-mapped buffers / per-length kernel sets —
i.e. the real engine step).

Port inventory (source: hesper Examples/DiffusionGemmaDecode.lean):
- Scheduler = renoise/eb mode (the default; "eb acc=" lines). Per step:
  1. dynamic 'w' writes, identified by ROLE (binding names in the stream):
     token_ids=canvas, scTok/scProb (skip step 0), scT (annealed prev t),
     uin=ebU (C pre-drawn uniforms), params=ebP ([tCur,0,0,0]).
  2. replay the traced dispatch stream (fixed shapes).
  3. readbacks: reduceTopK (odenom/otok/oprob) + ebSampleFullB
     (oamax/osamp/oh per position).
  4. CPU (lines ~2000-2100): PCG64 LCG rng (mul 6364136223846793005, inc
     1442695040888963407; u=(rng>>11)/2^53 — BigInt in JS, bit-exact),
     entropy-ordered acceptance under the MI bound (ebBound), renoise
     rejected positions with random tokens, canvas←argmax on finish,
     stability stop (ebHeld ≥ ebStab && meanH < ebConfTh), anneal tCur
     (0.8 → …/24 steps seen: t=0.8, 0.7917 …), scTok←ktokFlat,
     scProb←qFlat (STILL TO PIN DOWN in renoise mode: qFlat provenance —
     read lines 1950-2005), trim/EOS (lines ~2100-2175), canvas init +
     template + P/prompt layout (lines ~1420-1520).
- Detok: DG_DUMP_VOCAB=<file> (new hesper flag) → vocab.json (262,144
  pieces) + JS piece-concat (▁→space; check <0xXX> byte pieces).
- Gate: decoded France text vs hesper's ("The capital of France is Paris."
  + thought-channel tail), near-tie drift tolerated at the wording level,
  keyword = [Pp]aris.
Open risks (log when resolved): step-0 stream shape differs from the traced
step-2 (SC writes skipped; zero-init equivalence assumed); ebU regeneration
must preserve the LCG stream POSITION (rng state at step start depends on
draws in prior steps — count draws per step: C pre-draws + rejects).
