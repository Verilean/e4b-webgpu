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
