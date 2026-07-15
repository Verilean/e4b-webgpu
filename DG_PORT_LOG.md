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

## R13-R17 (2026-07-15): M2a scheduler port + the step-0 confidence gap

hesper golden: step 0 = acc=141 / meanH=0.636 / oh[0]=0.037; engine: acc=1 /
meanH=4.09 / oh[0]=1.09 — same top-1 token, top-1 prob 0.9999→0.595:
a systematic ~20× logit-confidence muting on Chrome.

| # | change / probe | detected | missed / misled | improvement |
|---|---|---|---|---|
| R13 | zero SC buffers at init | no effect (bit-identical meanH) | assumed SC state was read at step 0 — it is NOT (SC ops absent from the step-0 stream) | check the stream before theorizing about state |
| R14 | 3-step marker-sliced trace (step-0 stream ≠ step-2 stream: SC only runs step>0) | structural fix, correct in itself | only 4.29→4.09 — not the main cause | streams per step-shape now standard in the trace |
| R15 | recorded step-0 hex vs my LCG | **canvas + uArr BIT-IDENTICAL** (BigInt LCG port exact) | — | recorded w-hex doubles as an input golden |
| R16 | step-0 readback goldens (r-hex in trace) | muting is GPU-side, systematic, deterministic across runs (not a race) | — | r-hex = free per-step gate |
| R17 | per-dispatch drift curve (relRMS, first 60) | smooth f16-accumulation growth 1e-7→1e-3 (#0-30, WMMA class), then ONE explosion #32 MoE down+acc (1.5e-2→13) | float-RMS metric is blind to INTEGER buffers (idxs) — routing flips invisible | metric TODO: integer-aware compare for idx buffers |

Open hypotheses (bisect running): (A) cross-compiler f16-WMMA accumulation
drift (legit per spec) flips router near-ties EVERY layer → 30 layers of
semi-random expert paths → muted logits; (B) a deterministic semantic
difference on Chrome in the grouped-MoE chain (counting-sort/scatter — the
same stores wgsl-check flags as data-dependent WARNs). Discriminator in
flight: DG_NOMOERB=1 DG_NOQKVRB=1 trace (no subgroup-matrix, per-slot MoE)
replayed on Chrome — if step-0 goldens then match ≈exactly, the reg/grouped
class is the culprit; if still muted, suspicion moves to dp4a/warp/atomics.

## R18 (2026-07-15): the router pinpointed — cancellation-amplified FMA drift

dp4a-path trace (DG_NOMOERB=1 DG_NOQKVRB=1; hesper still decodes fine:
"Capital: Paris.", acc=122): the Chrome replay curve is PURE ULP (1e-6/-7)
through attention/dense/quantize — confirming the reg-path's 1e-4..1e-3
floor was f16-WMMA accumulation order. Then ONE kernel jumps ×1000:
**#30 router GEMV (rw·tmps → rlogits): 1.9e-7 in → 2.5e-3 out.**
Kernel text = plain sequential f32 dot (K=2816), no subgroups, no
transcendentals ⇒ the only cross-compiler difference is FMA contraction;
router logits are heavily-cancelling sums, so ~1e-6 absolute per-term drift
becomes ~1e-3 RELATIVE on the result. Downstream: top-8 near-tie flips
(#31 wts 3.1e-3, #38 gather 5.5e-2 — integer idxs flips invisible to the
float metric, noted in R17) → MoE mixture differs → decoherence.

OPEN CONTRADICTION (next discriminator): hesper tolerates KERNEL SWAPS
(dp4a↔reg, f16 dense) with only near-tie wording changes — so "a few expert
flips" should not GLOBALLY mute confidence (engine oh mean 4.09 vs golden
0.64 across ~all positions). Either the flips compound differently across
30 layers than kernel swaps do, or something else still lurks.
**Queued experiment**: swap the engine's router dispatch for a
Kahan-summation router kernel (Chrome-side only). If step-0 oh ≈ golden
(0.037-class) → cancellation-drift is THE lever and the engine ships a
compensated router by default; if not → hunt continues with per-position
H distributions (mean/max metrics hide the shape).

## R19 (2026-07-15): THE UNLOCK — initial state must be PRE-STEP-0

Mid-window checksums (head-only windows had validated ONLY position 0's row
— R17's metric note, now paid in full) first "showed" garbage (relRMS 5e9) at
the FIRST dp4a matmul — but its CONSUMER was clean: the "garbage" was the
window landing in never-read tails of REUSED scratch buffers, where the
engine's pre-step-2 dumps carry step-1 leftovers that hesper's step 0 never
had. Same class as R13's lesson, one level deeper: **partial-state dumps
poison every window comparison and (via truly-read stale state) the step-0
forward itself.**

Fix: `jsTraceDumpAllRegistry` — dump ALL ~950 registry buffers (24GB) at
step-0 START (no ref-set needed) = the true pre-step-0 state; engine loads
these as its initial state.

RESULT: the trajectory CONVERGES for the first time — step 1 meanH 0.41
(hesper-class 0.35-0.38; was 3.9), descending to 0.06-0.15 by step 35-47
with healthy accept counts. Step-0 readbacks still differ from golden
(cross-compiler near-tie routing, expected). Text extraction next (argmax[0]
is eos-ish → detok must skip channel-marker eos, not break).

Iteration count this arc so far: 8 engine runs, 4 trace regens.

## R20 (2026-07-15): seed 2×2 — drift theory REJECTED, defect is systematic

| | seed 12345 | seed 12346 |
|---|---|---|
| hesper (dp4a) | "Paris" ✓ | "Paris" ✓ |
| Chrome engine | eos-collapse ("Spectimp{{") | eos-collapse ("cul吐{") |

Chrome falls into the empty-answer attractor on BOTH seeds (canvas → eos at
position 0 + eos-fill) while hesper is seed-robust. KILLER ARGUMENT against
the numeric-drift story: hesper's own dp4a↔reg kernel swap is a FAR larger
numeric change than cross-compiler FMA drift, yet preserves behavior
("Capital: Paris" vs "The capital ... Paris."). Therefore Chrome computes
something DIFFERENTLY IN KIND, not just in rounding. Post-R19 trajectory
health (steps ≥1 meanH matches hesper-class, healthy convergence) says the
DYNAMICS work; the model is systematically un-confident about the ANSWER
HEAD (position 0 → eos = "the turn is already over" — misframing signature).
Unexamined suspects: the full-vocab scan kernels (reduceTopKB /
ebSampleFullB — they PRODUCE the readbacks; likely subgroup-reduction
patterns) and the post-R19 trustworthy drift curve (in flight).

## R21 (2026-07-15): fastMath hypothesis tested and KILLED (cleanly)

Mechanism found in Dawn source: hesper's pinned Dawn compiles MSL with
fastMathEnabled = !strictMath (default FAST); newer Dawn/Chrome moved to
strict. Implemented HESPER_STRICT_MATH=1 (bridge: ShaderModuleCompilationOptions
chain + ShaderModuleCompilationOptions device feature).

TRAP LOGGED: first run showed bit-identical results — the native bridge was
CACHED (lakefile only rebuilds if libhesper_native.dylib is missing); the
edit never compiled. `rm .lake/build/native/libhesper_native.dylib` forces it.
A bit-identical "no effect" result should always raise the did-my-change-
even-run question first.

RESULT (real): hesper-strict step0 acc=140/meanH=0.661 (vs fast 122/0.634 —
flag ACTIVE, near-tie-level shift) and still decodes "The capital of France
is Paris." Chrome's divergence (meanH 2.09, eos-collapse) is a DIFFERENT
KIND of difference, not math mode. Strongest remaining suspect: real
codegen difference in Chrome's newer Tint for some kernel class (first
mid-row jump: dp4a MMQ matmul #15, 2.5e-3 at mid rows / ULP at row 0,
deterministic, inputs clean).

NEXT DECISIVE EXPERIMENT: launch Chrome with
--enable-dawn-features=dump_shaders, capture its MSL for the #15 kernel,
and TEXT-DIFF against hesper's HESPER_DUMP_MSL for the same WGSL — makes
the codegen difference directly visible. (Both dumps exist as tooling
already; no new infrastructure needed.)

## R22 (2026-07-15, in progress): MSL text-diff — materials secured

- hesper MSL: HESPER_DUMP_MSL=1 run → 34MB stderr; block structure
  `// Dumped WGSL:` … `/* Dumped generated MSL */` …; kernel #15 extracted by
  WGSL signature (1622016+319104+252645135) → **debug/k15/hesper-k15.msl
  (74.5KB, full)**.
- Chrome MSL: `--enable-dawn-features=dump_shaders,disable_symbol_renaming
  --enable-logging=stderr` works, BUT the dump rides console messages and
  Chrome TRUNCATES long console lines → only 2.9KB fragment
  (debug/k15/chrome-k15-TRUNCATED.msl). Console-based dumping cannot yield
  the full kernel.
- NEXT: build the tint CLI at a recent (Chrome-era) Dawn revision and
  offline-compile debug/k15/k15.wgsl; diff against hesper-k15.msl (hesper's
  own tint CLI at /tmp/tint-build/tint is the PINNED version — rebuild
  needed at newer rev). The k15 WGSL is checked in for reproducibility.

## R23 (2026-07-15): tint CLI cross-version diff of k15 — NO semantic delta

Built tint CLI at Dawn main (July 2026, /tmp/tint-new) vs hesper's pinned
CLI (May 2026 snapshot — dawn.tar.gz date; the dawn-src git commit date is
a local-init artifact, do not trust it). Compiled debug/k15/k15.wgsl with
both (debug/k15/k15-tint-{old,new}.msl):
- differences: threadgroup-read robustness clamps ELIDED in new (indices
  provably in-bounds — verified max 1166 < 2336, benign), dot4I8 polyfill
  refactored into a temp (identical math), statement scheduling/renaming.
- **No difference that changes in-bounds numeric semantics.** The CLI-level
  codegen delta does NOT explain the 2.5e-3 mid-row divergence.

Remaining explanation space: RUNTIME compilation differences (Dawn pipeline
path adds robustness/size-UBO transforms — hesper's runtime MSL is 74.5KB vs
25KB CLI output; Chrome's runtime MSL unobtainable via console) or an
execution-environment difference (Metal PSO options, denormal handling at
mid-row DATA values — note strict-math was ruled out on hesper, but
Chrome's Metal compile options are not fully known).

NEXT (fast-iteration path): micro-repro — the engine loads kernels from
files, so EDIT debug/k15-class WGSL in the trace dir and replay (~3 min per
variant) to bisect which construct triggers the divergence (candidates:
unpack2x16float f16-scale path / denormal-adjacent data at mid rows / the
y_block staging). Alternatively build Dawn-native at July rev and swap it
under hesper to get an apples-to-apples runtime MSL diff.

## R24 (2026-07-15): tile-permutation + strict-trace — TWO more eliminations

- **Tile permutation** (wid.y → (y+4)%9, semantics-preserving): #15 relRMS
  EXACTLY 2.5e-3 unchanged → divergence is DATA-LOCKED (follows the rows'
  values, not workgroup index/scheduling). Kills scheduling/UB theories for
  this kernel.
- **hesper-STRICT trace replayed on Chrome**: #15 still 4.8e-3 → Chrome
  matches NEITHER hesper-fast NOR hesper-strict. fastMath is fully dead as
  the explanation (both directions now tested).

Remaining suspects (ranked):
1. **Tint MSL backend generation path**: Tint is migrating MSL output
   AST→IR; Chrome's runtime and the CLI may use different generators — the
   CLI-vs-CLI diff (R23) compared the same default path and would miss it.
   Probe: tint CLI has a use-ir/backend flag? or diff Dawn runtime code for
   which path Chrome enables.
2. Data-shaped numeric path (cancellation/denormal at mid-row values) that
   both hesper modes treat one way and Chrome's compile treats another.
   Probe: element-level histogram of the #15 mid-window diff (indices,
   got/want values — are the bad elements tiny/cancelled?) — snapshot
   c15_2.bin already on disk; needs one replay-only run with a dump tweak.
3. Micro-repro bisection of the kernel body (scale-path vs dp4a-path), with
   hesper-side re-goldens per semantics-changing variant (~8 min each).

Iteration count total: 15 engine/replay runs today on this hunt; 9 trace
regens; 11 hypotheses eliminated with evidence.

## R25 (2026-07-15): read-checker ships; k15 fully in-bounds; final hypothesis

Implemented OOB-READ detection in wgsl-check (v1.1: reads clamp on old Tint
vs predicate-to-zero on newer robustness = silent cross-compiler VALUE
divergence — the deferred read class, now live; fixtures still 3 FAIL).
DG manifest re-check: **k15 = ok (stores AND reads in-bounds)** → the
clamp-vs-predication OOB theory is dead for k15.

SURVIVING HYPOTHESIS (consistent with every observation): both hesper and
Chrome compile fastMath, but the newer Tint backend emits structurally
different (semantically equal) MSL whose temporaries change the METAL
compiler's FMA-fusion choices; on cancellation-heavy accumulations this
yields data-locked ~1e-3 deviations. Fusion is invisible in MSL text
(explains R23's null diff), data-locked (R24), and differs from BOTH hesper
modes (strict=unfused ≠ fast-fused-A ≠ fast-fused-B).

Consequence if true: kernel-level value parity across Tint versions is
IRREDUCIBLE; M2a must make the DECODE robust instead:
(a) more seeds (n=2 may be luck) — pending; (b) schedule robustness for
lower step-0 confidence (ebBound/anneal tuning on the Chrome side);
(c) cancellation hardening (Kahan) in the ROUTER to stabilize top-8
near-ties against any compiler's fusion choices.

## R26 (2026-07-16): seed sweep n=3 — SYSTEMATIC confirmed

Seeds 12345/12346/12349 on Chrome ALL collapse to the empty-answer attractor
(12347/48 lost to harness timeouts — TWICE undersized (1500s, 1800s) against
the strict-trace's real ~35 min/run; the cksum-mode stream is ~4500
events/step and even with 'c' skipped + submit batching runs ~35-40s/step.
TRAP (repeat): estimate run time from MEASURED per-step wall, not hope).
Also: a first sweep attempt WEDGED Chrome ~85 min in step-0's per-dispatch
readback probes → all debug probes now gated behind opts.debug (default off).

Verdict: the Chrome trajectory is systematically degenerate, not unlucky.
Two forks remain:
(A) the per-layer Lyapunov amplification (measured: 2.5e-3 @#15 → 2e-2 @#16
    → 1e-1 @#19 → O(1) by layer 2) makes ANY numeric difference a different
    trajectory, AND Chrome's difference is BIASED (e.g. magnitude loss)
    pushing toward low-confidence/eos — explains hesper's kernel-swap
    tolerance (unbiased swaps) vs Chrome (biased difference).
(B) a shared engine-dynamics bug all seeds hit (scheduler port subtlety).
DISCRIMINATOR queued: bias statistics of the #15 deviation — dump Chrome's
mid-window bytes for #15, compare vs c-snapshot: mean(got−want) vs
mean|got−want| (bias ratio), and |diff| correlation with |want|
(FTZ/cancellation signature). All reference data already on disk.

## R27 (2026-07-16): the amplifier identified — Q8 quantization boundary flips

Bias probe at #15 (strict trace):
- output deviation: meanW=2.67, meanAbsDiff=2.85e-3, **biasRatio=0.025 ≈
  SYMMETRIC** (not a systematic magnitude shift).
- **input_q8 itself differs at mid rows**: low-bit u32 deltas = ±1 int8
  QUANTIZATION CODES. Mechanism: ULP-level activation differences land on
  round-to-nearest boundaries → ±1 codes (~8e-3 relative per element) →
  **the ×1000 amplifier is the Q8 quantizer, not matmul FMA**; every layer
  re-quantizes → measured Lyapunov chain (2.5e-3 → 2e-2 → 1e-1 → O(1) by
  layer 2). This is inherent to int-quantized inference across ANY numeric
  difference (llama.cpp cross-backend has the same property).
Also verified: scProb IS all-zeros in hesper renoise (full 8192B checked —
earlier only 12 head bytes had been inspected; engine port correct);
ebSampleFullB and reduceTopKB contain NO subgroup ops (width-assumption
theory dead).

REMAINING PUZZLE (one number): step-0 H DISTRIBUTION — hesper variants
(fast/strict/dp4a/reg) all keep acc≈120-140 & meanH≈0.63-0.66 while Chrome
collapses to acc=1 & meanH≈2.1, despite the same decorrelation amplifier
acting on all of them. QUEUED DISCRIMINATOR: element-wise oh(hesper-fast)
vs oh(hesper-strict) from the two traces' r-hex — if hesper variants'
element-wise H values are CLOSE, native decorrelation is somehow bounded
and Chrome's is anomalous (a real Chrome-side defect remains); if far-but-
low-mean, distribution shape survives decorrelation natively and Chrome's
mean inflation is the anomaly to hunt.

## R28 (2026-07-16): H-distribution discriminator + quantizer delta anatomy

- **oh(hesper-fast) vs oh(hesper-strict), element-wise (r-hex from the two
  traces): corr = 0.9944, median |ΔH| = 0.0014** — native variants are
  nearly element-wise identical at step 0 despite full fusion differences.
- **Metric blindness, third strike**: the "clean 2.5e-11" at #14 (quantizer)
  was a MISMATCHED checksum whose u32 content, float-interpreted, hid the
  difference. TRUE first divergence = #14 (quantizer output codes).
- **Int8 delta anatomy of #14 (Chrome vs hesper window)**: 111/4096 bytes
  (~2.7%), histogram dominated by ±1 (62), few larger (likely scale bytes).
  The quantizer is faithful — it amplifies upstream f32 FMA noise into
  boundary flips as designed.
Mechanism chain now: Chrome-fusion f32 deltas → ±1 Q8 flips (~2-3%) →
1e-3 matmul deltas → per-layer compounding → decorrelated trajectory.
OPEN (one number): flip rate for NATIVE fast-vs-strict. If also ~2-3%,
flips do NOT decorrelate H natively → Chrome H inflation needs another
cause; if ≪, Chrome's f32 deltas are anomalously large → hunt #13's f32
delta magnitude. Needs one fast+CKSUM trace regen (~15 min) for fast-c14.

## R29 (2026-07-16): BASELINE CONTAMINATION discovered

Native fast-vs-strict quantizer codes: **111/4096 (2.71%), histogram
(-1:43, 1:17, -2:7, 2:6, -4:6, -6:3) — nearly BYTE-IDENTICAL to the
"Chrome-vs-hesper" measurement (111/4096, -1:44, 1:18, …)**. Since
(Chrome vs strict) ≈ (fast vs strict), CHROME ≈ HESPER-FAST at the layer-0
quantizer. ⇒ Every measurement since R24 that used the STRICT trace as
reference (bias probe, #15 4.8e-3, q14 dump) was dominated by the
fast-vs-strict delta, NOT by a Chrome anomaly.

**LESSON (hygiene): when regenerating a reference trace with a changed
config (strict), every downstream comparison silently changes its baseline.
Reference traces must be labeled with their config and comparisons must
state their baseline.**

Re-baselining now: fresh FAST trace (dumps+cksums) + Chrome curve → the
TRUE Chrome-vs-fast first divergence. Earlier fast-baseline evidence
(R16-R20: oh 30× off, collapse) still stands — Chrome ≠ fast SOMEWHERE,
but possibly much later/narrower than the recent strict-contaminated
measurements suggested.

## R30 (2026-07-16): THE DECISIVE CURVE — decorrelation is native physics; Chrome's defect is in the H tail-chain

Native fast-vs-strict snapshot curve (offline, c-files, NO Chrome):
#15 5.1e-3 → #16 2.3e-2 → #20 4.8e-1 → #300 6.1 → #600 25 → lm-head region
(#1130-1137) ~1-2. **Identical shape to Chrome's curve** ⇒ full activation
decorrelation between numeric variants is INHERENT (q8-flip Lyapunov), and
NATIVELY HARMLESS: despite it, oh(fast) vs oh(strict) correlate 0.9944 and
both decode Paris. H is a decorrelation-robust functional of the canvas.

⇒ Chrome's 30× H inflation CANNOT be trajectory noise. The defect is a
different-in-kind computation in the TAIL CHAIN that produces H:
final-norm → Q6_K full-vocab lm_head → softcap → logitsCanvas slice-copies
(8×) → ebSampleFullB (barrier-based workgroup reduction over 262144).
Also retract R29's "Chrome≈fast" (histogram similarity ≠ byte identity;
all three variants are pairwise ~2.7% q8-flipped — statistically equidistant).

NEXT (sharp, tooling in place): Chrome-vs-fast tail curve (remove the #20
early stop) → which tail dispatch first shows a K I N D-different deviation
(vs the native ~1-2 decorrelation floor); then read that kernel. Candidates:
softcap (tanh), slice-copy (mod idiom), ebSampleFullB (workgroup reduction,
log/exp over 262k).

## R31 (2026-07-16): tail-chain mapping + native tail floor — the final norm SQUASHES decorrelation

Corrections first: step-0 slice = **1426 dispatches** (not ~1138); R30's "lm-head
region #1130-1137" was WRONG — that's still layer-23 MoE. True tail chain mapped
from ops.jsonl: #1398 final-norm → #1400-1423 = 8× (lm_head slice matmul a,b,c →
softcap logits → slice-copy src,dst) → #1424 reduce → #1425 ebSampleFullB.

Native fast-vs-strict tail (offline c-files): mid-network O(10) decorrelation
**collapses to 7.4e-2 at the final RMSNorm** (#1398), logits ~0.13-0.32, ebSample
outs 0.105. MECHANISM FOUND: normalization turns trajectory decorrelation into
bounded direction noise — THIS is why native H correlates 0.994 and decode is
robust. So the Chrome question sharpens: does Chrome's #1398 also collapse to
~7e-2? If yes but H still inflates 30×, the defect is INSIDE #1400-1425 (softcap/
copy/reduce/ebSample). If Chrome's #1398 stays O(1), the defect is upstream but
must be KIND-different (bias, not noise) to survive normalization.

Also: Chrome-vs-fast #1100-1130 = 7.6-21 vs native 4.0 — same order (no kind
difference mid-network ✓ consistent with R30). #1109-1111 integer-buffer relRMS
e-34 = float-blind metric again (routing idxs — need byte compare, strike 4 risk).
Chrome replay of full step-0 slice completes in minutes (not 40 — earlier fear
was cksum-readback traces; this one streams).
