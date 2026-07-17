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

## R32 (2026-07-16): ROOT CAUSE FOUND — two trace holes (invisible writers), not numerics

Bisect chain that got here: norm outputs (post-RMSNorm) are direction checkpoints
immune to the amplitude-chaos that blinds raw relRMS. Chrome-vs-fast at norms:
fine (≤5e-2 abs) through #28 (attn+dense of layer 0), then #43 (post-MoE norm)
= 22 vs native 2.4e-3 — a 9000× KIND difference. Inside the MoE block: routing
idxs byte-identical, gate/up+geglu fine (1.5e-2), but wacc (#42, din·wts→acc,
a PURE guarded gather-sum — no race possible) explodes 2.5e-2 → 19. Pure kernel
+ wrong output ⇒ corrupt INPUT: din (uid …882688) is READ by 90 dispatches
(30 layers × 3 steps) and **WRITTEN BY NOTHING in the entire ops stream**.

CAUSE 1: hesper's MoE DOWN matmul is hand-MSL gated by DG_NOMSLDOWN — a
SEPARATE flag from DG_NOMSL (which only covers gate/up). Capture used only
DG_NOMSL=1 ⇒ the down ran as MSL, invisible to the Dawn-level JSTrace ⇒
replay/engine consumed the frozen pre-step-0 snapshot of din for every layer,
every step. Explains everything: layer-0 direction break, per-layer compounding,
native fast+strict both correct, systematic across seeds.

CAUSE 2 (same class): "tbuf" (SC temperature, array<f32,4>) has no 'w' event
either (untraced write path); the engine looked for a 4-byte w event and never
substituted it ⇒ SC softmax ran with stale t.

TOOL PAYOFF: wrote scripts/dgtrace-validate.py (read-but-never-written non-weight
buffers, WGSL access-mode aware). On the bad trace it flags EXACTLY {din, tbuf}
after static/dyn filtering. This check is mechanical — run it on every capture
from now on (added to the protocol). LESSON: a trace is a CLAIM about
completeness; validate it before debugging numerics — would have saved R20-R31's
numeric hunt (though that hunt produced the norm-checkpoint method + native-floor
calibration, both reusable).

FIXES: recapture with DG_NOMSL=1 DG_NOMSLDOWN=1 (audited: no other MSL gates
default-on); engine discovers tbuf by binding name and writes [prevT,0,0,0]
per step. NEXT: validate new trace → replay (expect native-floor curve) →
engine France → "[Pp]aris".

## R33 (2026-07-16): M2a PASS — Chrome engine decodes France correctly

With the recaptured hole-free trace (DG_NOMSL=1 DG_NOMSLDOWN=1; dgtrace-validate
= 0 holes, din now written by 90 dispatches) + the engine tbuf fix:

  ENGINE PASS: 6 steps | meanH step0 = 0.5636 (healthy; was 2.09 broken)
  acc 112→189→212→218→233→242, STOP at step 5
  text: <|channel>thought … Capital: Paris … <channel|>The capital of France is Paris.

Native reference on the same config decoded the same structure ("…is Paris.").
The entire R16-R31 "numeric divergence" hunt was chasing a phantom: the engine
was starving 30 layers of MoE-down output. One validator run would have caught
it. Score for the campaign: 33 logged iterations, 2 real bugs (both trace
holes), 4 metric-blindness incidents, 2 timeout losses, 1 baseline
contamination; reusable artifacts: dgtrace-validate.py, norm-checkpoint
direction diffing, native-floor calibration, marker-sliced replay harness.

Perf note: 56.7s/step in Chrome (verbatim per-dispatch replay, no batching/
caching) — correctness first; perf campaign is next (M2b eval-8, then ternary/
schedule/delta-prop toward the 250-400 tok/s targets).

## R34 (2026-07-16): Chrome 57.2s → 2.27s/step (25×) — one kernel class was 96% of the time

Phase timing (added enc/gpu/rb instrumentation + single-compute-pass encoding):
enc 1-4ms, rb 14ms, gpu 56.8s ⇒ pure kernel time, orchestration innocent.
timestamp-query per-dispatch profile (step 1): **13 dispatches × 4.21s = 54.7s of
56.7s** — all one kernel class: block-parallel fusedQ6KBatchKernel (Q6_K bmm,
2816→2048, the 13 Q6_K layers). Pathologies: 11/256 active threads AND a 530KB
un-CSE'd WGSL body (manual-f16-decode expression duplicated per use — the known
ShaderM let-substitution disease). Native Metal absorbs it (~100ms); Chrome's
July Tint does not (~40×).

FIX: DG_Q6KWARP=1 (hesper, flag-gated) — swap to the existing warp-per-row
fusedQ6KBatchF32WarpKernel. Native France: text answer identical ("The capital
of France is Paris."), native itself -0.9s/step (3974→3058ms). Recaptured trace
(0 holes), engine:

  ENGINE PASS: 2267ms/step (was 57156) | gpu 2.1s | text: …Paris. ✓
  Chrome now BEATS native same-config (3058ms) by 26%.

TPS now: canvas 256/13.6s = 18.8 tok/s; useful 45/13.6 ≈ 3.3 tok/s.
New profile is flat (top 195ms n=1, then ≤124ms) — no single villain left.
NEXT (step ②): optimal-config trace — drop DG_NOMOERB/DG_NOQKVRB (reg WMMA +
grouped MoE, all WGSL), keep DG_NOMSL(+DOWN)=1 + DG_Q6KWARP=1. Risk: Chrome's
chromium-experimental-subgroup-matrix syntax drift vs hesper's May Dawn.

## R35 (2026-07-16): optimal-config trace on Chrome — WMMA works, 2163ms/step, near-native parity

Captured the reg-kernel config (DG_NOMSL=1 DG_NOMSLDOWN=1 DG_Q6KWARP=1, all
other defaults = QKVRB WMMA + grouped MoE, all-WGSL): native 2576ms/step traced
(text Paris ✓), 0 holes. Chrome: **ENGINE PASS 2163ms/step, Paris ✓** —
chromium-experimental-subgroup-matrix kernels run fine on Chrome 150 (no syntax
drift). Chrome ≈ 1.2-1.3× native untraced (~1700ms), and BEATS its own traced
native run.

TPS ladder (Chrome, France): 0.13 → 3.3 → **3.5 useful tok/s** (canvas 19.7).
Session total: 57156 → 2163 ms/step = **26.4×**.

Remaining profile (flat): grouped MoE gate/up reg 23ms×30=690ms (2× native
per-layer — Chrome WMMA slower), one 198ms a,b,c WMMA (n=1, near step head),
long tail ≤3.8ms. Next levers are the REAL perf campaign (③): fewer eff-steps
(schedule/CONF), ternary/delta-prop kernel work, committed-caching — targets
llama.cpp 64 tok/s then beyond.

## R36 (2026-07-16): M2b PASS — Chrome engine 8/8 on the dg_eval suite

harness/eval8-chrome.sh: per-prompt capture (optimal traceable config) → hole
validation → Chrome engine → keyword check → delete trace (~21GB each, disk
can't hold 8). Native same-config: 8/8.

Chrome: first pass 6/8 + TWO FALSE FAILS from the engine's 200-char text log
truncating before the keyword ("cold"/"Armstrong" sat beyond) — re-judged with
1200-char logging: both PASS ⇒ **8/8, gate TOTAL == native TOTAL, M2b PASSED.**
Per-step 2.15-2.29s across all prompts (prompt-length invariant).

Bugs found this round (all harness, none engine-core):
1. Engine verdict was France-hardcoded (/[Pp]aris/) — FAILed 7/8 prompts while
   decoding them correctly; now prompt-agnostic (?expect= optional).
2. Text-log truncation (200 chars) caused keyword false-FAILs → 1200.
3. eval loop left the previous Chrome (20.6GB GPU) alive during the native
   capture → swap hit 26.5GB/27.6GB (user caught it: "swapがおおすぎる");
   pkill before capture; swap recovered to 3.3GB. Per-step times were stable
   throughout — pressure hurt headroom, not this run's numbers.
4. Editing a bash script WHILE it runs shifts bytes under bash's incremental
   read → spurious syntax error at the tail. Edit copies, not live scripts.

STATE: M0/M1/M2a/M2b all PASS. Chrome = trusted lab at ~2.2s/step, 8/8 quality.
NEXT: ③ perf campaign on the Chrome lab (eff-steps schedule, ternary,
delta-prop, committed-caching) toward 250-400 tok/s targets.

## R37 (2026-07-16): ③ perf campaign opened — Chrome-vs-native gap anatomy

Baseline: Chrome 2163ms/step vs native-untraced ~883ms (full config) = 2.45×.
Recoverable items by size:
1. UNIFORM ~2.4× spread across the flat profile (1184 dispatches, avg 1.7ms)
   → suspect robustness bounds-checks (native disables via disable_robustness;
   Chrome forces them). TESTING NOW: --enable-dawn-features=disable_robustness
   (lab-only flag, engine-dg-fast.sh harness).
2. MoE gate/up reg 23ms×30 = 690ms (2× native per-layer).
3. SC expected-embedding 198ms×1 — identified: probs[C,262k] × lmW[262k,2816]
   = 189 GFLOP dense (~1 TFLOPS achieved). Top-K sparsification candidate
   (scK=8 already exists) — numerics-gated.
On the user's swap question: swap explained the eval-loop pressure (26.5GB,
fixed by pre-capture pkill) but NOT step speed — per-step was 2.15-2.29s
with swap full AND after recovery. The 2.4× is code-side, not paging.

## R38 (2026-07-16): disable_robustness = -11%; the REAL gap is WGSL-vs-MSL, not Chrome

--enable-dawn-features=disable_robustness (engine-dg-fast.sh): steady gpu
1990→1780ms, PASS text intact ⇒ real but minor; keep for the lab.

REFRAME (important): the eval-run natives (same all-WGSL kernel set) ran
2148-2288ms/step — **Chrome 2083ms is ALREADY at parity with all-WGSL native.**
The remembered "883ms native" config includes the hand-MSL gate/up (+down).
So the 2.4× is the known Tint-quality gap of OUR WGSL kernels, not a Chrome
tax ⇒ kernel-quality work pays BOTH runtimes; the Chrome lab can iterate on
.wgsl files directly (edit → re-run, no Lean rebuild).

Campaign queue (by recoverable ms, all-WGSL step = ~2100):
(a) RE-JUDGE blocked -190ms dense kernels under the chat template
    (DG_DENSEF16 / DG_DENSEDOWNRB / both) — native eval ×3 RUNNING.
(b) MoE gate/up reg 23ms×30 — WGSL-quality pass on the lab.
(c) SC expectation 198ms — top-K sparsify (numerics-gated).

## R39 (2026-07-16): (b) micro-CSE REJECTED; (a) DENSEF16 UNBLOCKED 8/8

(b) Lab experiment: mechanically CSE'd the Q4_K grouped-reg gate/up scale-decode
(hoist 4 b[] word loads + base index into lets; 30 kernel files patched in the
trace, bit-identical semantics): 23ms → ~22ms = NO WIN. Chrome already CSEs;
the kernel is WMMA/global-load bound. ⇒ the 2× WGSL-vs-MSL gap on this kernel
class is instruction-selection depth, unreachable by WGSL surface edits.
Conclusion: kernel-side headroom on Chrome is thin; the campaign pivots to
ALGORITHMIC reduction (committed-row shrink, fewer steps, SC sparsify).
Side lesson: editing kernel files invalidates Chrome's shader cache → step-0
lazy-compile spike (81s); steady-state unchanged — measure from step ≥3.

(a) Native re-judge under the chat template (the near-tie fixer):
**DG_DENSEF16: 8/8 @ avg 849ms/step** — the -190ms dense kernel is UNBLOCKED
(was rejected pre-template for Jupiter/Water/moon flips; the flips are gone).
DENSEDOWNRB partial (running): passes so far but noisy step times.

## R40 (2026-07-16): CORRECTIONS — DENSEF16 was already default; true Chrome gap = 2.1×

Three errors caught and corrected this round:
1. DG_DENSEF16 became DEFAULT-ON in hesper (flag inverted to DG_NODENSEF16) —
   my "DENSEF16 A/B" was a no-op (dgtrace4/5 dispatch compositions are
   IDENTICAL). The 8/8 "unblock" re-judge was really a baseline re-confirmation.
2. The eval ms figures (849/1759/889) are CONTAMINATED — I ran GPU captures and
   engine runs CONCURRENTLY with the timing evals. Quality scores (8/8×3)
   stand; timings don't. LESSON: never run anything on the GPU during a timing
   eval — serialize all GPU jobs.
3. R38's "Chrome ≈ native parity" compared Chrome to TRACED native (trace
   overhead ~1.3s/step). TRUE picture: native all-WGSL ≈ 850ms/step untraced,
   Chrome ≈ 1790ms = **2.1× uniform gap on identical WGSL**.

Gap suspects after robustness (-11%) and micro-CSE (nil): Chrome likely
compiles MSL with fastMath OFF (native Dawn: ON). Further Dawn toggles
(skip_validation, disable_workgroup_init) were BLOCKED by the permission
classifier — user's call if we want to test those on the lab.

STATE: Chrome lab = 1.79s/step steady, 8/8, full profiling. Next big levers
are ALGORITHMIC (nothing kernel-side is cheap anymore):
(A) committed-row shrink — forward only masked rows (M 279→~60 by step 3),
    hesper-side dynamic-shape work, est. ~2× average;
(B) SC expectation sparsify (198ms → ~5ms, top-K, numerics-gated);
(C) schedule tuning (eff-steps 6-7 already < llama.cpp's 11 — thin).

## R41 (2026-07-16): Dawn toggles exhausted — skip_validation+disable_workgroup_init = NIL

User-approved A/B: EXTRA_DAWN="disable_workgroup_init,skip_validation" on top of
disable_robustness: steady 1787-1829ms = no change (robustness alone: 1780-1815).
Chrome-flag levers are DONE: -11% total (robustness), everything else nil.
The 2.1× residual vs native lives below the flag surface (MSL fast-math and/or
Tint codegen differences; ShaderModuleCompilationOptions isn't page-exposed).
DECISION: accept the 2.1× as the lab's constant factor; all further wins must
be algorithmic — starting (B) SC expectation top-K sparsify (198ms→~5ms
candidate, eval-gated), then (A) committed-row shrink (~2×, hesper dynamic
shapes).

## R42 (2026-07-16): (B) SC top-K sparsify REJECTED — SC strength buys eff-steps

DG_SCTOPK=1 (renoise SC = renormalized top-8 expectation via the existing
sparse gather path; hesper change: fullSC forcing lifted + per-position
renorm): quality 8/8 BUT eff-steps EXPLODE 5-13 → 30-48 (48 = ceiling hit).
Weakened self-conditioning → slower entropy collapse → stability stop never
fires. Net: 3-6× SLOWER end-to-end despite removing the ~200ms dispatch.
MECHANISM INSIGHT: the full-vocab SC expectation is load-bearing for
CONVERGENCE SPEED, not just fidelity — "SC quality ↔ eff-steps" is a real
axis. (Follow-up if ever needed: larger K (32-256) might buy most of the
convergence back — untested; reduce kernel takes K as a param.)
Flag kept opt-in; default unchanged. Both GPU-serialized, clean timings.

### (A) committed-row shrink — design sketch for the next work item
Goal: forward work ∝ nMasked (256→~40 by step 3), avg ~2× per decode.
- Committed rows' K/V per layer: FREEZE at commit time (cache buffers
  [30][N,kvDim]); recompute only masked+prompt rows each step.
- Attention: masked-row queries attend over full K/V (cached ∪ fresh).
- Dense/MoE/norms/lm_head: masked rows only (gather → compute → scatter).
- Dynamic dims via padded buckets (M ∈ {64,128,192,256}+P) — kernels are
  dim-baked, so 4 bucket variants per matmul; engine picks per step.
- Approximation caveat (from memory): committed rows' hiddens feeding NEXT
  layer's K/V change when masked rows change — freezing them is APPROXIMATE
  mid-stack (exact only last-layer/lm_head). Eval-gate decides; llama.cpp
  diffusion does full recompute, so this would be a genuine structural edge.

## R43 (2026-07-16): (A′) delta-prop opportunity MEASURED — 32% of rows, ~2.3-3.1× candidate

Correction first: renoise mode has NO mid-decode commit (masked[] flips only at
finish; non-accepted rows re-randomize every step) ⇒ "committed-row shrink"
morphs into DELTA-PROP: recompute only rows whose INPUT token changed — and the
changed set is known EXACTLY on the CPU before each forward (the scheduler
writes toks itself). DG_DELTASTAT=1 measurement (France, hesper):
  inputChanged/step = 256, 111, 74, 50, 40, 20, 15
  Σ changed = 566 vs 1792 full-rows = 32% ⇒ ~3.1× theoretical on emb+fwd
  (795ms of the 850ms step); with M-buckets {64,128,256}: ~2.3×.
Projection: native step 850 → ~380-450ms ≈ llama.cpp per-step parity, and our
eff-steps (7) < theirs (11) ⇒ end-to-end WOULD EXCEED llama.cpp (43 → ~85-95
canvas tok/s). Chrome lab follows at its 2.1× factor (~20 → ~45).

BUILD PLAN (next session-scale; staged, each stage eval-gated):
1. Per-layer K/V caches [30][N,kvHeads*hd] persisted across steps; step 0
   fills them (full forward).
2. Rectangular attention variant: M_q = changed rows (gathered), K/V = full N
   (cache ∪ fresh rows' K/V scattered in). battnB currently assumes square N.
3. Gather changed rows → norms/qkv/dense/MoE chains at bucketed M ∈
   {64,128,256}+P (kernels are dim-baked → 3 variants per matmul; reuse the
   grouped-MoE machinery which already handles variable token counts!).
4. Scatter fresh hiddens/logits; unchanged rows keep stale K/V + stale
   logits (approximation — drift compounds; DG_DELTAREFRESH=<R> full forward
   every R steps as the guardrail).
5. Gate: dg_eval 8/8 + step-count non-regression (the SCTOPK lesson: watch
   eff-steps, not just text).
Risk note: stale K/V for unchanged rows is the same approximation class that
mask-mode committed-caching would make; llama.cpp recomputes everything, so
passing the gate here is a genuine structural win over the reference.

## R44 (2026-07-16): delta-prop build STARTED — kernel vocabulary (stage 1)

Chain mapped: every generator already takes N as a parameter → gathered
contiguous [M,·] rows make all dim-baked kernels reusable with N:=M. Only 5
genuinely new kernels, now implemented in DiffusionGemmaDecode.lean:
  rowGatherB / rowScatterB     — the indirection boundary (rows[] u32, absolute)
  qkNormRopeDeltaB             — RoPE position via rows[] (1-line indirection)
  battnDeltaB (M,N)            — rectangular attention: q/ctx [M], k/v full [N]
                                 caches; mask dropped (canvas rows ≥ P always
                                 allowed, bidirectional)
  copyCanvasLogitsDeltaB       — logits scatter to canvas rows via rows[]
Padding trick: buckets pad by DUPLICATING rows[0] → scatters write identical
data to the same destination = idempotent, no masking needed anywhere.
Architecture: hesper runs DG_DELTA natively (native wins directly); per-bucket
dispatch streams are FIXED graphs → the JS engine later replays "stream for
bucket B" with dyn-substituted rows/tokens (fits the existing marker-slice +
dyn-substitution engine design — no dynamic composition needed).
Next: K/V cache buffers (30×2×[N,kvDim] ≈ 68MB) + the delta step path in the
decode loop + DG_DELTAREFRESH guardrail; parity check = DG_DELTA with
refresh-every-step must be bit-identical to baseline.

## R45 (2026-07-16): DG_DELTA v1 LANDS (hesper 15495ee) — machinery exact, policy loses on convergence

Fork-implemented per the R43/R44 design; validation ladder:
(a) baseline 846ms/step, Paris ✓
(b) DG_DELTA=1 REFRESH=1: **bit-identical trajectory** (all step stats match to
    every printed digit) — cache-rebind + full-pass wiring is EXACT.
(c) refresh=4: delta steps fired at 111→M128, 74→M128, 45→M64;
    **warm delta step 573ms vs 845 = -32%**; France correct, 6 steps.
(d) eval: **8/8 quality, avg 725ms/step** — BUT eff-steps 62→97 total (+56%,
    e.g. "opposite of hot" 13→25) ⇒ net wall time WORSE (~70s vs ~53s suite).

VERDICT: same disease as SCTOPK in milder form — staleness (frozen K/V+logits
of unchanged rows) slows entropy convergence. The eval gate's step-count check
(added after SCTOPK) caught it exactly as designed. DG_DELTA stays opt-in.
KEY REFRAME: per-step speed and convergence speed TRADE OFF through staleness;
the win condition is a recompute policy that keeps convergence — candidates:
refresh=2/3 sweep, and CONFIDENCE-GATED freezing (recompute changed ∪ high-H
rows; the frozen high-H rows are likely what stalls acceptance).

## R46 (2026-07-16): DG_DELTAREFRESH=2 = the FIRST net delta-prop win (~-6% suite wall)

Refresh sweep (all 8/8 quality):
  REFRESH=2: 74 eff-steps (+19% vs 62), avg 669ms → suite ~49.5s vs ~52.6s
             baseline = **net -6% end-to-end, quality intact** ✓ first win.
  REFRESH=3: 96 steps — loses on steps alone (timings untrustworthy: swap crept
             to 7.5GB across 16 consecutive 15.7GB model loads — suite harness
             measurement caveat: STEP COUNTS are the reliable cross-config
             metric; build a single-load eval mode before timing sweeps).
Entropy-gated recompute (DG_DELTAHMIN, hesper 2450616, refresh-1 bit-identity
re-verified): at REFRESH=8 HMIN=5 FAILS suite-wide (101 steps) BUT France
single-prompt = 10 steps × 439ms = 4.4s — the strongest single-prompt result
yet (baseline 5.9s). The staleness/convergence frontier is real and tunable;
untested: HMIN at REFRESH=2.

## R47 (2026-07-16): delta-prop frontier CLOSED — production setting = DG_DELTAREFRESH=2

Final sweep (R2 vs R2+HMIN5 vs R2+HMIN10): all 8/8, and all three produce
IDENTICAL per-prompt step counts (8,11,12,8,8,6,12,9 = 74) — at refresh-2 the
drift window is 2 steps, so high-H rows coincide with token-changed rows and
the entropy gate never fires. HMIN's niche was high-refresh (8+) where it
already failed to restore convergence ⇒ knob stays, default 0.

RECOMMENDATION: DG_DELTA=1 DG_DELTAREFRESH=2 (74 steps × ~669 vs 62 × ~849 ≈
net -6%), with one caveat: all suite timings today are swap-contaminated
(7.9GB residue, flat across runs — the runs don't grow it, macOS just won't
reclaim); a single clean-boot confirmation measurement should finalize the -6%.
Step counts (deterministic) are trustworthy throughout.

CAMPAIGN LEDGER (session): 57.2s→2.16s/step Chrome (26×); native 0.85s;
M0-M2b all passed; delta-prop machinery landed bit-exact (15495ee, 2450616)
with the first structural win llama.cpp cannot replicate (-6%, quality 8/8).
Open items: clean-boot timing confirmation; single-load eval harness (kills
the swap-creep measurement class); HMIN×high-refresh convergence research;
JS engine port of the delta streams (buckets are fixed graphs → engine-ready).

## R48 (2026-07-16): delta-prop PORTED to the Chrome engine — delta step = -50% on Chrome

Fork-built (e4b 3ca9e59 + hesper f7714da DG_TRACE_END): per-step stream
classification (full vs delta-M by embed grid, threshold derived from step-0 —
P varies by prompt!), runtime policy mirrors hesper (odd steps, fit-to-bucket,
full fallback), rows/tokDelta identified from the delta stream's w-events and
dyn-substituted, padding = dup-entry-0. Lazy per-stream pipeline compile added
(3402 kernels upfront hung Chrome 25min ×2; lazy = 0.3-0.6s per batch).
Chrome France: PASS "…Paris."; full step ~1820ms; **DELTA steps 899-1121ms =
-50%**; buckets [64,128] fired; engine 8 eff-steps vs native 6 (known q8-flip
trajectory divergence, not delta-specific). Ops discipline: disk had filled
(100%) with stale traces — cleared 179GB; trace window needed DG_TRACE_END.

## R49 (2026-07-16): fast-math WALL hypothesis REFUTED

HESPER_STRICT_MATH=1 native France (no trace): **843-846ms/step — IDENTICAL to
the fast-math baseline** (and the trajectory shifted acc 145→143, proving the
flag engaged). Strict math costs native nothing ⇒ Chrome's 2.1× is NOT
fp-strictness. Remaining suspects: Tint May-vs-July codegen at scale (k15 CLI
diff showed only clamp/scheduling deltas — but that was one kernel), Metal
pipeline/QoS differences in Chrome's sandboxed GPU process, binding-level
robustness residue. Decisive-but-heavy next experiment: build hesper against a
July Dawn snapshot (2h build) — if native slows to ~1.8s, it's the Dawn
version, not Chrome. PARKED as an open item; the lab's constant factor stands.

## R50 (2026-07-16): convergence tax DIAGNOSED — real dynamics, policy-irreducible; delta ceiling = refresh-2

Phase-1 decomposition (planet prompt, DG_DELTADIAG @ hesper 943de53):
frozen rows' H ≈ 5e-5 (drags meanH DOWN, not up), sort first, consume ~0 cumE,
argmax bytewise stable ⇒ hypotheses (a) stale-H stop-stall, (b) acceptance
perturbation, (c) stability delay — ALL REFUTED. The smoking gun is the
all-fresh refresh step: meanH 0.0257 vs baseline 0.0072 (3.5×) — the drift is
BAKED INTO THE TOKEN TRAJECTORY: rows commit against 1-step-stale K/V of
frozen rows, and slightly worse commits slow real convergence. Probe
DG_DELTAMINROWS=96 (restrict delta to one early step): 9 steps — worse than R2
(8) and baseline (7) ⇒ ANY single delta step forks the trajectory with a small
negative step bias (the known near-tie fragility class). No fix ships; the
menu targeting (a)/(b)/(c) would have been theater.

VERDICT: the +~1 step/prompt tax at refresh=2 is architecturally irreducible
by freeze policy — refreshing frozen rows' K/V requires their hiddens = a full
pass. **Delta-prop is CLOSED at its measured ceiling: DG_DELTAREFRESH=2, net
-6%, banked.** Side caveat logged: on delta steps the stop reads a diluted
meanH (fresh-rows H ~0.17 at fire time); quality tolerated it (8/8).
Honest research ledger for delta-prop overall: 1 structural win (-6%),
4 policy ideas refuted with mechanism-level evidence, machinery bit-exact and
reusable (the caches/buckets/rect-attention will serve any future partial-
recompute scheme, e.g. mask-mode decoding where commits ARE sticky).

## R51 (2026-07-16): THE 2.1× LAB FACTOR SOLVED — a Dawn May→July runtime regression

Stage 1 (tint CLI MSL diff, hot WMMA kernels): instruction selection IDENTICAL
(same MMA/load counts, native mixed-precision, no fast-math pragmas; July even
elides ~half the robustness clamps; side-note: July tint ICEs on mixed-precision
simdgroup_multiply_accumulate unless IR validation asserts are off). Tint
codegen EXONERATED.
Stage 2 (hesper rebuilt on July Dawn a192e3019; bridge.cpp needed ZERO changes):
**July-Dawn native = 2068-2262ms/step vs May = 854-896ms — uniform ~2.4×,
matching Chrome's 1790-1820ms.** lmhead+reduce 185 vs 53ms. Restore verified
(exact baseline trajectory back).

VERDICT: the Chrome lab's 2.1× is a **Dawn runtime regression** (Metal backend
behavior — NOT shader codegen, since the MSL is identical): suspicion set =
inter-dispatch barrier/pass-splitting/resource-tracking changes. Chrome is
"innocent" only in that it faithfully ships the regressed Dawn.
Consequences: (1) hesper's May Dawn pin is PROTECTIVE — do not upgrade blindly;
(2) cleanly bisectable (~10 builds, artifacts kept at /tmp/dawn-july-*,
/tmp/hesper-native-july, dylib backup .may); (3) upstream bug report warranted
after bisect; (4) if upstream fixes it, the Chrome lab gets ~2× for free
(→ ~40 canvas tok/s in-browser).

## R52 (2026-07-17): Chrome version pin experiment — 2.1× decomposed into TWO factors

Chrome for Testing 147/148 (Dawn ~Apr/early-May) vs 150 (July), same trace/
harness (CHROME_BIN override added to engine-dg-fast.sh; fresh profiles):
  147: full 1387-1635ms, delta 536/766ms (delta ≈ NATIVE's 573!)
  148: full 1233-1582ms, delta 763/827ms
  150: full 1780-1840ms, delta 899-1121ms
  native-May 854-896/573; native-July 2068-2262.
DECOMPOSITION: (1) Dawn May→July regression ≈ 1.2× in Chrome (150 vs 147/148),
confirmed independently by the native A/B (2.4× standalone — the standalone
July build is even worse than Chrome 150, suggesting Chrome carries partial
mitigations or standalone validation differences). (2) A residual Chrome-env
factor ≈ 1.6× on FULL steps at 147/148 — NOT uniform: delta steps reach native
parity (536 vs 573ms) while full steps don't ⇒ the env penalty concentrates in
the big-M kernel classes (WMMA/MoE at M=277), worth a per-kernel profile on
148 someday. Gotcha: first launch of a fresh CfT profile can no-op (profile
creation race) — retry; and first run pays a big shader-compile step-0 (13s).
ACTION: the lab gains ~17-20% by pinning Chrome 147/148 (CHROME_BIN env,
binaries kept at /tmp/claude-503/cft/). Chrome-150 numbers remain the
comparable series in this log unless noted.

## R53 (2026-07-17): KERNEL PANIC during Stage 1 — resource guardrails now mandatory

Machine panicked 21:12 ("watchdog timeout: no checkins from watchdogd in 93s")
with Jetsam OOM events preceding — the system-unresponsive-from-memory-pressure
signature. Our footprint is implicated: ~20GB wired GPU buffers + repeated
swap excursions (7-26GB today) + ~21GB trace copies on a ~90%-full disk +
Chrome engine residents. The "never kill -9 GPU work" rule protected the GPU
state but we were breaking the machine a DIFFERENT way (aggregate pressure).
NEW MANDATORY GUARDRAILS (issued to all workers):
  - pre-flight before heavy steps: swap used <4GB AND disk free >60GB, else
    clean up first; abort gracefully if swap >10GB mid-run;
  - APFS clones (cp -cR) / hardlinks for trace dump copies — never cp -r 21GB;
  - ONE heavy process at a time; kill lab Chrome profiles right after each
    measurement (no resident 20GB tabs).
Stage 1 (hand-WGSL twin) resumes on the clean post-reboot machine.

## R54 (2026-07-17): Stage 1 verdict + A MEASUREMENT-HISTORY REWRITE

Twin experiment (hand-WGSL with all hygiene fixes — unpack2x16float, single
loads, staged locals — vs generated DSL kernel, Chrome 148, clean boot,
paired): **IDENTICAL (10.2-12.4 vs 10.1-12.6ms)**. Surface authorship quality
is IRRELEVANT — the Metal compiler already extracts full performance from the
generated WGSL (consistent with R39 CSE-nil). Phase-0 strategy diff confirmed
the hand-MSL and DSL kernels are ALGORITHM-IDENTICAL (same tiling/fragments);
the DSL's ugliness (exp2-chain f16 decode, double loads, monster B-fill) is
cosmetic. Side finds: 30 layers compile 30 byte-identical kernels under
different hashes (redundant pipeline compiles — fix candidate); Tint MMA
lowering is correct.

THE REWRITE: pre-panic machine degradation had inflated ALL Chrome-side
numbers ~1.7-2× (same kernel 23→11ms, full step 1250-1580→**757-790ms** after
clean boot, zero code changes). Native was INSENSITIVE (841-887ms throughout)
— Chrome's sandboxed GPU process is the memory-pressure victim, native Dawn
is not. Consequences:
  - Clean Chrome 148 lab (757-790ms) is now FASTER than native (841-887ms).
  - R52's "1.6× Chrome env factor" = likely contamination; re-baselining now
    (clean Chrome 150 run in flight).
  - The 8332c90 "hand-MSL 1.61×" claim also needs clean re-verification —
    Stage 2's premise is UNDECIDED until then.
  - Native-vs-native results (July-Dawn 2.4× regression, delta-prop numbers,
    eval step counts) are unaffected (native insensitive + step counts
    deterministic).
NEW MEASUREMENT RULE: Chrome-side timings are only valid with a clean-state
pre-flight (swap <1GB, no Jetsam events since boot); log the machine state
with every measurement.

## R55 (2026-07-17): re-baseline complete — the picture is finally COHERENT

Clean-state measurements (post-reboot, guardrails, serial):
  Chrome 148 (May-class Dawn): 757-790ms | native all-WGSL (May Dawn): 841-845
  Chrome 150 (July Dawn): 1780-1807 (UNCHANGED from "contaminated" era — 150's
  slowness was always real) | native July rebuild: 2068-2262
  native + hand-MSL gate/up+down: **662-663ms = -21% vs all-WGSL** ✓ REAL
FINAL DECOMPOSITION: the lab factor was the Dawn May→July regression, PERIOD
(~2.3×, present in both embedders). The "1.6× Chrome env factor" (R52) is
RETRACTED — an artifact of measuring 147/148 on the degraded pre-panic
machine; clean Chrome 148 is FASTER than native-WGSL. Contamination asymmetry
explained: the degraded state hit Chrome's GPU process, not native.
M-METAL PREMISE RE-VERIFIED CLEAN: the direct-Metal path's edge is real
(-180ms from 2 kernels; sources: Tint-vs-hand MSL codegen + skipping Dawn
dispatch + no robustness clamps — note run A had native robustness ON).
Stage 1 verdict stands: WGSL surface authorship is irrelevant; the wins are in
the EXECUTION PATH (Metal direct) and ALGORITHM classes, not WGSL phrasing.
STAGE 2 = GO. Projection: full-Metal backend ~500-600ms/step, then kernel
algorithm work toward llama.cpp's 363; with DG_DELTA + fewer eff-steps the
64 tok/s end-to-end target is credible.
Lab operating point going forward: Chrome 148 pinned (CHROME_BIN), clean-state
pre-flight mandatory, machine-state logged with every Chrome measurement.

## R56 (2026-07-17): M-Metal Stage 2 LANDS (hesper 1b9536e) — Dawn out of the hot path

Thin Metal backend (~650 LOC: metal_backend.mm + 25 branched FFI entries;
HESPER_BACKEND=metal, zero change when unset): shared-mode buffers explicitly
ZEROED (Dawn parity — uninitialized-read class stays dead); runtime WGSL→MSL
via the pinned May tint CLI (--disable-robustness, fast-math; content-hash
disk cache — also kills the 30-duplicate-compile finding); serial compute
encoder per batch = Dawn barrier semantics; reads fence on the last batch.
Tint CLI rebuilt in-tree (.lake/build/tint-cli, HESPER_TINT override).

Gates: (a) coherency smoke 20/20; (b) France "Paris." with a trajectory
IDENTICAL TO DAWN to every printed digit (same tint codegen + serial order ⇒
same reduction order); (c) **762ms/step vs Dawn 841-845 = -10%**, 44.1 canvas
tok/s; (d) eval **8/8, eff-steps 62 = exact baseline**.

THE THESIS IS RUNNING: checker-covered kernels, robustness OFF, no Dawn, no
July-regression exposure — with bit-identical numerics and full quality.
Gap to the 662ms target = the 2 hand-MSL kernels (Dawn-coupled; metal mode
needs DG_NOMSL=1 for now) → stage-2b routes them onto the metal queue,
projected ~580-600ms. Ladder: llama.cpp 363 | MSL-hybrid 662 | METAL 762 |
Dawn-WGSL 841 | lab 757-790 | Dawn-July 1780+.

## R57 (2026-07-18): Stage 2b LANDS (hesper 0ec5d40) — fastest config ever; llama.cpp within noise

Hand-MSL gate/up + down now dispatch on the metal backend's own queue (MTLBufs
direct from HMBuf, no Dawn-internal extraction; commit order = execution order
so the Lean flush contracts carry over unchanged). HESPER_BACKEND=metal now
runs the FULL default kernel set with no exclusion flags.

| config | France /step | eval emb+fwd | eval |
|---|---|---|---|
| Dawn all-WGSL | 841-845 | — | 8/8 (62 steps) |
| Dawn + hand-MSL (old best) | 662 | ~608 | 8/8 |
| Metal, all tint-MSL | 762 | 722 | 8/8 (62 exact) |
| **Metal + hand-MSL (2b)** | **611-612** | **554** | **8/8 (63, family)** |

-8% vs the best Dawn config; **canvas ≈ 60 tok/s vs llama.cpp 64 — parity
within measurement noise**, with Stage 3 (MSL checker) and Stage 4 (concurrent
submission) still ahead. The M-Metal thesis is fully operational: verified
kernels, robustness off, Dawn structurally out, July-regression immune.
Session ladder (per-step): 2700 (campaign start) → 883 → 830 → 662 → 611.
Remaining known slack: Stage 4 concurrency, kernel algorithm classes, delta-prop
(-6% banked, composes), schedule. Housekeeping debt: gpu-roundtrip exe bit-rot,
stray float64_to_bytes debug print.

## R58 (2026-07-18): Stage 3 LANDS (hesper d8d6491) — the verification leg closes

MSL front end on wgsl-check (~460 LOC): kernel-signature scan ([[buffer(n)]] →
storages, thread attrs → builtin map), body normalization to the shared walker
(inline-helper substitution makes rd_byte's reads visible; ternary→select;
C decls→let/var), manifests generated from the REAL dispatch sites with
line-documented provenance. Shared-analysis upgrades benefit WGSL too
(select-under-refinement, let-bound bool guards, v+const refinement,
var-initializer bounds). **simdgroup_load/store footprint check = the R32
WMMA-tail heap-stomp class, now caught statically** (fixture proves it).

Production verdicts (q4k_gateup / q8_down / q5_down): **0 FAIL, 5 WARN** — the
q4k simdgroup stores prove in-bounds TIGHT BY ONE ELEMENT; the WARNs are
honest trust boundaries (counting-sort invariants for idx/pos/slot indirection;
DG_FUSEDOWN-only dead code), not noise. Unsoundness boundaries documented
(no pointer locals / threadgroup-mem bounds / multi-stmt helpers → WARN).
Gate: `bash scripts/msl_check.sh` (fixtures must FAIL + production must not;
exit 2 if the suite itself breaks). WGSL spec suite regression: identical.

M-METAL STATUS: stages 1-3 DONE. Every kernel in the metal backend is now
statically checked — tint-translated ones at their WGSL source, hand-MSL by
this front end. Remaining: Stage 4 (concurrent submission via static hazard
sets) — the beyond-Dawn endgame. Note for later: manifests are France-config
(N=277); per-config regeneration from traces generalizes the gate.

## R59 (2026-07-18): Stage 4 LANDS (hesper 53692f9) — M-METAL COMPLETE (all 4 stages)

Concurrent encoder + static hazard analysis (barrier only on RAW/WAW/WAR at
buffer granularity; write masks parsed from WGSL with store-site refinement —
declared modes alone gave zero overlap since WMMA generators declare all
read_write; wrong demotion = missed hazard = bit-identity gate failure;
HESPER_METAL_SERIAL / _NOREFINE kill-switches; _STATS accounting).
Gates: bit-identity PASS (every printed stat identical); eval 8/8 (steps
identical by construction); coherency 20/20; msl_check 0 FAIL.
Timing: 622-663 → **609-644ms (~-2%)**, barrier ratio 47.2%. HONEST verdict:
correct but small — the hoped-for overlaps are blocked by buffer-granularity
WAW false hazards (8 lm_head chunks write disjoint REGIONS of one logits
buffer → serialize) and genuinely RAW-chained pipelines. Future unlocks noted:
region-granularity footprints, or splitting lm_head chunk outputs into 8
buffers (cheap hesper change). Concurrent stays metal-default (never slower,
bit-identical).

### M-METAL MILESTONE CLOSED — ledger
S1 twin: WGSL authorship irrelevant; measurement-history rewrite (machine
contamination; env-factor retracted; Dawn July regression = the whole lab gap).
S2 thin Metal backend: Dawn out of hot path, -10%, bit-identical, 8/8.
S2b hand-MSL hosted: 611ms = fastest ever (-8% vs best Dawn).
S3 MSL checker: verification closed over both kernel languages; R32 bug class
statically caught; production 0 FAIL.
S4 concurrent: correct (+bit-identical) but -2%; submission order is no longer
the bottleneck.
FINAL STATE: ~609-611ms/step, canvas ≈60 tok/s vs llama.cpp 64 (noise-level).
The 363ms horizon = kernel/algorithm work (region hazards, lm_head split,
matmul classes) + delta-prop composition (-6% banked) + schedule.

## R60 (2026-07-18): compute-utilization diff vs llama.cpp — user hypothesis CONFIRMED (WMMA class)

Analysis-only pass (recipes/DG_COMPUTE_ANALYSIS.md, clean-anchored):
  MoE gate/up+down (hand-MSL): 230-280ms @ 30-40% — ALREADY ≥ llama.cpp
    mul_mat_id (anti-finding, leave alone).
  **dense+qkv+attnO f16 WMMA (M=277): 165-205ms @ 10-15% util — THE recovery
    source, occupancy/pipeline-limited.** Root config diff (llama.cpp source
    read): their kernel_mul_mm uses TG tile 64×128 (ours 64×32 = 1/4 the work
    per TG) and stages ONLY quantized A in threadgroup memory — f16 B is
    simdgroup_load'ed DIRECT from device; we double-stage A and B via
    div/mod-chain per-element indexing. Not WGSL-vs-MSL, not surface code
    (R54 proved that layer irrelevant) — TILE GEOMETRY and LOAD PATH.
  battnB: 30-45ms @ 1-2% — algorithm class (no flash-attn).
  elementwise tail 60-90ms latency-bound; SC/lm_head near roofline.
RANKED PLAN: ① WMMA tile widening + direct-B (-100~140ms, cheap generator
parameter sweep) → ② elementwise fusion re-judged under template (-20~40) →
③ flash-attention (-20~35, high effort). All three → ~400-455ms/step →
canvas ~85-91 tok/s = clearly BEYOND llama.cpp end-to-end.
Measurement notes: metal-mode DG_PROF inflates 2.4× (per-mark waits; ratios
only); Chrome profile under 4.6GB swap residue (ranking only).

## R61 (2026-07-18): ① tile sweep = honest NEGATIVE + R60's top line RETRACTED

Sweep (real DG shapes, golden-gated, metal backend): deployed 64×32-staged is
already at **8.2-8.9 TFLOPS ≈ 60% of peak** on the big shapes; wide tiles LOSE
(register pressure; 128×64 >2× worse); direct-B works (May tint accepts
storage-space subgroupMatrixLoad — capability banked as the generator's directB
param, hesper fe3ef86, defaults unchanged, France bit-identical) but wins only
~16ms/step total < 25ms threshold → NOT integrated, per the pre-registered
stop rule.

**RETRACTION: R60's "WMMA class at 10-15% util, -100~140ms recoverable" was a
wrong utilization estimate.** In-decode WMMA time ≈ Σ(bench time × layers) —
the class already runs at bench speed, near llama.cpp-class efficiency. The
611-vs-363 gap re-attributes to: MoE auxiliary chain (router/sort/gather/
scatter/geglu/q80 dispatches), elementwise tail (60-90ms; llama.cpp fuses
heavily), attention algorithm (no flash-attn), SC. The recovery list must be
re-ranked from these — each a 20-90ms grind, no single big lever left.
Ledger note: two analysis passes in a row produced wrong top-line estimates
(R60 util; R52 env factor) — both caught by the measure-before-integrate
discipline before any code shipped on them.
