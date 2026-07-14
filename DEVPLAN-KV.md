# Campaign 5 (KV-cache compression) — separate track from the kernel/engine campaigns

This is MODEL-BEHAVIOR / algorithm work (replication+adaptation of the
SnapKV/TOVA line to gemma4) plus an ENGINE FEATURE (long context for the
browser A4B). It is deliberately documented apart from the kernel-speed
campaigns (1-3): the claims are different in kind. Campaign 5 does NOT claim
faster-than-llama.cpp anything; it claims (a) the browser engine gains
bounded-memory long context with measured retrieval quality, and (b) one
model-specific mechanism finding (gemma4 retrieval attention is
generation-time; a one-token probe repairs SnapKV's assumption). Novelty
honestly ~80% replication / 20% new (the mechanism observation + gemma4
adaptation). Python prototype lives in e2b-jamba/src/kv_proto.py.
# Campaign 5: KV-cache compression — long context for the browser engines
# (M0 begun 2026-07-13 12:38; pivot decision recorded in e2b-jamba Campaign 4)

**Why this, from measured evidence:** Campaign 4 established that replacing
attention MATH costs retrieval precision (0.2-1 nats/layer, additive).
Cache-side compression keeps EXACT softmax retrieval for surviving entries —
the better-conditioned attack on the same goal (bounded-memory history).
Product bite: the engines cap at MAXSEQ=640 today; gemma4's architecture
(sliding windows + KV sharing) already bounds everything EXCEPT the
full-attention layers, whose caches grow linearly and gate long context.

**Two-layer plan (campaign-3 pattern):** prototype POLICIES in PyTorch on E2B
(reusing e2b-jamba venv/model/goldens — same architecture family incl. the
full/sliding/KV-shared structure), then port the winner to the A4B WebGPU
engine (WGSL) where memory actually binds.

**Policies to prototype** (full-attention layers only; sliding layers are
already window-bounded): (a) StreamingLLM: k sink tokens + recent window;
(b) SnapKV-class: prefill-time top-B selection by last-window attention mass;
(c) merge variant: evicted entries MERGED (weighted) into survivors — the
"representation-space summarization" the owner asked about; (d) random-evict
control; (e) full cache baseline.

**Eval design (pre-committed):** long-context needle retrieval (synthetic
fact placed at depth {10, 50, 90}% of a 4-8k context; exact-answer scoring),
long-seq ppl on held-out markdown, ΔKL vs full-cache on identical inputs, at
budgets {12.5%, 25%, 50%} of the full-layer cache.

**Pre-registered predictions:**
- **P1**: StreamingLLM at 25% budget keeps ppl within 5% but FAILS mid-depth
  needle (<50% retrieval) — position-based eviction can't keep content it
  never looks for. (conf 70%)
- **P2**: SnapKV-class at 25% budget keeps needle ≥80% at all depths and
  ΔKL ≤0.1 nats. (conf 55%)
- **P3**: merge beats plain eviction at 12.5% budget on ppl (≥10% relative)
  but NOT on needle. (conf 50%)
- **P4**: the winning policy in WGSL costs ≤5% decode overhead at 8k context
  on the A4B engine. (conf 60%)
- **P5**: author time — python prototype with full eval grid ≤1 leg; engine
  port ≤1 leg.

**Milestones:** M0 this pre-registration (★) → M1 python long-context harness
(needle gen + budgeted-cache generate loop) + policy grid → M2 verdicts +
policy pick (★) → M3 WGSL port to the A4B engine (MAXSEQ 640→8192, budgeted
full-layer caches) + gate/perf → M4 record.

## Campaign 5 M1/M2 results (2026-07-13): policy verdicts — and a mechanism find

Needle grid (E2B python, ctx 4096, depths {10,50,90}%, budgets {12.5,25,50}%,
full-layer caches only; harness gate full-cache 3/3):

| policy | needle | notes |
|---|---|---|
| stream (sinks+recent) | 3/9 (only d=0.9) | **P1 HIT** — position-based eviction keeps only recent needles |
| snap, prefill-window queries | 0/9 | first implementation FAILED — see mechanism |
| **snap, generation-query scoring** | **9/9 incl 12.5% budget** | **P2 HIT decisively** (bar: ≥80% @25%) |
| merge (+absorb evicted) | 9/9, ppl ≈ snap (30.4 vs 30.6 @12.5%) | **P3 MISSED** — merging adds nothing here |
| rand control | 1/9 | sanity ✓ |

**Mechanism finding (the campaign's keeper):** with prefill-window queries the
needle ranked 663/4096 at L4 but **2593/4096 at L14** — deep full layers
barely attend to the fact during prefill. Scored with the FIRST GENERATED
token's queries instead (TOVA-style), the needle ranks **0-23 at every
layer**. In gemma4, retrieval-shaped attention appears at generation time,
not in the prompt's last window — SnapKV's observation-window assumption
fails here; the fix costs one probe token before compression.

Retrieval-stress ppl (tiled-copy continuation — measures copy-source
retention, NOT natural ppl; noted): full 1.11 / snap 8.2@25%, 30.6@12.5% /
stream 150-213 (catastrophic). Ranking consistent with needle.

**M2 policy pick (★): snap with generation-query scoring + pooling; drop
merge (no gain) and stream (fails the point).** M3 = WGSL port to the A4B
engine: MAXSEQ 640→8192, budgeted full-layer caches, in-engine probe-token
scoring. P4 (≤5% decode overhead) and P5 (port ≤1 leg) stand.

## M3 design note (2026-07-13): what compression actually buys the engine

Honest re-derivation before porting: at 8k context the A4B full-layer caches
are only ~168 MB — memory alone does NOT require compression at 8k. The real
binding constraint is the ATTENTION KERNEL: attnf32 materializes
`probs[MAXSEQ]` in workgroup memory (32 KB cap → MAXSEQ ≤ ~2k with the other
arrays). So the sharp claim for M3 is:

**A budget-capped cache (B=1024 + margin) keeps the existing simple kernel
shape valid at ANY context length — unbounded context with FIXED memory and
FIXED per-token attention cost, quality measured by the M2 gates.**

Design: sliding layers → ring KV (slot = pos mod window; window already
bounds them). Full layers → capacity CAP = B+128; the attention pass
accumulates per-slot score mass (probs summed over heads — data it already
computes); every time count hits CAP, a compaction evicts to top-B by score
(v1: tiny readback of CAP floats every 128 tokens ≈ ≤0.3%/token, then a GPU
gather; GPU-side selection later if the tax shows). Prefill beyond CAP:
chunked prefill with a one-token generation-style probe per chunk boundary
(applies the M2 mechanism finding). Robustness check added to the gates:
needle with ≥2 question phrasings (the mechanism finding could be
task-shaped).

## Positioning correction (2026-07-13, owner asked "does llama.cpp already
## have this? shipped?")

Verified against the local fork + web: llama.cpp SHIPS (a) `--ctx-shift`
(positional: keep-prefix + shift = our "stream" class), (b) cache-type
quantization `-ctk/-ctv`, (c) iSWA bounded sliding caches. It does NOT ship
importance-scored eviction (SnapKV/TOVA/H2O class) — structurally hard there
because FA kernels don't expose attention weights. Our M2 measurement is
therefore directly a comparison of "what llama.cpp has" vs "what it lacks":
positional (stream) keeps 3/9 needles; generation-query scoring keeps 9/9 at
12.5% budget. Claim upgrade: the engine implementation is AHEAD of llama.cpp
for this mechanism class (research itself remains ~80% replication).

## M3 scope refinement (Stage A landed; the probs[] wall)

Stage A (ring sliding + counted full slots) landed bit-exact (gates PASS,
decode 10.57 ms unchanged). Honest scope cut discovered while designing
Stage B: the attention kernel's workgroup `probs[]` caps prefill attention at
~2k entries (32 KB workgroup memory), so M3-v1 supports **prompts ≤ 2048**
with PRECAP=2304 full-layer slots; compression happens ONCE at the
prefill→decode transition using the M2-validated generation-probe scores
(BUDGET=640), then decode stays budget-maintained with in-flight scores
(re-compact every 128 tokens). 8k+ prompts need an online-softmax attention
kernel — deferred to M3-v2, recorded. The needle gate runs at ctx 2048 with
budget 640 (31%) and 512 (25%) — python M2 held 9/9 at 12.5%, so margin
exists.

## M3 verdicts (2026-07-13): the port landed — and compression is FREE-or-better

Stage B: scored decode attention (in-flight per-slot mass), kvgather
compaction (CPU top-B every 128 tokens, probe-then-compact per the M2
mechanism), generateLong; needle gate = 6 cases (3 depths × 2 phrasings, ctx
2048, budget 640 = 31%).

- **Regression gates: PASS** (short-context decode/prefill untouched).
- **Needle absolute: 2/6 — but the FULL-CACHE CONTROL is IDENTICAL 2/6**,
  with near-identical outputs per case. The failures are the MODEL's (A4B
  cannot retrieve d=0.1 needles at 2k even uncompressed; phrasing q1
  degenerates) — E2B (python, 9/9) and A4B differ substantially on this task.
  The correct compression metric is PARITY WITH FULL CACHE: ~6/6.
- **P4 HIT with sign flipped**: budget-640 decode at 2048 ctx = **11.18
  ms/tok vs full-cache 12.96 = 14% FASTER** — fewer attention entries beat
  the compaction tax (bar was ≤5% overhead).
- **P5 HIT**: port ≈ one leg (Stage A bit-exact refactor + Stage B).
- Caveats recorded: v1 prompt cap 2048 (workgroup probs wall; 8k = online
  softmax, v2); the params[] extension invalidates Campaign 3's traced Metal
  manifests (re-trace needed if the runner is used again); needle task is
  phrasing-sensitive on A4B — parity, not absolute retrieval, is the gate.

## M3-v2 + M4 close (2026-07-13): 8k landed; the claim is now measured end-to-end

v2 work: (1) **ring clobber bug found & fixed** (Stage A's ring = exactly
window let a prefill chunk overwrite entries its own earlier tokens needed;
RING = window+512 slack + per-slot position recovery for window/causality —
full-cache needle improved 2/6 → 3/6, so the bug had been suppressing
results); (2) **attnos.wgsl** — online-softmax full-layer attention (8
subgroups stream positions with running max/sum/acc + log-sum-exp merge; no
workgroup probs[] → no context cap; the uniform-trip-count pattern for
subgroupAdd), SCORE pass re-derives normalized mass for the compactor;
PRECAP → 8320.

**Final ladder (browser A4B engine, budget 640, all gates green):**

| ctx | needle (full) | needle (budget) | decode full | decode budget |
|---|---|---|---|---|
| 2048 | 3/6 | 3/6 (parity) | 12.96 ms/tok | 11.18 (-14%) |
| 4096 | 3/6 | 3/6 (parity) | — | — |
| 8192 | 3/6 | 3/6 (parity) | 21.82 ms/tok | **12.48 (-43%)** |

(3/6 = the q0 phrasing retrieves at ALL depths at every length — at 8k the
budget is 7.8% and the needle still comes back; q1 phrasing fails even
uncompressed = model-level, both paths identical.)

**Campaign 5 verdicts: P1 ✓, P2 ✓ (decisively), P3 ✗, P4 ✓✓ (sign flipped:
compression is FASTER, -14% @2k, -43% @8k), P5 ✓.** Engine capability: MAXSEQ
640 → 8192 with fixed decode memory/cost; the only WebGPU A4B engine now does
bounded-budget long context with measured retrieval parity. Known limits
recorded: prompts ≤ 8192 (PRECAP; extendable), needle is phrasing-sensitive
on A4B, Campaign-3 Metal traces need re-capture after the params[] extension.

# Campaign 6 Phase B: ternarizing the A4B q4_0 engine (owner-directed)
# (B-M0 begun 2026-07-13 23:35; E2B phase recorded in e2b-jamba DEVPLAN)

**Why A4B is the honest host:** the E2B QAT-mobile base already int2's its
insensitive layers (Google's QAT did our mixed-precision job); the A4B GGUF
is uniformly q4_0, and its MoE EXPERT tensors are ~12.8 GB of the 14.4 GB —
ternarizing experts alone takes the engine to ~7.3 GB (2-bit planes + per-row
f16 scales), on a bandwidth-bound decoder.

**Scope (phase B): EXPERT tensors only, decode path first.** Dense/attention
stay q4_0. Offline python ternarizer reads the GGUF q4_0 blocks directly and
writes a sidecar (t2 planes packed 16 vals/u32 in shift-grouped order so the
WGSL unpack yields contiguous vec4s + per-row scales); the engine loads the
sidecar under ?t2=1; two kernel variants carry decode (q40gu expert path,
q40moedown).

**Pre-registered predictions:**
- **B-P1**: RTN per-row ternary on ALL experts: the token gate FAILS but text
  stays coherent-ish (MoE expert redundancy absorbs more than dense layers
  did on E2B); engine-measured ΔKL ∈ [0.5, 3]. (conf 50%)
- **B-P2**: AA (h-diag from engine-captured moeIn/geglu second moments)
  improves expert ternarization by ≥ 1.5× on ΔKL, as on E2B. (conf 55%)
- **B-P3**: layer-mixed precision (worst ~8 layers' experts stay q4_0)
  reaches gate-token drift only after ≥ 8 tokens AND ΔKL ≤ 0.3, at engine
  memory ≤ 9 GB (vs 14.4). (conf 40%)
- **B-P4**: decode gets FASTER: expert reads shrink ~2.8×; MoE-bound decode
  fraction ~55% → predicted total decode ≥ 1.25× speedup. (conf 55%)
- **B-P5**: ternarizer + loader + 2 kernels + first gate in ≤ 2 legs.

### Phase B measurements (2026-07-14)

- t2 kernels verified BIT-EXACT vs JS reference after the harness fix:
  yGpu=-1.0953922 vs yJS=-1.0953921 (fp32 rounding), ratio 1.0000.
  (An earlier "ratio 1.0000" claim in-session was FALSE — the JS side was
  NaN from double-f16 decode + stale page JS; corrected, re-run clean.)
- Quality (RTN, experts-only, relMSE ~0.28/tensor): coherent text; token-level
  gate diverges at token 0-2 (expected for a lossy transform); **needle 3/6 =
  EXACT PARITY with the full-precision full-cache baseline** — long-context
  retrieval survives expert ternarization. But prompt-0 top-1 logit flips
  (818 @22.56 vs baseline top-1 236776 @15.74 rank-3) — drift is real;
  AA + mixed precision remain untested on A4B.
- **B-P4 decode speed: MISS.** grp=0 q4_0 baseline 12.46 ms/tok (80.2 tok/s)
  vs t2 11.82 ms/tok (84.6 tok/s) = 1.05x, predicted >= 1.25x. Post-hoc
  reason (obvious in hindsight): decode reads only top-8/128 experts
  (~24 MB/tok); shrinking expert bytes 2.2x cuts ~5% of per-token traffic.
  The 12.8 GB -> 5.74 GB memory saving stands (B-P5 class); the SPEED story
  for ternary lives in PREFILL (grouped path reads all routed experts) —
  t2 grouped-prefill kernels not built yet.

## Campaign V (verification probe, 2026-07-14): TLA+ model of the CACHEMODE-1
## ring protocol

Prior-art check: generic circular-buffer TLA+ specs are classic tutorial
material; NO formal spec / model check of an LLM KV-cache protocol (sliding
window + chunked prefill + slot position recovery) found. The niche is open.

Protocol under check (attnf32.wgsl CACHEMODE 1 + headprep + engine):
write slot = pos % RING, RING = W+512, chunk C = MPRE = 512, all chunk writes
precede its reads; reader recovers ps = maxW-((maxW+RING-t)%RING), dead iff
ps > qp or ps+W <= qp. Paper analysis: worst live distance = maxW-(qp-W+1)
with qp = chunk base -> W+C-2, so safety needs RING >= W+C-1.

- **V-P1 (soundness)**: for EVERY ring size, a live (non-dead) slot's
  recovered position equals its true content — clobber manifests ONLY as
  silent context loss (needle degradation), never as reading a wrong K/V as
  position p. (conf 75%)
- **V-P2 (completeness bound)**: the window is fully served iff
  RING >= W+C-1. Current RING = W+C is therefore safe with exactly ONE slot
  of spare margin: raising MPRE past 513 (or any second in-flight writer)
  silently breaks it. TLC finds the loss at RING = W+C-2 and proves the
  small-instance bound tight. (conf 70%)

### Campaign V verdicts (2026-07-14, TLC): BOTH HIT
Sweep (NCHUNKS=4, GEN=6): boundary exactly RING = W+C-1 in all three
(W,C) instances — (4,3): fail@5/pass@6; (6,5): fail@9/pass@10; (5,2):
fail@5/pass@6. SoundRec never violated anywhere (every failure is
Complete) => V-P1 confirmed: clobber is always SILENT CONTEXT LOSS,
never a stale K/V misread — matching the observed symptom class
(needle degradation, no garbage).
Production constants (W=1024, C=MPRE=512): RING=1536 (shipped) PASS,
1535 (minimum) PASS, 1534 FAIL => V-P2 confirmed, bound tight.
**Actionable: the shipped ring has exactly ONE slot of slack. Raising
MPRE past 513 without growing the ring breaks the window silently.**
Guard added value: specs/KVRing.tla + specs/sweep.sh re-check in ~30s;
run after any change to MPRE / window / ring / recovery formula.

### Campaign V, part 2 (2026-07-15): the checker toolchain is live

- **wgsl-check** (hesper `lake exe wgsl-check`, pure Lean): static
  write-bounds checker for WGSL compute kernels. Detects the 54a2a60 race
  class (roundup-grid unguarded stores clamp-writing the last element) from
  kernel source + dispatch manifest, incl. runtime guards via traced params
  values. Validated on THIS engine's full trace (1464 ops -> 144 unique
  dispatches, 96 kernels): **0 FAIL, 9 WARN, zero false positives** after
  six precision fixes earned on real kernels (dead template branches,
  loop-bound semantics, builtin-component refinement, lazy lets, else-if
  chain negation, taint tracking).
- The 9 WARNs are exactly the data-dependent stores: expgroup counting sort
  (chunkExp/chunkEnt) and grouped-MoE gather (y/yh). The expgroup one is now
  DISCHARGED by proof: hesper `specs/ChunkCap.lean` shows cap 640 =
  the provable bound sum ceil(n_e/8) <= T/8 + E (T=MPRE*K=4096, E=128) for
  ALL routings (attainable max 624, slack 16). Constraint made explicit:
  **raising MPRE*K past 4096 or experts past 128 overflows chunkExp
  silently.**
- Workflow: run the resident `{"mode":"trace"}` cmd once, then
  `curl :8877/check` (serve.py /check endpoint) — converts the trace
  (scripts/trace2check.py) and runs the checker, ~seconds.
- WMMA store-extent checking deferred WITH REASON: this engine's kernels
  stage subgroupMatrixStore through workgroup scratch and write storage via
  ordinary guarded stores (the safe pattern) — zero storage-target WMMA
  stores in the trace, nothing to validate the machinery against.

## Campaign V part 3 pre-registration (2026-07-15): mutation study — does the
## checker get you to the bug FASTER than expectation matching?

Design: inject mechanical mutations into copies of the traced production
kernels (96 kernels, real manifest), classes: (G) delete a store guard,
(R) round a grid dim up past the logical count, (S) stride off-by-one in a
store index, (B) shrink a binding's byte size, (N) numeric-only control
(perturb a constant in the VALUE computation, memory-safe). Measure:
1. wgsl-check detection + line-accuracy per class,
2. golden-gate detection across R repeated engine runs with the mutated
   kernel actually running (false-pass rate of expectation matching),
3. time-to-line: checker (automatic) vs symptom-driven manual localization.

- **V-P4**: checker detects ≥ 80% of G/R/S/B while flagging 0% of N (the
  class boundary is the claim, not omniscience). (conf 70%)
- **V-P5**: golden false-pass: for ≥ 1/3 of DETECTABLE race-class mutations,
  a single gate run PASSES at least once in R=5 (non-determinism makes
  expectation matching unreliable exactly where the checker is strong).
  (conf 55%)
- **V-P6 (novelty edge)**: ≥ 1 mutation class exists where detection
  REQUIRES the manifest context (real grid/sizes/params) — i.e. the same
  kernel text is safe under one dispatch and broken under another, which
  kernel-only tools (GPUVerify-class) cannot decide. (conf 85% — B is
  constructed to be this)
Qualitative companion: the DG->JS port diary (every /check FAIL caught
before first run, vs bugs that survived to runtime debugging).
