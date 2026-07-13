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
