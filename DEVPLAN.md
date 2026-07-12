# e4b-webgpu — Gemma-4 E4B WebGPU engine (webml replication study)

**What this is.** A replication of webml-community/gemma-4-webgpu-kernels (E2B,
~250 tok/s, authored by Fable 5 in a reported ~30 min) on **Gemma-4 E4B**, developed
by the method prescribed in hesper's post-mortem
(`hesper/docs/ANATOMY_OF_DECODE_REPORT.md` §6 + Appendix A): kernels as data,
seconds-TAT loop, golden gates, cold-honest timing, predictions registered before
measurement. **The author-time clock is part of the experiment.**

Session cycle (same as hesper): read this file → work a milestone → update state +
decision log + author-time → commit code and DEVPLAN together. ★ = user review gate.

## Pre-registered predictions (2026-07-07, BEFORE any engine code)

Byte accounting from the actual safetensors header (`NOTES/qat-format.md`):
per-token decode reads ≈ **2.20 GB** (42 layers × ~47.2 MB + lm_head 169 MB +
per_layer_model_projection 55 MB; the 705 MB PLE table is row-indexed, towers unused).

| scenario | effective BW | ms/token | tok/s |
|---|---|---|---|
| absolute floor (peak 546 GB/s) | 546 | 4.0 | 248 |
| webml-E2B-class kernels | 287 | 7.7 | **~130** |
| llama.cpp-class kernels | 200 | 11.0 | ~91 |
| hesper-E2B-class kernels | 186 | 11.8 | ~85 |

- **P1**: the webml swap test (their unmodified engine + E4B repo id) either fails on
  a loader assumption or runs at 100–140 tok/s. Registered before running it.
- **P2**: llama.cpp with an E4B QAT GGUF lands 60–85 tok/s on this box (E2B was
  147 t/s at 1.26 GB/token; E4B reads ~2.2 GB → scale ≈ 147×1.26/2.2 ≈ 84, minus
  per-layer-embedding overhead llama pays too).
- **P3 (success bar)**: our scratch engine ≥ llama.cpp E4B (must), 100–130 tok/s
  (stretch = webml-class effective BW).
- **P4 (the method claim)**: bring-up to token-exact in ≤ 1 author-day; the whole
  study in ≤ 2–3 author-days — an order less than the hesper E2B campaign.

## Milestones

| # | goal | gate |
|---|---|---|
| M0 | repo + this pre-registration + format & reading notes | ★ |
| M1 | baselines: webml swap test, llama.cpp E4B (llama-bench tg64, cool) | numbers recorded |
| M2 | harness (serve+collector+headless chrome) + goldens (transformers CPU, 3 fixed prompts) | goldens reproduce transformers greedy |
| M3 | scratch bring-up: loader + naive kernels → **greedy token match vs oracle** | token-exact ×3 prompts |
| M4 | optimize: per-class budget → webml-pattern fusion → cold timing | ≥ llama.cpp E4B; stretch 100–130 t/s |
| M5 | record: author-time, 3-way table, Replication section in hesper report | ★ |

**Fallback (user decision, recorded):** if the scratch path stalls, switch to the
swap pattern (webml unmodified + E4B repo id) and conclude as a measurement study.

## M1 results (2026-07-07, cool box, serial runs)

| baseline | result | vs prediction |
|---|---|---|
| llama.cpp E4B QAT q4_0 (llama-bench tg64, r=3) | **102.4 ± 0.3 tok/s** (9.76 ms/token) | **P2 MISSED** (predicted 60–85 — llama is faster than byte-scaling suggested; E4B per-token GGUF bytes likely below the naive estimate, note for M4 analysis) |
| webml UNMODIFIED + E4B repo id (headless Chrome, 64 tok, steady median) | **123.5 tok/s** (8.10 ms/token), coherent poem, load 95.5 s incl. 3.4 GB download | **P1 HELD** (the "runs at 100–140" branch). Their engine is config-driven enough to run a 2× model from a multimodal checkpoint UNMODIFIED |

Effective BW of webml-E4B: 2.20 GB / 8.10 ms ≈ **272 GB/s** — consistent with their
E2B 287 GB/s class (validates our byte accounting AND their kernel generality).

**Targets locked (per the pre-registered rule):** must-beat = llama.cpp **102.4**;
stretch = webml-E4B **123.5** (the oracle now exists on this box).

## M2–M3 results (2026-07-07)

**M2**: harness (range-serving dev server, headless-chrome runner, hot-reload kernels);
goldens from transformers f32-CPU on the same checkpoint (3 prompts, greedy ids, step0
logits, all-43 layer fingerprints). Dequant + SRQ semantics confirmed from
`transformers/integrations/gemma_quant.py`; full text-decoder spec verified first-hand
against `modeling_gemma4.py` (NOTES/arch-spec.md).

**M3 — bring-up: PASS on first execution.** The scratch engine (9 WGSL kernel files,
~500 lines of JS) produced, on its very first run, token-exact agreement with the
oracle on prompt 1 (full sequence incl. EOS, "Paris"). Layer-bisect: embedding and
layer 0 bit-exact; all 42 layer meanAbs ratios ≈ 0.99–1.005.

**Finding (gate amendment): token-exactness vs a CPU oracle is UNATTAINABLE for SRQ
checkpoints, by mechanism.** Static-range activation quantization snaps activations to
per-layer grids (steps 0.3–0.98); any 1-ulp cross-implementation difference at a grid
boundary becomes a FULL quantization step. The drift stays bounded (SRQ re-snaps each
layer) but flips near-tie tokens. Evidence that the engine is nonetheless correct:
prompt 0's divergent output is **word-for-word identical to the unmodified webml
engine's E4B output** ("A canvas vast of shifting blue…") — two independent GPU
implementations converge with each other; prompt 2 is semantically identical to the
oracle (correct Rayleigh explanation). Amended M3 gate: (a) ≥1 prompt token-exact vs
CPU oracle ✓, (b) cross-engine agreement with webml-E4B ✓, (c) layer-path ratios
≈1.000 ✓, (d) coherent/semantically-equal text on all prompts ✓.

Naive-engine speed: 65–120 ms/token (~15 tok/s) — the M4 starting point (8× to the
webml-E4B stretch target of 8.1 ms).

## M4 results (2026-07-07)

**Final: 11.11 ms/token = 90.0 tok/s (cool box, n=64 pipelined decode; 198 GB/s
effective on 2.20 GB/token).** Ladder: 65-120 (naive M3) → 20.8 (srq8 int8 pre-quant
+ cooperative matvec + GPU argmax) → 18.8 (vectorized loads, split-K attention) →
15.5 (matvec4: 2-row-interleaved subgroup matvec — pure-kernel 484 GB/s = 89% of the
M4 Max 546 peak) → 14.0 (fusions: rmssrq/rmsacc/geglusrq/plemulsrq + xq3 regions +
grouped qkv/gate-up concat matvecs) → 13.2 (headprep, attention-epilogue SRQ,
layer-boundary fusion) → 13.0 (1-dispatch attention, plegatemv, matvecgu2) →
11.4 (GPU-feedback pipelined decode: feedtok + token ring, zero CPU sync in loop) →
**11.11** (lm_head 64-row tiles, 251 GB/s) → **10.92 = 91.6 tok/s** (activation
loads/unpacks hoisted out of the dot helper — shared across both row
accumulations; gu −7%, down unchanged, so the no-CSE hypothesis was only
PARTIALLY right and the ~335 GB/s int4 cap is mostly elsewhere — recorded).

**vs targets: must-beat llama.cpp 102.4 tok/s — NOT reached (88%). Stretch
webml-E4B 123.5 — not reached (73%).** Honest gaps, measured not guessed:

1. **Serialized-dispatch cost model** (the load-bearing finding): a fenced dispatch
   costs `bytes/streaming-rate + ~9-33 µs fixed ramp/drain`. Decode is a fully
   linear dependency chain (~11 dispatches/layer × 42), so fences are ~40% of the
   wall. webml's 8.1 ms fits the same model with ~316 ops — op count is the lever,
   exactly as the hesper report's fat-kernel prescription said.
2. **int4 matvec streams at ~335 GB/s vs 484 for f32** — the int8-activation
   second load stream (2× weight bytes in the LSU) + unpack ALU. R2-interleave is
   the local optimum: R4/serial-rows/shared-staging all measured WORSE (register
   spill / no LSU relief). llama.cpp's native-Metal simdgroup kernels do ~400+.
3. Rejected with data: matvecgu v1 (8 accs → spill, 92 GB/s), SER=2 serial rows
   (+8%), shared xq staging (+3%), R4 accumulator array (273 GB/s), dual xq
   streams in matvecg (−25%, load-CSE pathology — hesper class).

Remaining levers (unexploited): f16 per_layer_model_projection (−0.12 ms),
attention V-loop parallelism at long seq, int4 unpack ALU reduction, batch-2
streams (fence overlap — but breaks single-stream comparability).

**Correctness discipline held**: every step re-gated (p1 token-exact incl. EOS;
p0/p2 near-tie SRQ-grid flips only, texts coherent and semantically equal —
p0 "A restless heart of endless blue…", p2 correct Rayleigh). Layer-bisect +
xqdiff differential mode caught 2 real bugs mid-M4: a lost-edit (boundary fusion
never dispatched → stale xq3) and a pleProj 10× under-dispatch that the token
gate ALONE had masked — gate condition (c) ratios are not optional.

## M4b results (2026-07-07 afternoon): 10.92 → 10.06 ms/token = 99.4 tok/s (97% of llama.cpp)

**The unlock came from READING THE REFERENCE (webml kernels in hesper
refs/webml-gemma4/wgsl), not from more guessing**: our int4 dequant chain
(unpack4xU8 → vec4i → vec4f → −8) is a Tint POLYFILL costing ~15 µs per 13 MB
matvec; webml uses `unpack4x8unorm`/`unpack4x8snorm` — NATIVE Metal single
instructions — with the /255·/127 folds moved into the output scale and the −8
zero-point deferred as `−8·Σq` (producer emits Σq; exact: verified vs integer
truth in a JS row-probe, err 1.5e-6). Ported to qkv (matvecg) + gate/up
(matvecgu): gu 3.47→2.55 ms (60.7 µs = 432 GB/s in-graph). Ladder this leg:
10.92 → 10.70 (single-dispatch attention + plegatemv + pleprep, matvecgu2) →
10.09 (unorm/snorm qkv+gu) → **10.06**.

**Two harness traps found (cost ~1.5 h of misattributed debugging):**
1. `--enable-dawn-features=disable_robustness` (hesper-proven lever, −0.3 ms)
   SILENTLY CORRUPTS this engine — some access relies on robustness clamping.
   Every flag flip must re-run the gate; a bench-only check let the corruption
   masquerade as a numerics bug in the (innocent) unorm port for an hour.
2. Chrome's persistent-profile disk cache serves STALE .js modules on
   same-second edits (the earlier "lost edit" mystery) — server now sends
   Cache-Control: no-store and the runner uses --disk-cache-size=1.

**Open (isolated, reproducible)**: matvec4-unorm for down/o with atomic-Σq
producers — A/B shows down-unorm alone → bisect ratio 0.53 (≈2× smell:
suspect the atomic Σ or its zeroing order). Row-probe method ready. Landing
this ≈ −0.6 ms → ~9.4 ms ≈ 106 tok/s (beats llama.cpp); then lm_head int2-unorm
(−0.2) and attention/fence trims → ~9.0. webml-SOTA 8.1 additionally needs
their op schedule (~316 ops vs our ~460).

## M4c results (2026-07-07): **8.05 ms/token = 124.2 tok/s — SOTA on this box**

Beats BOTH baselines: llama.cpp 102.4 (121%) and the webml engine itself
(123.5 → 124.2, 8.10 → 8.05 ms). Cool-box, n=64 median, gate-clean
(p1 token-exact incl. EOS; p0/p2 near-tie SRQ flips, coherent).

Ladder this leg: 10.06 → 9.17 (down/o unorm — the "ratio 0.53" was a silent-
replace failure: finish() never got the ZP refold; a SYNTHETIC downtest with
JS-controlled xq/Σq pinned it in one run) → 8.93 (lm_head int2 native-unorm,
459 GB/s) → 8.31 (subgroup 2-barrier reductions in rmsaccsrq — the barrier
TREES were ~50-80 barriers per 1-WG dispatch!) → 8.15 (same in attention1/
headprep/rmssrq/rmsacc + DT=1 attention + V t-partition) → **8.05**
(per_layer_model_projection bf16→f16, native unpack2x16float, 110→55 MB).

Effective BW 273 GB/s (= webml's number). Biggest remaining classes if anyone
wants more: gu 2.47ms (445 GB/s in-graph, near the int4 ceiling), attention
1.3ms (8-WG latency floor at short seq), fences (~470 dispatches).

## Author-time log

| span | wall clock | what |
|---|---|---|
| start | 2026-07-07 09:41 | M0 begun (repo, header analysis, notes) |
| ~10:05 | +24 min | M0 done (★approved), downloads started |
| ~10:35 | +30 min | M1 done: llama.cpp 102.4, webml-swap 123.5 (P1 held, P2 missed) |
| ~11:10 | +35 min | M2 done: harness + goldens + dequant/arch spec verified first-hand |
| ~12:00 | +50 min | M3 done: scratch engine token-exact on first run (p1); SRQ-grid finding; ~15 tok/s naive |
| 11:27 | +70 min | M4 done at 90 tok/s (15→90, 6×); must-beat NOT reached (88% of llama.cpp); cost model + rejected-experiments log above. **Total M0→M4 = 1h46 wall.** |
| ~11:45 | +15 min | M4 addendum: hoisted-activation dot → 10.92 ms = 91.6 tok/s final. User note recorded: this replication developed FASTER than hesper itself — the compact-surface/seconds-TAT effect, observed live. |

## Decision log

| date | decision | basis |
|---|---|---|
| 2026-07-07 | New minimal repo; scratch engine with webml as reference reading; swap = baseline/oracle + fallback | user; hesper report P2 (context compactness), principle 7 |
| afternoon | ~2.5 h | M4b: reference-read unlock (native unorm/snorm unpack) → 99.4 tok/s; 2 harness traps documented |
| evening | ~1.5 h | M4c: SOTA — 124.2 tok/s (webml 123.5, llama.cpp 102.4). Levers: synthetic-test debugging, native unorm everywhere, subgroup reductions, f16 projection |

---

# Campaign 2: gemma-4-26B-A4B (MoE) — same method, second replication

Started 2026-07-08 12:33 (GGUF download kicked off). User: 「次はgemma4 26b a4bで試そうか。
同様に計測して記録を残しましょう。」

## M0 — recon (verified first-hand) + pre-registered predictions

**Checkpoints**: NO qat-mobile-transformers exists for 26B-A4B → the SRQ format does
not apply. Available: `google/gemma-4-26B-A4B-it-qat-q4_0-gguf` (14.44 GB, single
file) and `-qat-q4_0-unquantized` (bf16 ~53 GB). **Decision: load the GGUF q4_0
DIRECTLY** — bit-identical weights to llama.cpp ⇒ clean cross-engine gate, and with
f32 activations (no SRQ grids) token-exact agreement with llama.cpp is IN PRINCIPLE
achievable, unlike Campaign 1.

**Arch (from config + transformers 5.13 modeling_gemma4.py, read first-hand):**
hidden 2816, 30 layers (full at 5,11,17,23,29), 16 q-heads; sliding: 8 kv-heads ×
head_dim 256, window 1024, θ=10k; full: 2 kv-heads × 512, proportional RoPE
(factor 0.25, θ=1M), **k_eq_v: full layers have NO v_proj — V = v_norm(k_proj out)**
(25% less attn weight read there). No PLE, no KV-sharing, tied embeddings
(lm_head = embed, q4_0, 262144×2816 ≈ 415 MB/token — the biggest single read).
Per layer: dense MLP (inter 2112) AND a parallel MoE branch (128 experts, top-8,
inter 704, fused gate_up [128,1408,2816] + down [128,2816,704]); router input =
the PRE-norm residual; combine = postFfn1(mlp) + postFfn2(moe), then postFfnNorm,
+res, ×layer_scalar. Router: softmax → top-8 → renormalize → × per_expert_scale.

**Per-token active read (q4_0)**: attn+dense+8 experts+lm_head ≈ **2.2 GB — same as
E4B** ⇒ at our proven 273 GB/s the physics ceiling is ~120 tok/s.

**Pre-registered predictions (before ANY measurement):**
- **P1**: the unmodified webml engine FAILS to run 26B-A4B (no mobile checkpoint;
  no MoE kernels). The swap test is run anyway to record the failure mode.
- **P2**: llama.cpp llama-bench tg64 (q4_0, this box) lands at **55–80 tok/s**
  (2.2 GB/token at their ~225 GB/s eff, minus MoE mul_mat_id inefficiency —
  hesper's DiffusionGemma data point: llama.cpp MoE ran at 27% MFU).
- **P3**: our engine beats llama.cpp (must); **≥100 tok/s** stretch. Sub-prediction:
  greedy tokens MATCH llama.cpp token-exactly on ≥1 prompt at n=24 (no SRQ grids).
- **P4**: author time — oracle-agreeing bring-up ≤ 1 day (GGUF parser + MoE +
  k_eq_v are new); competitive (P3-must) ≤ 2 days total.

**Method adaptations pre-committed:**
1. **Resident-tab harness** (the 26B TAT killer identified in §8 discussion):
   the page stays alive holding 14.4 GB on GPU; the dev server gets a command
   endpoint; kernels are re-fetched and pipelines rebuilt per iteration — gate
   and bench runs without weight reloads.
2. Oracle = llama.cpp greedy tokens (primary) + llama-eval-callback per-layer
   fingerprints (condition (c)); transformers-CPU oracle is NOT feasible at f32
   (104 GB) — recorded as a method limit.
3. q4_0 kernels: block-32 f16 scales → the unorm trick refactors per block:
   Σ_b d_b·(Σ v·x − 8·Σ_b x) needs per-block activation sums (88 f32 per vector),
   produced by the norm kernels like Campaign 1's Σq.
4. Same repo, same gate discipline; E4B gate must STAY green (regression check)
   behind a model switch.

## Campaign 2 — M1 results (2026-07-08)

**llama.cpp (fork w/ gemma4, build 73d820a, Metal): tg64 = 112.51 ± 0.45 tok/s
(8.89 ms/token).** → must-beat = 112.5. **P2 (55–80) MISSED LOW AGAIN** — same
bias direction as Campaign 1's P2: I keep underestimating llama.cpp's Metal MoE
path (prior was anchored on hesper's DiffusionGemma mul_mat_id 27%-MFU data
point; mainline gemma4 MoE is clearly better). Two-for-two: predictions about
MY OWN side hold; predictions about the competitor's efficiency run low.

Physics check: ~2.0–2.2 GB active/token ⇒ llama.cpp is at ~225–250 GB/s eff
(consistent with their E4B number); our proven 273 GB/s eff ⇒ ~7.3–8.0 ms
≈ **125–137 tok/s potential** → stretch target: ≥125 tok/s.

**GGUF ground truth (inspected first-hand, gguf_inspect.py):** 658 tensors:
Q4_0 ×265 (all matmuls incl. fused MoE 3D tensors: gate_up_exps [2816,1408,128],
down_exps [704,2816,128] + per-expert F32 scale [128]), F32 ×392 (norms, router
[2816,128] + router input scale [2816], rope_freqs[256], layer_output_scale),
**Q6_K ×1 = token_embd (2816×262144, TIED lm_head — 484 MB/token, the single
biggest read; needs a Q6_K kernel).** Full layers: attn_k only 1024 wide + NO
attn_v (k_eq_v). data_start=15821792, align 32.

## Campaign 2 — M2+M3 results (2026-07-08)

**M2**: goldens = llama.cpp greedy (llama-server, temp 0, 3 prompts × 24 ids,
return_tokens). Resident-tab harness built (serve.py /cmd long-poll + cmd.sh +
resident-a4b.html): **14.4 GB loads in 30 s, then kernel edits hot-reload per
command with zero weight reloads** — the TAT adaptation works as designed.

**M3 — bring-up: GATE PASS, all 3 prompts TOKEN-EXACT vs llama.cpp** (first
substantive run; one stale-epilogue kernel fix in between — caught in minutes by
the resident loop). **P3's sub-prediction confirmed and exceeded: without SRQ
grids, cross-engine token-exactness is achievable — 3/3, not just ≥1.** The
per-layer stats mode showed healthy activations through the k_eq_v full layers
on the first look. New engine surface: gguf.js (82 lines), engine-a4b.js
(~330), 6 new kernels (q40mv with MoE expert indirection via a GPU-side topk
buffer — no CPU readback in the MoE path; q6k embed+lm_head; router with
in-kernel top-8; slot geglu; moe combine; f32 attention variant).

Naive speed: **45.1 ms/token = 22.2 tok/s** (M4 start; llama.cpp 112.5;
physics ceiling ~125-137). Wall clock M0→M3: **12:33 → 13:03 = 30 min.**

## Campaign 2 — M4 progress log (2026-07-08 afternoon)

Ladder (all steps GATE PASS = 3/3 prompts token-exact vs llama.cpp, verified
via the resident tab per change):
- 45.1 ms (naive M3)
- → 18.1 ms: **Q6_K lm_head vectorized** (q = ql + 16·qh2 decomposition — BOTH
  planes decode via native unpack4x8unorm with a single /255 refold; −32 zero
  point deferred through per-16-elem group sums computed in-WG; 10.8→2.5 ms,
  195 GB/s) + **router split** (was a 154 µs single-WG serial monster, 18% of
  the token! → rmsnorm-reuse + matvec2f + tiny top-8 kernel ≈ 25 µs).
- → 14.6 ms: rms3 (one reduction, three scaled outputs for ffn/router/pre-ffw-2)
  + a4btail (moe-combine + postFfw1/2 + add + post-norm + residual + layer
  scalar = 5 dispatches → 1).
- → **13.4 ms = 74.6 tok/s**: qkv concat (k_eq_v layers concat q+k only),
  gate+up concat, rmsacc3 (post-attn residual fused with the triple norm).
  ~14 dispatches/layer (from 24 naive).

Traps hit: 9-storage-binding kernel exceeded the DEFAULT
maxStorageBuffersPerShaderStage=8 (raise in requiredLimits); concat scale
planes must pad to 4-byte multiples for writeBuffer.

Current budget (serialized): guExps 1.40 (338 GB/s ✓near-cap), lm_head 2.46
(195 GB/s — next: 64-row tile repack, campaign-1 pattern), downExps 0.96,
dense gu 1.05→concat'd, attn 0.38+0.12, top8 0.46 (fence-bound smalls).
Remaining to must-beat 8.89 ms: lm_head tiling (−1.2), small-op fences,
downExps short-row shape.

## Campaign 2 — M4 status at end of leg (2026-07-08)

Ladder continued (every step GATE PASS = 3/3 token-exact vs llama.cpp):
13.4 → 12.65 (lm_head TILE=128) → **11.58 ms = 86.3 tok/s** (Q6_K planes
TILE-TRANSPOSED at load — 128-row tiles, fully coalesced reads; lm_head
2.06→1.25 ms = 388 GB/s, same layout serves the embed row-gather — one
TILE-vs-repack mismatch caught by the gate immediately; + a4btail boundary
fusion: the layer tail now also emits the NEXT layer's attn-normed input).

REJECTED with data: router2 single-WG fused scores+top8 (one WG reading 1.4 MB
= latency disaster, 12.65→16.9 ms — same lesson class as campaign 1's 8-acc
fusion: don't starve the GPU to save a dispatch).

**vs targets: 86.3 / must-beat 112.5 (77%) / stretch ~125.** Remaining levers,
in order: (1) geglu fused into the gu matvec epilogues (dense + experts,
campaign-1 matvecgu2 pattern, f32 out = no pack race), (2) downExps short-row
shape (IN=704 → 22 blocks < 32 lanes: 1/3 idle, 183 GB/s), (3) ~13
dispatches/layer × ~6 µs fences ≈ 2.3 ms wall-vs-profile gap — more boundary
fusions, (4) thermal-clean re-measure (box ran hot all afternoon; profile sum
stayed 9.4 ms while wall wobbled 11.6-14.9).

Author time this leg (M0 12:33 → here): ~3.5 h wall including the 112.5-tok/s
llama.cpp rebuild and all downloads.

## Campaign 2 — M4 leg 2 (2026-07-08 evening)

- q40gu: gate+up+geglu fused (dense + experts; f32 out, campaign-1 matvecgu2
  shape) — best observed **11.47 ms = 87.2 tok/s**, GATE PASS (token-exact).
- REJECTED with data: (1) top8 absorbed into the guExps prologue (redundant
  per-WG top-8 ≈ the saved fence; neutral 11.47→11.69); (2) attn2f — headprep
  absorbed into attention at DT=1 (campaign-1 att2 redux: neutral speed AND
  broke the gate; not worth debugging a neutral lever — reverted, file kept).
- **Measurement honesty**: the same binary measures 11.5–13.3 ms depending on
  box state (swap 0, memory 90% free, no strays — pure clock/thermal variance;
  fresh-tab vs aged-tab excluded by A/B). Serialized profile is stable at
  ~9.2 ms throughout. Final Campaign-2 numbers need the campaign-1 protocol:
  cold box, single shot, first run of the day.

Status vs targets: best 87.2 / must-beat 112.5 (78%). Ranked remaining:
downExps 183 GB/s (short rows), dense-down 172 GB/s (small dispatch),
~12 fences/layer ≈ 2.2 ms gap, lm_head 388 GB/s ceiling-close.

## Analysis: cooperative matrix multiply (subgroup-matrix) — where it does and does not apply

**Availability (measured)**: this Chrome's adapter exposes
`chromium-experimental-subgroup-matrix` (plus `shader-f16`) under
--enable-unsafe-webgpu. The API also worked on hesper's own Dawn/Metal
(DiffusionGemma reg kernels: 39% MFU / 6 TFLOPs on the matrix units).

**Decode (M=1, the tg64 benchmark): does NOT help — wrong regime.**
Per-token work is ~8 GFLOP against 2.2 GB of weight reads: at 14 TFLOPs f32
the ALU cost is ~0.6 ms of an 11.5 ms token — decode is DRAM-bound, and matrix
units raise FLOP throughput, not bandwidth. Worse, subgroup-matrix takes
f16/f32 operands, so q4_0 weights would need an f16-staged copy — DOUBLING the
bytes read per token (the hesper-DG f16-predequant recipe paid there precisely
because diffusion decodes M=277 tokens per step; autoregressive M=1 is the
opposite regime). The remaining 87→112 gap lives in bandwidth shapes
(downExps/dense-down), fences, and box-state variance — not FLOPs.

**Where it WILL pay in this repo:**
1. **Prefill** — currently token-by-token (M=1 loop): a 2000-token prompt costs
   ~23 s. A GEMM prefill path (M = prompt length) is compute-bound → exactly
   the subgroup-matrix use case. Not measured by tg64, but required for a
   usable demo/Space. This is the natural Campaign-2 M6.
2. Speculative / multi-token decode (M=4–8), if ever.
3. (Bonus, unrelated to coop-matrix): `shader-f16` enables an f16 KV cache —
   halves attention cache traffic at long context.

## Campaign 2 — shader-f16 deployed: f16 KV cache (2026-07-08)

Answering 「shader-f16 も使ってない?」— it wasn't (the f16 SCALES were already
decoded via core `unpack2x16float`, no extension needed). Now deployed where it
genuinely pays: **the KV caches are stored f16** (headprep writes f16,
attention reads convert). GATE PASS — token-exactness vs llama.cpp SURVIVES,
which makes sense: llama.cpp's own KV is f16, so this matches THEIR precision
rather than degrading below it. Cache memory halved; attention cache traffic
halved (at context 640 that is ~286→143 MB/token ≈ −0.5 ms; at bench length
~100 it is inside the box-variance noise — the win grows with context).
Remaining f16 candidates rejected for M=1 decode: f16 matvec arithmetic (ALU
is not the wall; precision risk) and f16 activations (broadcast-cached, no
traffic to save).

## Campaign 2 — M4 leg 3 (2026-07-08 night): 12.2 → 11.11 ms = 90.0 tok/s

- **q40moedown**: MoE down re-designed slot-COMBINED — each WG owns 2 output
  rows and iterates all 8 slots (expert rows via topk, per-slot x), emitting the
  topkW-weighted sum directly. Kills the downSlots buffer, the combine work in
  the tail, and the 69%-lane-idle shape. −1.1 ms wall (adjacent A/B).
- **routertop**: router scores + top-8 in ONE dispatch via the decoupled
  last-WG pattern (atomic completion counter; scores atomicStore'd for cross-WG
  coherency; Tint uniformity satisfied by hoisting the barrier out of the
  flagged branch). −1 dispatch/layer.
- **Submit batching**: decodeChunk now encodes 16 tokens per command encoder
  using a params-ring (positions known ahead; token ids flow GPU-side through
  feedTok) — one queue.submit per 16 tokens instead of per token. −0.8 ms.
- Dispatches/layer: 24 (naive) → 11. GATE PASS (3/3 token-exact) throughout.
- **Measurement protocol note**: the FIRST bench after a resident restart is
  polluted by lazy Metal pipeline compilation — always measure from the second
  run. Warm-band best now 11.11 ms; serialized profile ≈ 9.5 ms.

Remaining: cold-box truth run (morning protocol); kernel-sum floor ≈ 9.4 ms
means parity with llama.cpp (8.89) also needs ~0.5 ms of kernel wins
(lm_head 388→460 GB/s, qkv 268→330, moedown LSU shape) — all identified,
diminishing, honest.

## Campaign 2 — leg 3 close (2026-07-08 late night)

WG-templated q40mv (32/64/128 rows-per-WG shapes; Tint lessons: subgroup-index
-derived early returns break subgroupAdd uniformity — guard with a `valid` flag
instead of returning). WG=128 on qkv/o/down measured NEUTRAL (11.24 vs 11.11) —
kept at WG=32. A transient "104 tok/s" was a silent-edit artifact computing 1/4
of the rows — caught by the gate, as designed; the ASSERT-your-edit-anchors
lesson is now burned in twice.

**Box noise reached ±1.5 ms between adjacent identical runs (12.9 vs 11.1 on
the same binary) — optimization A/B below ~1 ms is impossible tonight.
Discipline: stop. Next session MUST open with the cold-box protocol run.**

Leg summary: 13.3-band → 11.11 best (moedown, routertop, 16-token submits).
Serialized kernel floor ≈ 9.4-9.6 ms. llama.cpp 8.89.

## Campaign 2 — M4 leg 4 (2026-07-09): f16 activations — profile 9.6→8.60 ms, wall ~10.6 = 94 tok/s

**Discipline change**: interactive-box wall noise (±1.5-2 ms) made wall A/B
useless → switched to the serialized-profile total (±0.1-0.2 ms resolution) as
the optimization metric; wall checked at milestones only.

- qkv WG=64 (307-322 GB/s), moedown WG=64: profile −0.3.
- **f16 activations end-to-end** (the x-LSU theory: matvec activation reads run
  2-8× the weight bytes through the LSU; halving them lifts the q4_0 kernels):
  gegluSlots, normed, moeIn, dense geglu, attnOut all stored f16 (producers
  write f16; q40mv gains an XF16 template; routerIn and the final norm stay f32
  for the router and Q6_K lm_head). **GATE STILL TOKEN-EXACT vs llama.cpp** —
  consistent with llama.cpp's own q8_0-quantized activations being COARSER than
  f16 on these paths. moedown 0.99→0.89, and the full set took the serialized
  kernel total BELOW llama.cpp's wall: 8.60 vs 8.89.
- Fourth silent-edit incident (unpx pattern mismatch → XF16 read a dummy buffer
  → "104 tok/s"-class garbage); the synthetic mvtest (JS-exact row recompute)
  pinned it in one run. EVERY kernel-edit python block now asserts its anchors.

Remaining gap = wall−profile ≈ 2.0 ms (submit/fence side): ~11 dispatches ×
30 layers; llama.cpp wall 8.89. Levers: further dispatch cuts, and the ambient
CPU contention (the measurement box runs an interactive session).

## Campaign 2 — leg 4 close (2026-07-09)

- 32-token encoders: saturated (≡16).
- **False-hazard finding**: dead read_write bindings fed with REAL buffers
  (A.mlpOut/A.tmp as dummy slots) made Dawn track phantom writes → false
  WAW/RAW edges serializing the dense and MoE branches. Fixed with dedicated
  never-aliased dummy buffers + read-only decls for unused x slots, and the
  two branches' dispatches interleaved (hazard-free neighbors can overlap).
  Profile-neutral (expected — profile serializes by construction); wall effect
  unmeasurable today: the box entered a contended mode (iTerm 37%, sysmond 26%
  — active interactive use) shifting walls +2 ms. GATE PASS.
- **Best clean-window wall this leg: 10.63 ms = 94.1 tok/s** (3 consistent runs
  at midday); serialized kernel total 8.54-8.73 ms — BELOW llama.cpp's 8.89 ms
  wall. The remaining ~2 ms is dispatch-boundary cost × ~330 dispatches + box
  contention — the §8 serialized-dispatch model, third confirmation.

Status: 94.1 best / must-beat 112.5 (84%). The kernel side is done to within
~0.3 ms of its ceiling; further gains need either fewer dispatches (arch floor
~11/layer reached) or a quiet box for honest walls.

## Campaign 2 — leg 5 (2026-07-09): plan-replay + gu merge — profile 8.40 ms, wall 10.56 = 94.7 tok/s

- **Token-plan record/replay**: each ring slot's ~340-dispatch encode is recorded
  once and replayed as flat [pipeline, bindGroup, x,y,z] tuples. The per-dispatch
  JS was ~2 ms under CPU contention — decode walls now stable at ~10.6 ms even
  on a busy interactive box (was bimodal 10.6/12.5).
- **gu merged dispatch** (dense z=0 + 8 expert slots z=1..8 in one grid): −0.11 ms.
- REJECTED with profile data: down+moedown union kernel (+0.17 ms — register
  pressure of the union body; unlike gu, the two down shapes share no benefit).
- State: serialized kernel total **8.40 ms** (llama.cpp WALL is 8.89); our wall
  10.56 = 94.7 tok/s (84%). The ~2.1 ms delta is GPU dispatch-boundary cost
  (~300 boundaries) — the §8 model; kernels are done to ~0.2 ms of ceiling.
  Next honest step: a genuinely idle-box wall (overnight window), then M5.

## Campaign 2 — measurement embargo + nightrun (2026-07-09 evening)

GPU contention discovered as the LAST noise channel: WindowServer (display
compositing, 39% CPU) shares the GPU — under it even the serialized-profile
totals inflate (8.40 → 9.4-9.6 ms). With CPU contention already defeated by
plan-replay, the remaining variance is display-GPU sharing: NO metric is
trustworthy while the user works the machine. feedTok folded into the embed
(TOKSRC template, gate-verified) — its measurement awaits the quiet window.

**nightrun.sh armed**: polls every 30 s for load<0.8 AND WindowServer<5%, then
runs the full protocol (fresh resident, warmup, 3 walls, profile, gate) into
harness/night.log. Best defensible numbers so far: profile 8.40 ms, wall 10.56
= 94.7 tok/s (llama.cpp 8.89 / 112.5).

## attn2f status (parked as ?attn2=1 opt-in)

Headprep-absorbed attention (~−0.15 ms candidate): embed/prep verified, layer
stats plausible-but-different; t==pos f16-parity fix applied but the gate still
diverges at token 0 (magnitude beyond summation-order rounding — undiagnosed).
Cost/benefit poor at 1.4% of wall — parked behind A4B_QUERY='?attn2=1'.
Default path (headprep + attnf32) re-verified GATE PASS.

---

# Campaign 2 — M6: batched prefill (started 2026-07-12)

Prefill is token-by-token today: a P-token prompt costs P × ~10.6 ms (20-token
gate prompts ≈ 210 ms; a 512-token prompt ≈ 5.4 s). Layer-wise batching removes
the serial-through-layers structure: all tokens advance one LAYER at a time —
matvecs widen to M columns (toward GEMM), K/V for the whole chunk lands before
the (causal) attention of that layer.

**Pre-registered predictions:**
- P5: a first, subgroup-matrix-FREE batched prefill (M-column loops on the
  existing q4_0 kernels) lands at **≥4× prefill throughput at M=20** and ≥6× at
  M=64 (utilization, amortized weight reads), with generation tokens after a
  batched prefill IDENTICAL to the token-by-token gate (the correctness bar).
- P6: adding subgroup-matrix (chromium-experimental) to the batched path buys a
  further ≥2× at M≥64 (compute-bound regime begins) — deferred until P5 holds.
- P7: author time — P5 within one focused leg (~2-3 h).

**Plan**: MTOK on activations (chunk buffers M×dim), M-looped q40 kernels,
causal chunk attention (score matrix in workgroup memory per (head, token)),
per-token router/MoE via grid, gate = batched-prefill → identical generation.

## Analysis: why we WON on E4B but TRAIL on A4B (the format-assist finding)

Byte accounting from the actual GGUF tensor tables settles it:

| | llama.cpp reads | we read | winner's edge |
|---|---|---|---|
| E4B | **2.68 GB/token** (q4_0 GGUF: Q6_K lm_head = 551 MB + q4_0 body) | **2.09 GB/token** (QAT-mobile: int2 lm_head = 168 MB + per-channel int4 body) | we read **28% fewer bytes** |
| A4B | 2.38 GB/token (q4_0 GGUF) | 2.38 GB/token (SAME file) | none — pure engine race |

Effective bandwidth (bytes/wall): E4B — llama.cpp **275 GB/s** vs ours 260;
A4B — llama.cpp **268 GB/s** vs ours 225.

**Conclusion: our engine was never faster per byte than llama.cpp. The E4B
victory came from the FORMAT — the QAT-mobile checkpoint (int2 tied lm_head,
per-channel int4) simply reads 28% less than the q4_0 GGUF llama.cpp had to
use. On A4B both engines read the identical file, which exposes the true
engine-side deficit: ~2 ms/token of WebGPU dispatch-boundary cost (~310
dispatches through Dawn validation + Metal fences) that llama.cpp's native
command encoding does not pay.** (webml-E2B's headline numbers also ride the
mobile-format byte advantage.)

Paths to A4B parity, in honesty order: (1) eliminate boundary cost (structural;
the §8 serialized-dispatch model, third confirmation), (2) a "compact variant"
data point — requantize the tied Q6_K embed (605 MB/token, 25% of all reads!)
to int4/int2 class — but that changes weights, forfeits token-exactness vs
llama.cpp, and llama.cpp could do the same; it is a model-config win, not an
engine win, and would be reported as a separate row, not a parity claim.

## The fencetest verdict + the webml k08 pattern (2026-07-12)

**「dispatch 境界税って本当?」— tested, and the honest answer is NUANCED:**
synthetic chains of 1000 trivial dispatches measure:
- same-buffer READ-MODIFY-WRITE chain: **87-156 µs/dispatch** (pathological)
- producer→consumer (ping-pong RAW): **4-5 µs**
- same-buffer WAW: 2.0 µs; independent: 1.9 µs
⇒ ordinary fences are nearly free; the catastrophe is RMW-on-one-buffer. Our
chain had THREE RMW sites × 30 layers (rmsacc3's hidden, tail's hidden,
headprep's in-place qkv) — all converted to split-in/out (hidden ping-pongs
A→B→A per layer; headprep writes a separate qPrep). GATE PASS; walls became
eerily stable (11.43 ± 0.02 under contention); the quiet-window verdict is
nightrun's.

**webml E2B kernel reading (per user direction) — the structural next step:**
- k08 = matvec + post-norm + residual in ONE dispatch via the last-WG pattern:
  matvec WGs atomicStore rows + ticket counter; the last WG re-reads through
  atomics and applies `hidden += RMSNorm(d)·w` (cross-WG visibility is only
  guaranteed through atomics — their comment says exactly this). This is
  their OprojNorm/DownNormAdd, and it removes our oMv→rmsacc3 and
  down/moedown→tail fences AND the 1-WG norm dispatches. Needs
  maxStorageBuffersPerShaderStage > 12 (concat norm-weight buffers) — planned
  as the next leg.
- k13 confirms subgroup-matrix is their PREFILL GEMM (M≥64, int8-code domain,
  f16 tiles, f32 accum, integer-exact) — validates M6/P6 as pre-registered.

## Campaign 2 — QUIET-WINDOW TRUTH (nightrun, 2026-07-12 16:37, load 0.51 / WindowServer 0%)

**Wall 10.52 ms = 95.0 tok/s** (runs 2-4: 10.60/10.52/10.55; run 1 = 12.20
compile-warmup, discarded per protocol). **Profile 8.36 ms. GATE PASS.**

Post-mortem of the gap (wall − profile = 2.16 ms over ~310 dispatches ≈ 7 µs
each): the RMW fix did NOT move the quiet-window wall (10.56 pre-fix ≈ 10.52
post) — the 87 µs RMW pathology of the 1-thread probe does not manifest at
real dispatch sizes; the honest per-dispatch cost is the ~4-5 µs RAW fence +
ramp, i.e. **境界税は本当だが ~5-7 µs/dispatch スケール** — matching webml's
economics (316 ops → same tax). Removing it requires fewer dispatches: the
webml k08 last-WG-merge pattern (matvec+norm+residual in one) is the scoped
next leg (−3-4 dispatches/layer ⇒ est. −0.6-0.8 ms ⇒ ~102 tok/s).

**A4B status vs targets: 95.0 / must-beat 112.5 = 84.4%** (llama.cpp reads the
SAME bytes here — see the format-assist analysis; E4B's win was byte-assisted).

## k08 transplant experiment: REJECTED with data (2026-07-12)

Implemented webml's k08 last-WG-merge (o-proj matvec + rmsacc3 epilogue in one
dispatch; WG=32 single-subgroup epilogue = barrier-free; workgroupUniformLoad
for the ticket flag — subgroupBroadcast is NOT Tint-provably uniform). Result:
**GATE PASS (math correct) but 11.5 → 15.7-17.7 ms — a 40-50% regression,
reproduced with per-layer pp buffers (not the cross-layer chain).** The
atomic-path cost (2816 atomicStores of the matvec output + 1408 counter
atomicAdds per dispatch) is pathological on Dawn/Metal for our shapes —
consistent with the fencetest RMW finding, and it does NOT transplant from
webml-E2B (open question: their H=2048/different economics, or Dawn version
differences). Both merge kernels kept in-tree as documented rejects
(omerge.wgsl; downMerge not attempted after the o-merge verdict).

Leg conclusion: dispatch floor stays ~10/layer; the honest A4B endpoint on
current WebGPU = quiet-window **10.52 ms = 95.0 tok/s** (84.4% of llama.cpp,
same-bytes comparison). Next value: M5 recording, then M6 prefill GEMM (k13
confirms webml's prefill recipe: subgroup-matrix, int8 codes, f16 tiles).

## P5 result (2026-07-12): batched prefill lands, gate exact — prediction MISSED honestly

Implementation (one leg): `q40mm.wgsl` (multi-column matvec: each WG loads its
2 rows' q4_0 blocks ONCE and reuses them across MCOLS=4 token columns; per-token
jb/subgroupAdd order identical to q40mv → bit-identical activations) + `BATCH`
templates on 7 kernels (q6k embed rows, headprep per-token pos/rows, attnf32
causal len=basePos+tok+1, rmsacc3/a4btail row offsets, routertop per-token
scores/ctr/topk, q40gu z=tok·(1+K)+slot, q40moedown z=tok). MoE stays
per-token (expert indirection unchanged). lm_head runs ONCE per prompt (the
step loop pays it per token). Chunks of MPRE=64.

- **Correctness: GATE2 PASS** — generation after batched prefill IDENTICAL to
  the token-by-token gate on all 3 prompts, first run. (The bit-identity design
  goal held: same kernels or same accumulation order throughout.)
- **Throughput: 2.26× at M=20** (8.19 → 3.62 ms/tok), **2.16× at M=64**
  (7.98 → 3.69). Decode-path regression gate: PASS (BATCH=0 folds to the old
  code).

**P5 verdict: prediction MISSED** (pre-registered ≥4× at M=20, ≥6× at M=64; got
~2.2×). The miss is diagnostic and was foreseeable from the byte budget: the
batched floor is M-INVARIANT (3.62 ≈ 3.69 ms/tok) — exactly the signature of
the un-amortized term. Per token the MoE reads 8 experts × (gate_up 2816×1408 +
down 704×2816) q4_0 ≈ 27 MB/layer × 30 ≈ 0.8 GB regardless of M; the amortized
parts (attn+dense ≈ 0.6 GB/tok at M=1) shrink 20-64×, and lm_head (0.6 GB)
drops out per-token — that predicts ≈ 0.85 GB/tok ≈ 2.3-2.4× — which is what
landed. ~3.7 ms/tok × 64 tok ≈ 51 GB / 236 ms ≈ 216 GB/s: the batched prefill
is STILL BW-bound, on expert weights.

**Consequence (P6 refined, same spirit):** the big prefill lever is not the
GEMM shape of the dense parts (already amortized) but **expert grouping**
(mul_mat_id): at M=64, 512 slot-draws hit ≤128 unique experts → grouped expert
reads shrink ~4× → ≈ 0.25 GB/tok ≈ ~1.2 ms/tok candidate. Subgroup-matrix (k13
recipe) then matters where compute becomes the wall. Order: group experts
first, then subgroup-matrix on the grouped GEMMs.

## P6 refined pre-registration (2026-07-12, after the P5 budget)

Prefill M=64 per-class budget (serialized 250 ms): MoE gate/up 96 + MoE down
41 (= 55%, un-amortized expert bytes) | q40mm matvecs qkv 46+9, o 26+11, dense
down 15 (= 107 ms, ~0.6 GB read once → ~6.5 GB/s = COMPUTE-bound, the matvec
shape is the wall) | everything else < 5.

Two levers, pre-registered:
- **P6a — subgroup-matrix GEMM** (chromium-experimental-subgroup-matrix; on the
  adapter here) replacing the batched q40mm sites (qkv / o / dense-down):
  k13-style 32M×64N×32K tiles, JIT-dequant q4_0 → tiles with the block scale
  folded, f32 accumulate. Predict: the 107 ms class → ≤ 35 ms; prefill M=64
  3.69 → ≤ 2.7 ms/tok. Correctness bar: gate2 generation still EXACT vs
  goldens (accumulation-order + one-rounding change; f32 tiles first = weight
  dequant stays exact, so the only diff vs q40mv is summation order).
- **P6b — expert grouping (mul_mat_id)**: GPU counting-sort of the M×8 slot
  draws per layer into per-expert chunks (MC=8 columns), gate/up (and later
  down) read each touched expert's weights ~once per layer instead of per
  token. Predict: MoE 138 → ~50 ms at M=64; combined with P6a ≤ 1.6 ms/tok at
  M=64 (≥ 5× vs tokenwise; ≥ 2.3× vs P5). Gate/up grouping is bit-identical
  per entry (same jb order); grouped DOWN changes the k-sum order → gate2 may
  flip near-ties, judged separately.

## P6a result (2026-07-12): subgroup-matrix GEMM — prediction HIT, gate exact

`q40sg.wgsl`: 32M×64N×32K tiles (one q4_0 block = one K-tile), WG=128 = 4
subgroups × (2×4) 8x8 f32 result mats, JIT-dequant B tiles with the block scale
folded, **f32 tiles** so the dequant stays exact — the only numeric change vs
q40mv is summation order. Feature `chromium-experimental-subgroup-matrix`
requested when present; `?sgm=0` falls back to q40mm.

- **gate2: PASS, all 3 prompts EXACT** — the accumulation-order change did not
  flip a single token (f32-tile design goal held).
- The q40mm class: 107 → **36.1 ms** (qkv 46+9 → 14.2+2.6, o 26+11 → 10.2+3.5,
  dense down 15 → 5.6). Predicted ≤ 35: **HIT** (36.1).
- Prefill M=64: 3.69 → **2.69 ms/tok** (predicted ≤ 2.7: **HIT**) = 2.97× vs
  tokenwise. M=20: 3.54 ms/tok (MoE-bound; single 32-row M-tile carries 12
  wasted rows).

Budget after P6a (M=64, 180 ms serialized): **MoE gate/up 98.2 + MoE down 41.1
= 77%** — the un-amortized expert reads are now cleanly the whole story → P6b.

## P6b result (2026-07-12): expert grouping — mixed verdict, honest miss on the byte model

Implementation: `expgroup.wgsl` (one-WG counting sort of the M×K draws into
MC=8-column per-expert chunk descriptors) + `q40gugrp.wgsl` (weights loaded
once per chunk, reused across the chunk's entries; per-entry math identical to
q40gu → bit-identical) + dense gate/up moved to a `q40sg` GEMM over the guCat
concat with a `geglub` epilogue + `q40downgrp/wacc` (grouped down, judged
separately).

- **Grouped gate/up: ACCEPTED** — 96 → 70 ms (first cut was 96 ms = NO win
  until sentinel columns were guarded to skip all work; the padding columns
  were doing 3× wasted compute).
- **Dense gate/up GEMM + geglu epilogue: ACCEPTED** — 26 → 9.6 + 3.9 ms.
- **Grouped down: REJECTED with data** — 43.5 ms (WG=64) / 49.6 ms (WG=128) vs
  41.3 ms per-token; reverted (kernel kept as documented reject).
- Gates: **GATE PASS + GATE2 PASS** throughout.

**Final M6 numbers (quiet-ish box, best-of-3):** prefill M=64 **2.35 ms/tok
(3.4× vs tokenwise 7.99)** ≈ 425 tok/s; M=20 **3.47 ms/tok (2.31×)** — the
3-prompt gate latency drops 160 → 69 ms.

**P6 combined prediction (≤1.6 ms/tok) MISSED — and the miss falsified the
byte model:** the "un-amortized expert bytes" story was wrong in an
interesting way. The per-token MoE kernels were ALREADY cache-grouped — only
~124 unique experts exist per layer (128×2.23 MB ≈ 285 MB working set, largely
SLC-resident across a layer), so explicit grouping buys far less than the
naive per-token byte count predicted (41 ms down ≈ 417 GB/s "effective" =
mostly L2/SLC hits). What grouping actually bought was dispatch-shape savings
(fewer, denser WGs). The remaining prefill floor (guGrp 70 + down 41 = 66%) is
kernel-shape/latency-bound, not DRAM-bound; the next lever would be a
chunk-shaped subgroup-matrix MoE GEMM (8M-tile), diminishing for this
campaign.

## M6 close-out: prefill 3-way (same GGUF, same box, 2026-07-12)

| engine | pp64 | pp512-class |
|---|---|---|
| ours, token-by-token (pre-M6) | 8.0 ms/tok = 125 tok/s | — (M-invariant) |
| ours, batched (P5+P6a+P6b) | **2.35 ms/tok = 425 tok/s** | ~same (MoE floor is M-invariant; MPRE=64 chunks) |
| llama.cpp fork 73d820a (Metal) | 1.60 ms/tok = **626 tok/s** | 0.68 ms/tok = **1470 tok/s** |

Batched prefill = 3.4× our own baseline, 68% of llama.cpp at M=64. llama.cpp
keeps scaling with M (1470 at 512) where our MoE floor is M-invariant — closing
that gap needs the chunk-shaped subgroup-matrix MoE GEMM (+ MPRE > 64), noted
as future work, diminishing for this campaign. Decode is untouched: GATE PASS,
95.0 tok/s quiet-window truth stands.

Author time M6 leg: ~2.5 h wall (P5 + P6a + P6b + baselines) — P7 (~2-3 h for
P5) HELD, and the whole of M6 fit in roughly the P5 allotment.

## P6c pre-registration (2026-07-12): chunk-shaped subgroup-matrix MoE GEMM

Q (user): is expert SELECTION the MoE bottleneck? A (from the budget): no —
routertop 2.2 ms + expgroup <1 ms ≈ 1-2% of prefill. The wall is the expert
matmul SHAPE: guGrp 70 ms vs ~33 ms byte floor, down 41 vs ~16 — scalar-matvec
kernels running 2-2.5× above their byte floors. If the shape is the wall, a
tensor-op GEMM over the SAME chunks should close toward the floor.

**Plan**: `q40gusg.wgsl` — per chunk (grid.z), M-tile = the chunk's ≤8 entries,
N = the expert's 2·704 gate|up rows (11 strips of 128), K = 2816 (88 q4_0
block-tiles). WG=128 = 4 subgroups × 4 result mats (8M×32N each). Raw GEMM out
per entry → `[M·K][2·FF]` scratch; geglu pairing done by a `geglub` variant
(every (tok,slot) entry is real, so the scratch is fully written).

**Predictions**: grouped gate/up 70 → **≤45 ms**; prefill M=64 2.35 →
**≤2.0 ms/tok**; gate2 PASS (accumulation-order change, tolerated by every
GEMM swap so far). If it lands, same treatment is a candidate for down.

## P6c/P6d result (2026-07-12): the MoE answer — selection is 1-2%, SHAPE and
## GRANULARITY are the wall, and fixing them tripled prefill again

User question that drove this leg: "is expert selection the MoE bottleneck?"
Measured answer: **no** (routertop 2.2 ms + expgroup <1 ms ≈ 1-2%); and it is
not JS either (JS only records dispatches). The wall was the expert matmul
shape × routing granularity, proven constructively:

- **P6c (MC=8 chunk GEMM, `q40gusg`)**: gate/up 70 → 60 ms at M=64 (prediction
  ≤45 **MISSED**), and REGRESSED M=20 (3.47 → 4.19) — at top-8/128 routing,
  n_e = M·K/E entries per expert ≈ 4 at M=64, so the 8-row tensor tiles ran
  half-empty. Kept behind an M ≥ 32 threshold. The MISS localized the real
  variable: **chunk fill, i.e. M**.
- **GEMM grouped-down (`q40downsg` + `wacc`)**: the scalar grouped-down that
  failed at M=64 (43.5 vs 41.3) WINS as a GEMM once chunks fill: down 321 →
  144 ms at M=512; also helped M=64 (2.28 → 2.09 ms/tok).
- **P6d (MC=32 chunks, `q40grpsg32`, one kernel for gate|up and down via
  ENTROW)**: at M=512 each expert's weights are read ~once per layer —
  gate/up 284 → **153 ms**, down 144 → **77 ms**.
- **MPRE 64 → 512** (prefill chunk size; ~150 MB extra activations) so large
  prompts actually reach the filled-chunk regime.

**Prefill ladder (batched, best-of-3, same box):**
| M | P5 | +P6a | +P6b | +P6c/d | tok/s |
|---|---|---|---|---|---|
| 20 | 3.62 | 3.54 | 3.47 | 3.47 | 288 |
| 64 | 3.69 | 2.69 | 2.35 | **2.09** | 478 |
| 512 | — | — | — | **0.99** | **1010** |

vs llama.cpp (same GGUF/box): pp64 626 tok/s → ours 76%; pp512 1470 → ours
**69%**. Gates: GATE PASS + GATE2 PASS after every step. Decode untouched
(10.6 ms wall today, profile 8.46 ≈ baseline 8.36).

Decode-side answer (same session): tokenwise prefill runs 8.0 ms/tok = 125
tok/s with the IDENTICAL per-token dispatch list, because token t's lm_head
(1.25 ms serialized) overlaps token t+1's layers when tokens are known; decode
pays it serially through the amax→embed feedback = 10.6 ms = 94 tok/s. The
125→94 gap is the price of autoregression, not a MoE or JS inefficiency.

## Pre-registration (2026-07-12 evening): the "existing WebGPU A4B" comparison

Survey result: NO existing WebGPU engine runs gemma-4-26B-A4B — webml is
E2B-only (M1/P1: cannot take A4B, no mobile checkpoint), WebLLM/MLC has no
gemma4 support (unknown model type), onnx-community ships only E2B/E4B. The
ONLY comparable is llama.cpp's own experimental ggml-webgpu backend (the
gemma4 fork has MUL_MAT_ID + flash-attn + fused rms in WGSL), built against
hesper's Dawn install, same GGUF, same box.

**Predictions**: the backend is young (no subgroup-matrix, generic WGSL): tg64
**30–70 tok/s** (below both Metal 112.5 and ours 95); pp512 **300–900 tok/s**
(below Metal 1470 and ours 1010). If it fails to load/run, that itself is the
finding (ours would be the only working WebGPU path for this model).

## Result (2026-07-12): there IS no existing WebGPU gemma-4-26B-A4B — measured, not assumed

Built llama.cpp (gemma4 fork 73d820a) with `GGML_WEBGPU=ON` against hesper's
Dawn install and ran the SAME GGUF: **crashes during prompt processing** —
Dawn validation error in `glu_geglu_f32_split`: "Writable storage buffer
binding aliasing … overlapping ranges (offset 2624000, size 2880768) and
(offset 2626816, …) in tensor_buf3". ggml's strided gate/up views of the fused
gemma4 FFN tensor legitimately overlap in one buffer, which WebGPU's aliasing
rules forbid — a structural port gap, not a perf gap. (Predictions moot;
the fail-to-run branch of the pre-registration is the finding.)

Survey of the rest: webml = E2B only (M1/P1), WebLLM/MLC = no gemma4
(`Unknown model type: gemma4`), onnx-community = E2B/E4B exports only.

**Conclusion: this engine is, as far as we can determine, the only working
WebGPU implementation of gemma-4-26B-A4B.** The only meaningful performance
references remain native: llama.cpp Metal (decode 112.5 → ours 84.4%; pp512
1470 → ours 69%) and the webml E2B engine as a method/architecture reference.
