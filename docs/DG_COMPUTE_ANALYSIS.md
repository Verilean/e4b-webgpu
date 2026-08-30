# DG compute-utilization analysis: hesper 611ms vs llama.cpp 363ms (2026-07-18)

Analysis-only pass (no kernel changes). Question: is the residual 1.7× a worker/tile
configuration gap (user hypothesis) or algorithm gap, class by class?

## Method + measurement caveats
- Native DG_PROF under the metal backend is INFLATED (per-mark commit+wait drains:
  sum ≈1480ms vs clean wall 611ms) — used for RATIOS only.
- Chrome-148 kernel profile this pass ran under 4.6GB swap residue (2.5× uniform
  inflation, R54 class) — used for RANKING only.
- Absolute anchors are the CLEAN in-session numbers: metal+MSL France steady
  611-612ms (lmhead+reduce 47-52ms), clean Chrome-148 all-WGSL 761ms with
  gate/up ≈11ms/layer, Dawn MSL-pair A/B delta -180ms, metal pair delta -151ms.
- Peaks assumed (empirical, earlier autotune work): ~15.4 TFLOPS matrix f16,
  546 GB/s DRAM (M4 Max 40-core).

## Per-class budget of the 611ms step (France steady, bands honest)

| class | est. ms | util (achieved/peak) | bound | note |
|---|---|---|---|---|
| MoE gate/up+down (hand-MSL pair) | 230-280 | ~30-40% | compute | ~51+26 GFLOP/layer; ALREADY ≥ llama.cpp's mul_mat_id (27%/4.25T vs our 39%/6T measured) — ANTI-FINDING |
| dense f16 WMMA (M=277) | 90-105 | **~12%** | neither → occupancy/pipeline | the recoverable class |
| qkv + attnO f16 WMMA (M=277) | 75-100 | **~10-15%** | neither | same class |
| battnB attention | 30-45 | **~1-2%** | latency (serial per-thread K loops) | algorithm gap vs flash-attn |
| elementwise/norm/router/grouping tail | 60-90 | n/a | latency (1-2ms × many dispatches) | fusion headroom |
| SC expectation (probs×embTT WMMA) | ~30 | **~70%+** | compute | near-roofline — ANTI-FINDING |
| lm_head + reduce | ~50 | ~40% + streams near BW | mixed | fine — ANTI-FINDING |

## llama.cpp Metal configs (read from refs/llama.cpp-diffusiongemma ggml-metal)

- `kernel_mul_mm` (dense/qkv class at ne11=277): threadgroup tile **64(M)×128(N)**,
  128 threads = 4 simdgroups (2×2), each SG a 32×64 quadrant of 8×8 fragments
  (N_MM_BLOCK 4×2, SIMD_GROUP 2×2, SZ 16); K-step loop stages **only the
  dequantized A tile** in threadgroup memory — the f16 activation B is read
  DIRECTLY by simdgroup_load from device (unified memory), no B staging.
  Chosen when `has_simdgroup_mm && ne00 >= 64 && ne11 > ne11_mm_min`; mul_mv
  variants only for ne11 ≤ 8.
- `kernel_flash_attn_ext`: fused tiled attention, dk/dv-templated (incl. 256),
  simdgroup-parallel — vs our battnB's 1-TG-per-(head,row) with serial loops.
- `mul_mm_id` MoE: we already match/beat it per-kernel (earlier fork benchmark).

## OUR reg kernel (matMulTransposeF16WMMARegKernel) vs theirs — the config diff

| aspect | ours | llama.cpp |
|---|---|---|
| TG output tile | 64×32 | 64×128 (**4× more work per TG**) |
| staging | BOTH A and B via threadgroup, per-element div/mod index chains | A only; B via direct simdgroup_load |
| threads/TG, shared | 128, ~6KB | 128, similar |
| consequence at M=277 | many small TGs, low accumulator reuse, barrier-heavy staging → 10-15% util | 4× fewer TGs, 4× B-reuse per load → 40-50% util typical |

Verdict on the hypothesis: **CONFIRMED for the WMMA class** — it is precisely a
worker/tile configuration + load-path gap (not WGSL-vs-MSL, not surface code
quality — R54 already showed surface edits do nothing). For attention it is an
algorithm gap (flash-attn). For MoE/SC/lm_head we are already at or above
llama.cpp class — the earlier "MoE is the gap" era ended when the reg kernels
deployed.

## Ranked recoverable list

1. **WMMA tile widening + B-direct-load (qkv/attnO/dense): recover ~100-140ms.**
   Cheap first probe: parameterize the generator's N-tile 32→64→128 (Cx 8→16
   register pressure watch), drop shared_B in favor of direct loads.
2. **battnB → fused tiled attention: recover ~20-35ms.** Higher effort, real
   algorithm work (llama.cpp's FA as the reference shape).
3. **Elementwise tail fusion/batching: recover ~20-40ms.** Retry geglu+q80 fuse
   under the chat template (its pre-template rejection was the near-tie disease,
   since cured); norm+rope; router chain.
4. MoE sentinel-trim residue: ≤30ms. 5. SC/lm_head: ~0 (leave).

Projection if 1-3 land: 611 → ~400-455ms/step. With our eff-steps advantage
(7 vs llama.cpp's 11 on France): ~256/(7×0.43) ≈ **85-91 canvas tok/s vs their 64**
— end-to-end win WITHOUT further schedule work. Remaining ~40-90ms to their 363
is their long-tail polish; not needed to win end-to-end.

## Top-3 experiment proposals (in order)

1. Tile-N sweep on the reg generator (32/64/128) + B-direct variant, golden +
   eval-gated, native metal backend. Expected -60..-140ms. Risk: register
   pressure (Cx 16 frags), medium. PERF_AUTOTUNE_LOOP flow applies as-is.
2. geglu+q80 + norm-chain fusions under template. Expected -20..-40ms. Low risk.
3. Flash-attention kernel (new, llama.cpp-shaped). Expected -20..-35ms. High
   effort — schedule after 1-2.
