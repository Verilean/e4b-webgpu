# DG per-op side-by-side: hesper metal vs llama.cpp (measured, 2026-07-18)

## Anchors (this box, clean state, France -n 256)

| runtime | wall/step | steps | canvas tok/s | GPU-busy/step (forward) |
|---|---|---|---|---|
| hesper metal backend | 611ms | 7 | ~60 | ~554ms (eval emb+fwd avg) |
| llama.cpp (refs/llama.cpp-diffusiongemma, warm) | **625-745ms** | 8 | 43-51 | **~192-203ms** (encoder replay min / isolated sum) |

**The historical "363ms/step, 64.2 tok/s" anchor does NOT reproduce** (cold run: 1004ms/step;
warm: 625-745). Treat 625-745ms as today's llama.cpp anchor.

**Headline asymmetry**: llama.cpp's GPU forward is ~2.7× faster than ours (192 vs 554ms), but
they burn ~430-550ms/step OUTSIDE the forward encoder (sampling encoders, CPU step logic,
sync) — which is why end-to-end we are at parity. Their kernels + our thin runtime + our
scheduler ≈ 200+60ms/step ≈ 7×260ms ≈ **~140 canvas tok/s theoretical** — that is the prize.

## llama.cpp per-op table (steady step, encoder=11, 1383 dispatches, isolated-replay sum 202.8ms)

Method: GGML_METAL_REPLAY=11 GGML_METAL_REPLAY_PEROP=1 (per-dispatch isolated replay ×3 min,
attribution via PSO→name registry; patch local to refs/, ggml-metal-device.m).

| op | n | total ms |
|---|---|---|
| mul_mm_q4_K_f32 (dense/attn projections) | 147 | 58.28 |
| mul_mm_id_q4_K (MoE gate/up) | 27 | 44.53 |
| mul_mm_q6_K_f32 | 11 | 32.01 |
| mul_mm_id_q5_0 (MoE down) | 16 | 13.79 |
| cpy_f32_f32 (one big copy) | 1 | 9.86 |
| mul_mm_id_q8_0 (MoE down) | 11 | 8.23 |
| mul_mm_q5_0_f32 | 16 | 4.34 |
| flash_attn_ext dk256 + dk512 | 27 | 6.09 |
| ALL elementwise (rms_norm_mul(_add), bin_fuse nf≤7, unary, geglu, rope, cpy, concat, repeat) | ~700 | **~16** |
| sampling-adjacent (argsort, soft_max, sum_rows, get_rows, map0) | ~190 | ~2.3 |
| **sum** | 1383 | **202.8** |

Whole-encoder replay: serial min 191.6ms; as-recorded-concurrent 284.8ms (their barrier layout
is a net LOSS at this shape — 957 barriers / 1383 ops; anti-finding for concurrency ports).

## Class diff (ours = clean session anchors, R60-62)

| class | hesper ms | llama ms | gap | verdict |
|---|---|---|---|---|
| MoE total (matmuls + aux chain) | 230-280 | **66.5** | **-165..215** | ① their monolithic mul_mm_id vs our router→sort→gather→reg-mm→scatter→wacc chain. (Old "our grouped-reg beats mul_mat_id per-kernel" was per-KERNEL; the CHAIN loses.) |
| dense+qkv+attnO matmuls | 165-205 | **~99** | -70..100 | ② their dequant-in-flight mul_mm (q4_K/q6_K direct) vs our f16-predequant two-stage; also they forward 256 canvas rows only (prompt KV-cached) vs our 277 |
| elementwise tail | 60-90 | **~16** | -45..75 | ③ their fusion families: rms_norm_mul(_add), bin_fuse nf up to 7, geglu — a direct blueprint for our fusion campaign |
| attention | ~19 (flash banked: ~10) | 6.1 | -9..13 | ④ their flash_attn_ext; our banked flashAttnB covers most of this |
| SC + lm_head + sampler | ~80 | not in this encoder (+cpy 9.9) | n/a | needs their sampling-encoder measurement for a fair compare |
| runtime overhead (wall − GPU) | **~60** | **~430-550** | +370..490 OUR WAY | ⑤ anti-finding: their runtime wastes 3×; do NOT port their orchestration |

## Ranked port candidates (MIT, with attribution; all go through our Stage-3 MSL checker)

1. **mul_mm_id family** (`kernel_mul_mm_id_*` + `kernel_mul_mm_id_map0` in ggml-metal.metal):
   -165..215ms. Effort HIGH (function-constant machinery, ids mapping, our expert layout
   differs) — but it replaces our entire grouped MoE chain including its aux dispatches.
2. **Elementwise fusion families** (`kernel_rms_norm_mul`, `_mul_add`, `kernel_bin_fuse` nf-chains,
   `kernel_geglu`): -45..75ms. Effort LOW-MED, incremental per family; their fusion pass
   (ggml-metal-common / ops.cpp) tells us WHICH sequences to fuse in our DSL.
3. **Dequant-in-flight mul_mm** (`kernel_mul_mm_q4_K_f32`, `_q6_K_f32`): -70..100ms. Effort MED.
   Overlaps partially with our existing dp4a/MMQ work — bench their kernel at our shapes first.
4. Prompt-KV structural (theirs proves canvas-only steps work) — ours forwards 277 vs their 256
   rows: small (-8%) — low priority.
5. flash_attn: already banked ours (DG_FLASH); their dk256 config is confirmation, not a port.

## Measurement provenance

- llama.cpp build: refs/llama.cpp-diffusiongemma @ 73d820a + local diffs (KV-cache ubatch patch,
  hrp replay harness, this task's per-op extension in ggml-metal-device.m).
- hesper: metal backend @ 0ec5d40 era anchors (611ms France / 554ms eval emb+fwd).
- Isolated-replay caveat: per-dispatch times exclude inter-op cache effects (sum 202.8 vs
  serial-encoder 191.6 = 6% inflation — small).
