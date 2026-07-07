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

## Author-time log

| span | wall clock | what |
|---|---|---|
| start | 2026-07-07 09:41 | M0 begun (repo, header analysis, notes) |

## Decision log

| date | decision | basis |
|---|---|---|
| 2026-07-07 | New minimal repo; scratch engine with webml as reference reading; swap = baseline/oracle + fallback | user; hesper report P2 (context compactness), principle 7 |
