# Reference reading: webml gemma-4-webgpu-kernels (principle 7 — read BEFORE designing)

Sources: the HF Space (index.html / landing.js / gemma-4-e2b.js) vendored at
`hesper/refs/webml-gemma4/`; a REAL captured decode token (steady, token #7) traced
via WebGPU API hooks — trace assets and per-op data in `hesper/tools/replay/webml/`
and hesper DEVPLAN §9c. Their E2B numbers on this box: ~4.0 ms/token real,
3.90 ms replayed GPU (287 GB/s effective), host ≲0.3 ms.

## The numbers that define the target shape

- **316 dispatches / token, 53 pipelines** (E2B, 30 layers-equivalent arch; E4B at 42
  layers ⇒ expect ~440 ops if we match their per-layer op count of ~10).
- ~10 ops per layer vs hesper's ~16: the difference is epilogue fusion (below).
- Fattest kernels dominate: per-op mean 12.3 µs — few, large, occupying dispatches.

## Per-layer op sequence (extracted from the trace + kernel labels)

rmsSrq → qkvProj → qkNormRope(+flash attention fused, 1 dispatch) → rms(v-lane) →
strided×2 (KV cache) → decodeAttention → **OprojNorm** → [gate/up] → **DownNormAdd**
→ decodePleGate → **PleProjNorm**

Fusion inventory (the patterns to reproduce, names from their pipeline labels):
- **RmsSrq** — RMS norm fused with static-range int8 quantization of the activation
  (produces the int8 operand the next matmul consumes; uses the checkpoint's
  input_activation_scale).
- **OprojNorm / DownNormAdd / GateUpNorm / NormAddNorm** — matmul with the following
  norm and/or residual-add folded into the kernel tail (epilogue fusion; the norm can
  ride the matmul because their matmul assigns whole output rows to one workgroup).
- **DecodeAttention** — qk-norm + RoPE + flash attention in ONE dispatch per layer.
- **PleGate / PleProjNorm** — per-layer-embedding gate + projection with norm folded.

## Kernel-authoring conventions (from wgsl/k00,k07 etc.)

- Kernels are Jinja templates; ALL dims/workgroup sizes flow from config at
  instantiation ({{ source.hidden }}, {{ HEAD_DIM }}, {{ WG }}); `enable f16` and
  subgroup use are template flags. Subgroup reductions (subgroupAdd) for norms;
  one workgroup owns one normalization row.
- Operands: int4 weights × int8 (SRQ) activations; f16 accumulation where safe.
- Buffers prebuilt once; bind groups cached; JS loop does nothing per token beyond
  encode+submit (host ≲0.3 ms measured).

## What we deliberately do differently

- Engine written from scratch (their space has no LICENSE; code is reference only).
- Goldens from transformers on the SAME QAT weights (they had Google-internal
  references; we make ours reproducible).
- Kernels live in kernels/*.wgsl as plain files with a trivial ${} substitution —
  no template engine dependency; hot-reload via the harness.
