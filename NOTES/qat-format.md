# QAT-mobile safetensors format — google/gemma-4-E4B-it-qat-mobile-transformers

From the safetensors header (418,608-byte JSON; full model 3,361 MB, 3,104 tensors).
**Multimodal checkpoint**: audio_tower / vision tower included — we load ONLY
`model.language_model.*` + `lm_head.*` (1,927 tensors, **3.199 GB**).

## Config (config.json → text_config)

hidden 2560 · 42 layers · 8 q-heads / 2 kv-heads · head_dim 256 · vocab 262144 ·
intermediate 10240 · sliding_window 512 · num_kv_shared_layers 18 ·
hidden_size_per_layer_input 256. No AltUp/Laurel tensors.

## Quantized-linear convention (per layer, names relative to `model.language_model.layers.N.`)

Every linear = 4 tensors:
- `<name>.weight` — U8 or I8, shape `[out, in_packed]`
- `<name>.weight_scale` — F32 `[out, 1]` (per-output-channel)
- `<name>.input_activation_scale`, `<name>.output_activation_scale` — F32 scalars
  (static activation quantization — the SRQ int8 regime webml exploits)

| tensor | dtype | shape | in-dim | packing |
|---|---|---|---|---|
| self_attn.q_proj.weight | U8 | [2048, 1280] | 2560 | **int4 ×2/byte** (in/2) |
| self_attn.k_proj / v_proj | U8 | [512, 1280] | 2560 | int4 |
| self_attn.o_proj | U8 | [2560, 1024] | 2048 | int4 |
| mlp.gate_proj / up_proj | U8 | [10240, 1280] | 2560 | int4 |
| mlp.down_proj | U8 | [2560, 5120] | 10240 | int4 |
| per_layer_input_gate | I8 | [256, 2560] | 2560 | **int8 plain** |
| per_layer_projection | I8 | [2560, 256] | 256 | int8 |

Norms: BF16 vectors (input_layernorm, post_attention_layernorm,
pre/post_feedforward_layernorm, post_per_layer_input_norm, q_norm/k_norm [256]).
Extras per layer: `layer_scalar` BF16 [1]; `self_attn.k_cache_scale` /
`v_cache_scale` F32 scalars (quantized KV cache!).

## Global tensors

| tensor | dtype | shape | note |
|---|---|---|---|
| lm_head.weight | U8 | [262144, 640] | in=2560 → 640 = in/4 — **OPEN QUESTION: 2-bit? or a different packing** (§verify) |
| lm_head.weight_scale | F32 | [262144, 1] | |
| embed_tokens.embedding_quantized | U8 | [262144, 640] | same /4 packing as lm_head (tied?) |
| embed_tokens.embedding_scale | F32 | [262144, 1] | per-row |
| embed_tokens_per_layer.embedding_quantized | U8 | [262144, 2688] | 42×256=10752 dims → /4 again |
| embed_tokens_per_layer.embedding_scale | F32 | [262144, 42] | per-row-per-layer |
| per_layer_model_projection.weight | BF16 | [10752, 2560] | 55 MB, read per token |
| norm.weight | BF16 | [2560] | final norm |

**VERIFY in M2 (goldens/gen.py)**: the `/4` tensors' exact packing (2-bit vs
4-bit-with-groups) and the int4 nibble order/zero-point — read HF transformers'
Gemma-4-mobile modeling/dequant code while generating goldens; the goldens make any
misreading immediately visible as a layer-0 mismatch.

## Per-token decode byte accounting (drives the BW floor in DEVPLAN)

- per layer: q 2.62 + k 0.66 + v 0.66 + o 2.62 + gate 13.11 + up 13.11 + down 13.11
  + PLE gate/proj 1.31 + scales/norms ≈ 0.07 ≈ **47.2 MB** → ×42 = **1.98 GB**
- lm_head 167.8 + 1.0 MB; per_layer_model_projection 55.1 MB
- embed row + PLE row: ~3 KB (indexed)
- **total ≈ 2.20 GB/token**
