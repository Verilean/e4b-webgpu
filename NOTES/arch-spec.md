# Gemma4 Text Decoder Forward-Pass Specification (Single-Token Decode, Batch=1)

**Generated from**: transformers/models/gemma4/modeling_gemma4.py and modular_gemma4.py  
**Config source**: config.json (42 layers, 2560 hidden_size, 8 attn heads, 2 kv heads)

---

## 1. EMBEDDING LAYER

### 1.1 Token Embedding
**Line ref**: `modeling_gemma4.py:1608-1610`, `Gemma4TextScaledWordEmbedding.forward()`

```
inputs_embeds = embed_tokens(input_ids)  # [1, 1, 2560]
              = embed_tokens.weight[input_ids] * embed_scale  # [1, 1, 2560]
```

**embed_scale formula** (line 1609):
```
embed_scale = hidden_size^0.5 = 2560^0.5 ≈ 50.596...
```
Cast to BF16 (applied in weight type).

### 1.2 Per-Layer Input Embeddings (PLE) — Token-Identity Component
**Line ref**: `modeling_gemma4.py:1682-1684`, `get_per_layer_inputs()` method

Only computed if `config.hidden_size_per_layer_input > 0` (true in config: 256).

```
per_layer_raw = embed_tokens_per_layer(input_ids)
              # [1, 1, num_layers * hidden_size_per_layer_input]
              # = [1, 1, 42 * 256] = [1, 1, 10752]
              # Apply embed_scale internally: hidden_size_per_layer_input^0.5 = 256^0.5 = 16

per_layer_inputs_identity = per_layer_raw.reshape(
    *input_ids.shape,
    config.num_hidden_layers,
    hidden_size_per_layer_input
)
# Result shape: [1, 1, 42, 256]
```

### 1.3 Per-Layer Input Embeddings — Context Projection
**Line ref**: `modeling_gemma4.py:1684`, `project_per_layer_inputs()` method

```
per_layer_projection = per_layer_model_projection(inputs_embeds)
                    # Linear(2560 -> 42*256=10752)
                    # [1, 1, 10752]
                    * per_layer_model_projection_scale

# per_layer_model_projection_scale = hidden_size^-0.5 = 2560^-0.5 ≈ 0.0198...

per_layer_projection = per_layer_projection.reshape(
    *inputs_embeds.shape[:-1],
    config.num_hidden_layers,
    hidden_size_per_layer_input
)
# Result: [1, 1, 42, 256]

per_layer_projection = per_layer_projection_norm(per_layer_projection)
# RMSNorm applied per layer-dim (across 256 features)
```

### 1.4 Per-Layer Input — Combined (Identity + Context)
**Line ref**: `modeling_gemma4.py:1821`

```
per_layer_inputs = (per_layer_projection + per_layer_inputs_identity) 
                 * per_layer_input_scale

# per_layer_input_scale = 2^-0.5 ≈ 0.7071...
# Final shape: [1, 1, 42, 256]
```

---

## 2. DECODER LAYER ARCHITECTURE

### Layer Structure (repeated 42 times, indices 0–41)
**Line ref**: `modeling_gemma4.py:1376-1403`, `Gemma4TextDecoderLayer` class

Each layer consists of:

1. **input_layernorm**: RMSNorm(hidden_size=2560)
2. **self_attn**: Gemma4TextAttention (see §3)
3. **post_attention_layernorm**: RMSNorm(hidden_size=2560)
4. **mlp**: Gemma4TextMLP (see §4)
5. **pre_feedforward_layernorm**: RMSNorm(hidden_size=2560)
6. **post_feedforward_layernorm**: RMSNorm(hidden_size=2560)
7. Optional: **per_layer_input_gate** + normalization (if hidden_size_per_layer_input > 0)
8. Optional: **moe** blocks (if enable_moe_block=true, but false in config)

### 2.1 RMSNorm Implementation
**Line ref**: `modeling_gemma4.py:197-215`

```python
def rms_norm(x, weight, eps=1e-6):
    # x: [1, 1, 2560]
    # weight: [2560]
    
    # Cast input to float32 for stability
    x_f32 = x.float()
    
    # Compute mean of squares along last dim
    mean_squared = (x_f32 ** 2).mean(dim=-1, keepdim=True) + eps
    # mean_squared: [1, 1, 1]
    
    # Normalize: use pow() for numerical stability
    normed = x_f32 * (mean_squared ** -0.5)
    # normed: [1, 1, 2560]
    
    # Apply weight (scale factor)
    output = normed * weight.float()
    
    # Cast back to input dtype
    return output.to(x.dtype)
```

**Formula**: `y = (x / sqrt(mean(x²) + eps)) * weight`

**eps values** (line 1637, 1615, 1384-1387):
- Default: `config.rms_norm_eps = 1e-6`

**Important**: RMSNorm internally casts to/from float32 (lines 212-215).

### 2.2 Decoder Layer Forward Pass
**Line ref**: `modeling_gemma4.py:1405-1462`

```
For i in range(42):  # num_hidden_layers
    # Input: hidden_states [1, 1, 2560], per_layer_input [1, 1, 256]
    
    # ===== ATTENTION BLOCK =====
    residual = hidden_states  # [1, 1, 2560]
    
    hidden_states = input_layernorm(hidden_states)
    hidden_states, _ = self_attn(
        hidden_states,
        position_embeddings=(cos, sin),  # layer-type-specific RoPE
        attention_mask=mask,
        shared_kv_states={layer_type: (K, V)},
        past_key_values=kv_cache,
        position_ids=position_ids
    )
    # Output: [1, 1, 2560]
    
    hidden_states = post_attention_layernorm(hidden_states)
    hidden_states = residual + hidden_states
    # Residual add: [1, 1, 2560]
    
    # ===== MLP BLOCK =====
    residual = hidden_states
    
    hidden_states = pre_feedforward_layernorm(hidden_states)
    hidden_states = mlp(hidden_states)
    # Output: [1, 1, 2560]
    
    hidden_states = post_feedforward_layernorm(hidden_states)
    hidden_states = residual + hidden_states
    # Residual add: [1, 1, 2560]
    
    # ===== PER-LAYER INPUT (PLE) BLOCK =====
    if hidden_size_per_layer_input > 0:
        residual = hidden_states
        
        hidden_states = per_layer_input_gate(hidden_states)
        # Linear(2560 -> 256): [1, 1, 256]
        
        hidden_states = act_fn(hidden_states)
        # Activation function: config.hidden_activation = "gelu_pytorch_tanh"
        # This applies GELU variant (see §5)
        
        hidden_states = hidden_states * per_layer_input
        # Element-wise multiply: [1, 1, 256]
        
        hidden_states = per_layer_projection(hidden_states)
        # Linear(256 -> 2560): [1, 1, 2560]
        
        hidden_states = post_per_layer_input_norm(hidden_states)
        # RMSNorm(2560): [1, 1, 2560]
        
        hidden_states = residual + hidden_states
        # Residual add: [1, 1, 2560]
    
    # ===== LAYER SCALAR =====
    hidden_states = hidden_states * layer_scalar
    # layer_scalar: [1] (BF16, initialized to 1.0)
    # Stored per-layer in register_buffer
```

---

## 3. ATTENTION MECHANISM

### 3.1 Configuration Parameters
**Line ref**: `config.json`, `configuration_gemma4.py`

```
# From config.json:
num_attention_heads = 8
num_key_value_heads = 2
head_dim = 256  (for sliding layers)
global_head_dim = 512  (for full attention layers)

# Layer types:
layer_types = [
    "sliding_attention" (indices 0-4),
    "full_attention" (index 5),
    "sliding_attention" (indices 6-10),
    ...
    "full_attention" (index 41 — last layer always full)
]

# RoPE parameters (modular_gemma4.py:210-215):
sliding_attention: {
    "rope_type": "default",
    "rope_theta": 10_000.0
}
full_attention: {
    "rope_type": "proportional",
    "partial_rotary_factor": 0.25,
    "rope_theta": 1_000_000.0
}

# Sliding window:
sliding_window = 512  (only for sliding_attention layers)
```

### 3.2 RoPE Calculation
**Line ref**: `modeling_gemma4.py:1093-1180`, `Gemma4TextRotaryEmbedding.forward()`

#### For **sliding_attention** layers (rope_type="default"):

```
theta = 10_000.0
dim = head_dim = 256

# Compute inverse frequencies (computed once, cached)
inv_freq = 1.0 / (theta ^ (arange(0, dim, 2) / dim))
# inv_freq shape: [128]  (half of head_dim)
# inv_freq in float32 during computation

# At inference time:
# position_ids shape: [1, 1] (position 0 for single token)

inv_freq_expanded = inv_freq[None, :, None].float()  # [1, 128, 1]
position_ids_expanded = position_ids[:, None, :].float()  # [1, 1, 1]

# Compute frequencies (in float32, no_grad context):
freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
# freqs shape: [1, 1, 128] (frequencies for this position)

# Duplicate to full dimension (standard RoPE):
emb = torch.cat((freqs, freqs), dim=-1)  # [1, 1, 256]

cos = emb.cos() * attention_scaling  # attention_scaling=1.0 for default
sin = emb.sin() * attention_scaling

# Cast to input dtype (BF16):
cos, sin = cos.to(x.dtype), sin.to(x.dtype)
```

#### For **full_attention** layers (rope_type="proportional"):

```
theta = 1_000_000.0
head_dim = global_head_dim = 512
partial_rotary_factor = 0.25

# Only rotate first 25% of head_dim:
rotated_dim = int(head_dim * partial_rotary_factor) = 128

inv_freq = 1.0 / (theta ^ (arange(0, rotated_dim, 2) / rotated_dim))
# inv_freq shape: [64]

# Same forward pass as sliding, but:
# 1. Only applies to first 128 dims of 512-dim heads
# 2. Remaining 384 dims are not rotated
# 3. attention_scaling may differ (determined by rope_init_fn)
```

### 3.3 Q/K/V Projections
**Line ref**: `modeling_gemma4.py:1212-1233`, `Gemma4TextAttention.__init__()`

#### Query projection:
```
query_states = self.q_proj(hidden_states)
# Linear(hidden_size=2560 -> num_attention_heads * head_dim)
# For sliding: 8 * 256 = 2048
# For full: 8 * 512 = 4096
# Shape after proj: [1, 1, 2048 or 4096]

query_states = query_states.view(*input_shape, -1, head_dim)
# [1, 1, 8, 256] or [1, 1, 8, 512]

query_states = q_norm(query_states)
# RMSNorm(dim=head_dim, eps=1e-6)
# Normalizes along head_dim: [1, 1, 8, 256] or [1, 1, 8, 512]

query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
# Apply RoPE to query (see §3.5)

query_states = query_states.transpose(1, 2)
# Rearrange to [1, 8, 1, 256] or [1, 8, 1, 512]
```

#### Key/Value projection (non-KV-shared layers):
```
key_states = self.k_proj(hidden_states)
# Linear(2560 -> num_key_value_heads * head_dim)
# For sliding: 2 * 256 = 512
# For full: 2 * 512 = 1024 (using global num_key_value_heads if config.num_global_key_value_heads is set)
# Shape: [1, 1, 512 or 1024]

key_states = key_states.view(*input_shape, -1, head_dim)
# [1, 1, 2, 256] or [1, 1, 2, 512]

key_states = k_norm(key_states)
# RMSNorm(dim=head_dim): [1, 1, 2, 256] or [1, 1, 2, 512]

key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)

key_states = key_states.transpose(1, 2)
# [1, 2, 1, 256] or [1, 2, 1, 512]

# Value projection (NOT shared for these layers):
value_states = self.v_proj(hidden_states)
# Linear(2560 -> num_key_value_heads * head_dim)
# Same output shape logic as keys

value_states = value_states.view(*input_shape, -1, head_dim)
value_states = v_norm(value_states)
# RMSNorm(dim=head_dim, with_scale=False)  ← no weight applied
value_states = value_states.transpose(1, 2)
# [1, 2, 1, 256] or [1, 2, 1, 512]
```

#### KV-Shared layers (last 18 layers):
```
if layer_idx >= (num_hidden_layers - num_kv_shared_layers):
    # Reuse K,V from shared_kv_states dict
    key_states, value_states = shared_kv_states[layer_type]
    # Move to current device
else:
    # Compute normally
    ...
    # Store full-length KV if this is the last sliding layer before full:
    if store_full_length_kv:
        shared_kv_states[layer_type] = key_states, value_states
```

### 3.4 KV Cache Update (Single-Token Decode)
**Line ref**: `modeling_gemma4.py:1273-1276`

For each layer with caching enabled:
```
if past_key_values is not None and not self.is_kv_shared_layer:
    key_states, value_states = past_key_values.update(
        key_states, value_states, self.layer_idx
    )
    # Appends current token KV to cache for next token decode
```

### 3.5 Attention Computation
**Line ref**: `modeling_gemma4.py:827-858`, `eager_attention_forward()`

```
# After RoPE is applied to Q, K
# Q: [1, 8, 1, 256] (or 512 for full)
# K: [1, 2, 1, 256] (or 512 for full)
# V: [1, 2, 1, 256] (or 512 for full)

# GQA: Repeat KV heads to match query heads
key_states = repeat_kv(key_states, num_key_value_groups=4)
# [1, 2, 1, 256] -> [1, 8, 1, 256]
# (Each of 2 KV heads is repeated 4 times)

value_states = repeat_kv(value_states, num_key_value_groups=4)
# [1, 2, 1, 256] -> [1, 8, 1, 256]

# Attention scores
attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
# Q: [1, 8, 1, 256], K^T: [1, 8, 256, 1]
# Result: [1, 8, 1, 1]

# Scaling factor (applied after matmul):
scaling = 1.0  # module.scaling = 1.0 (line 1200)
# Note: Per the eager_attention_forward, if scaling is None:
#   scaling = head_dim^-0.5 = 256^-0.5 ≈ 0.0625
# But since self.scaling=1.0 is explicitly set, standard 1/sqrt(d_k) is NOT applied here.
# This means the attention is NOT normalized by sqrt(head_dim) by default.
```

**Softcap (Audio model only, NOT used for text decoder)**:
- Audio attention has attention_logits_soft_cap=50.0
- Text decoder: no softcap for attention (only for final logits)

```
# Attention mask
if attention_mask is not None:
    attn_weights = attn_weights + attention_mask
    # attention_mask shape: [1, 1, 1, 1] or similar
    # Mask values are typically -inf for masked positions

# Softmax (upcasted to fp32)
attn_weights = softmax(attn_weights, dim=-1, dtype=torch.float32)
# [1, 8, 1, 1] -> [1, 8, 1, 1] (sum to 1 along key dim)
attn_weights = attn_weights.to(query.dtype)  # Back to BF16

# Dropout (eval mode → 0.0)
attn_weights = dropout(attn_weights, p=0.0, training=False)

# Attention output
attn_output = torch.matmul(attn_weights, value_states)
# Attn: [1, 8, 1, 1], V: [1, 8, 1, 256]
# Result: [1, 8, 1, 256]

# Reshape and project
attn_output = attn_output.transpose(1, 2).contiguous()
# [1, 8, 1, 256] -> [1, 1, 8, 256]

attn_output = attn_output.reshape(input_shape, -1)
# [1, 1, 8*256] = [1, 1, 2048]

attn_output = o_proj(attn_output)
# Linear(2048 -> 2560) or Linear(4096 -> 2560) for full attention
```

### 3.6 Sliding Window Masking
**Line ref**: `configuration_gemma4.py:195-208`, `create_sliding_window_causal_mask()`

For sliding_attention layers:
```
sliding_window = 512  # from config

# Mask is created such that position i can attend to positions [i-512, i]
# (left_window=512, right_window=0 for causal)
# Implemented as add_mask with -inf outside window
```

---

## 4. MLP LAYER

### 4.1 Gemma4TextMLP Structure
**Line ref**: `modeling_gemma4.py:1074-1090`, `Gemma4TextMLP`

```
# Inherits from Gemma3MLP, which has:
gate_proj: Linear(hidden_size -> intermediate_size)
up_proj: Linear(hidden_size -> intermediate_size)
down_proj: Linear(intermediate_size -> hidden_size)
act_fn: Activation function

# Config parameters:
hidden_size = 2560
intermediate_size = 10240
hidden_activation = "gelu_pytorch_tanh"

# KV-shared layers use double-wide MLP:
if enable_moe_block=false (true in config):
    # No double-wide
    intermediate_size = 10240
else if is_kv_shared_layer and use_double_wide_mlp=true:
    intermediate_size = 10240 * 2 = 20480
```

### 4.2 MLP Forward Pass
**Line ref**: `modeling_gemma4.py:1088-1090`

```
output = down_proj(act_fn(gate_proj(x)) * up_proj(x))
# x: [1, 1, 2560]

gate = gate_proj(x)  # Linear(2560 -> 10240): [1, 1, 10240]
up = up_proj(x)      # Linear(2560 -> 10240): [1, 1, 10240]

# Activation function: gelu_pytorch_tanh
# From transformers/activations.py: 
# gelu_pytorch_tanh(x) = sqrt(2/pi) * (x + 0.044715 * x^3 tanh(...))
gate_activated = act_fn(gate)  # [1, 1, 10240]

# Element-wise multiply
intermediate = gate_activated * up  # [1, 1, 10240]

# Project down
output = down_proj(intermediate)  # Linear(10240 -> 2560): [1, 1, 2560]
```

### 4.3 Activation Sparsity
**Line ref**: config.json, configuration_gemma4.py

**Status**: NOT FOUND in provided model configuration.
- `config` does not contain `activation_sparsity_pattern` field
- Activation sparsity may be a training-time feature not reflected in inference

---

## 5. ACTIVATION FUNCTIONS

### 5.1 gelu_pytorch_tanh
**Line ref**: transformers/activations.py (external)

Used in:
- MLP: hidden_activation = "gelu_pytorch_tanh"
- Per-layer input gate: self.act_fn = ACT2FN[config.hidden_activation]

```
gelu_pytorch_tanh(x) {
    cdf = 0.5 * (1.0 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    return x * cdf
}

# Approximately:
# 0.5 * (1 + tanh(0.7978845608... * (x + 0.044715 * x^3)))

# Or more precisely (from torch source):
kAlpha = sqrt(2.0 / M_PI) = 0.7978845608...
result = x * 0.5 * (1.0 + tanh(kAlpha * (x + 0.044715 * x * x * x)))
```

---

## 6. FINAL LAYER NORM AND OUTPUT

### 6.1 Final Normalization
**Line ref**: `modeling_gemma4.py:1736`, after all 42 layers

```
hidden_states = norm(hidden_states)
# norm = Gemma4RMSNorm(hidden_size=2560, eps=1e-6)
# Input: [1, 1, 2560] (output from last decoder layer)
# Output: [1, 1, 2560]
```

### 6.2 Language Model Head
**Line ref**: `modeling_gemma4.py:1836`, `Gemma4ForCausalLM`

```
logits = lm_head(hidden_states)
# Linear(hidden_size=2560 -> vocab_size=262144, bias=False)
# Input: [1, 1, 2560]
# Output: [1, 1, 262144]

# Note: lm_head is tied to embed_tokens.weight (line 1826):
# _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
```

### 6.3 Final Logit Softcapping
**Line ref**: `modeling_gemma4.py:1896-1899`, `Gemma4ForCausalLM.forward()`

From config.json:
```
final_logit_softcapping = 30.0
```

```
if final_logit_softcapping is not None:
    logits = logits / final_logit_softcapping
    # [1, 1, 262144] / 30.0
    
    logits = torch.tanh(logits)
    # Squash to [-1, 1]
    
    logits = logits * final_logit_softcapping
    # [1, 1, 262144] * 30.0
    # Result range: [-30, 30]
```

**Formula**:
```
logits_capped = 30.0 * tanh(logits / 30.0)
```

This is a softer clipping than hard clamping; logits are smoothly bounded to ~[-30, 30].

---

## 7. QUANTIZATION (SRQ) AND CACHE

### 7.1 SRQ Quantization Awareness
**Line ref**: config.json quantization_config

The model config includes quantization specs, but the inference path in modeling_gemma4.py does NOT show explicit `apply_srq()` calls at activation points. Quantization is likely applied:

1. **At checkpoint load time**: Weights are dequantized to BF16
2. **During forward**: Computations in BF16
3. **No runtime SRQ**: The modeling code shows pure BF16 arithmetic

**Quantized modules** (from config.json):
- lm_head: 2-bit
- embed_tokens, embed_tokens_per_layer: 2-bit
- attention layers (q_proj, k_proj, v_proj, o_proj): 4-bit
- mlp layers (gate_proj, up_proj, down_proj): 4-bit
- per_layer_input_gate, per_layer_projection: 8-bit

### 7.2 KV Cache
**Line ref**: `modeling_gemma4.py:1686-1687`, `DynamicCache`

For single-token decode (token 0):
```
past_key_values = DynamicCache(config=self.config)
# On first forward: empty cache

# After first token:
# cache["self_attention"][layer_idx] = (key_states, value_states)

# For token 1 onwards:
# key_states, value_states = past_key_values.update(key_states, value_states, layer_idx)
# Appends new KV to existing cache
```

**Cache format** (per layer):
- Key cache: [batch=1, num_kv_heads=2, seq_len_so_far, head_dim=256]
- Value cache: [batch=1, num_kv_heads=2, seq_len_so_far, head_dim=256]

**No explicit quantization in cache**: Stored in same dtype as computation (BF16).

---

## 8. LAYER SCALAR

### 8.1 Definition and Usage
**Line ref**: `modeling_gemma4.py:1388`, `1461`

```
layer_scalar: torch.Tensor = register_buffer(torch.ones(1))
# Shape: [1]
# Dtype: inherited from model (BF16)
# Initialized to 1.0
```

**Applied at end of each layer**:
```
# Line 1461:
hidden_states = hidden_states * layer_scalar
# [1, 1, 2560] * [1] -> [1, 1, 2560]
```

**Purpose**: Per-layer scaling factor (likely learned during training; acts as optional amplitude scaling).

**In inference**: Generally 1.0 unless specifically tuned in checkpoint.

---

## 9. POSITION EMBEDDINGS

### 9.1 Absolute Position IDs
**Line ref**: `modeling_gemma4.py:1689-1692`

```
position_ids = torch.arange(inputs_embeds.shape[1], device=device)
            + past_key_values.get_seq_length()
position_ids = position_ids.unsqueeze(0)
# For first token (seq_len=0): position_ids = [[0]]
# For second token (seq_len=1): position_ids = [[1]]
# Shape: [1, 1]
```

### 9.2 RoPE per Layer Type
**Line ref**: `modeling_gemma4.py:1712-1714`

```
position_embeddings = {}
for layer_type in self.unique_layer_types:  # {"sliding_attention", "full_attention"}
    position_embeddings[layer_type] = self.rotary_emb(
        hidden_states, position_ids, layer_type
    )
# Computes (cos, sin) for each layer type's RoPE params
```

---

## 10. DTYPE AND CASTING POINTS

### 10.1 Model Dtype: BF16
- Config specifies: `"dtype": "bfloat16"`
- All weights loaded as BF16
- Activations typically maintained in BF16

### 10.2 Float32 Casting (for numerical stability)
1. **RMSNorm** (line 212): Cast input to float32, compute norm, cast output back
2. **RoPE computation** (line 1174): Force float32 during frequency/cos/sin calc
3. **Softmax** (line 854): Upcast attention to float32, downcast result
4. **Final softcapping** (line 1897): Computed in-place (likely preserves BF16)

### 10.3 Quantization Casting Points
- Not explicitly visible in inference code
- Weights dequantized at load; inference proceeds in BF16

---

## 11. SUMMARY: Single-Token Forward Pass Order

```
1. Tokenize input_ids -> [1, 1]
2. embed_tokens(input_ids) -> [1, 1, 2560] (BF16)
3. PLE token-identity: embed_tokens_per_layer + reshape -> [1, 1, 42, 256]
4. PLE context projection: per_layer_model_projection + norm -> [1, 1, 42, 256]
5. PLE combine: (proj + identity) * sqrt(0.5) -> [1, 1, 42, 256]
6. position_ids = [0] -> [1, 1]
7. Compute RoPE (cos, sin) for each layer_type
8. For i in range(42):
    a. input_layernorm(hidden_states)
    b. Attention: Q/K/V proj -> RoPE -> GQA -> matmul -> softmax -> output proj
    c. post_attention_layernorm + residual
    d. pre_feedforward_layernorm + MLP (gate*up projected down) + post_norm + residual
    e. per_layer_input: gate -> act -> * per_layer[i] -> proj -> norm -> residual
    f. hidden_states *= layer_scalar
9. Final norm
10. lm_head projection: [1, 1, 2560] -> [1, 1, 262144]
11. Final softcapping: 30 * tanh(logits / 30)
```

---

## 12. CODE REFERENCES

| Component | File | Lines |
|-----------|------|-------|
| Embedding | modeling_gemma4.py | 1608-1610, 1465-1476 |
| PLE (token) | modeling_gemma4.py | 1744-1786 |
| PLE (context) | modeling_gemma4.py | 1788-1821 |
| RMSNorm | modeling_gemma4.py | 197-215 |
| Decoder Layer | modeling_gemma4.py | 1376-1462 |
| Attention | modeling_gemma4.py | 1183-1296 |
| RoPE | modeling_gemma4.py | 1093-1180 |
| MLP | modeling_gemma4.py | 1074-1090 |
| Final output | modeling_gemma4.py | 1736, 1836, 1896-1899 |
| Config | configuration_gemma4.py | 87-217 |
| Layer types | config.json | text_config.layer_types |

---

## 13. IMPORTANT CAVEATS FOR WGSL IMPLEMENTATION

1. **No explicit SRQ apply**: Weights assumed dequantized; inference in BF16
2. **Softmax in fp32**: Compute in float32, cast result back to BF16
3. **RMSNorm internal fp32**: Each norm computation temporarily casts to fp32
4. **RoPE frequencies**: Pre-compute once, store as float32 (or BF16 if sufficient precision)
5. **KV cache**: Single-token decode doesn't need cache for *current* token, but must append for next token
6. **Sliding window**: Only applies to sliding_attention layers (indices 0-4, 6-10, etc.); last layer (41) always full attention
7. **Layer scalar**: Trivial (usually 1.0) but must apply after each layer
8. **GQA (8q/2kv)**: Must repeat KV heads 4x to match query heads
9. **Position IDs**: Incremental (0, 1, 2, ...) for autoregressive decode
10. **Tied weights**: lm_head.weight == embed_tokens.weight (same parameter, not duplicated)

