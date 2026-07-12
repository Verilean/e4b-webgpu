// Per-slot geglu for MoE: out[k*FF + i] = gelu_tanh(gu[k*2FF + i]) * gu[k*2FF + FF + i]
// Params: FF, K, WG
@group(0) @binding(0) var<storage, read> gu: array<f32>;
@group(0) @binding(1) var<storage, read_write> outv: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${K}u * ${FF}u) { return; }
  let k = i / ${FF}u;
  let j = i % ${FF}u;
  let x = gu[k * 2u * ${FF}u + j];
  outv[i] = 0.5 * x * (1.0 + tanh(clamp(0.7978845608028654 * (x + 0.044715 * x*x*x), -20.0, 20.0))) * gu[k * 2u * ${FF}u + ${FF}u + j];
}
