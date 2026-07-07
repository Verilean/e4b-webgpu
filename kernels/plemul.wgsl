// y[i] = gelu_pytorch_tanh(a[i]) * ple[OFF + i]  (PLE gate × this layer's slice)
// Params: N, OFF, WG
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> ple: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${N}u) { return; }
  let x = a[i];
  y[i] = 0.5 * x * (1.0 + tanh(clamp(0.7978845608028654 * (x + 0.044715 * x*x*x), -20.0, 20.0))) * ple[${OFF}u + i];
}
