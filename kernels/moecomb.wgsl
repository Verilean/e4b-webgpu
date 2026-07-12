// Combine 8 expert down-outputs: y[i] = Σ_k topkW[k] · slots[k*H + i]
// (then the caller's norm/residual ops run as usual). Params: H, K, WG
@group(0) @binding(0) var<storage, read> slots: array<f32>;   // [K][H]
@group(0) @binding(1) var<storage, read> topkW: array<f32>;   // [K]
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${H}u) { return; }
  var acc: f32 = 0.0;
  for (var k: u32 = 0u; k < ${K}u; k = k + 1u) {
    acc = acc + topkW[k] * slots[k * ${H}u + i];
  }
  y[i] = acc;
}
