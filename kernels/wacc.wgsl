// MoE slot combine (prefill): moeOut[tok][i] = Σ_k topkW[tok*K+k] · downSlots[(tok*K+k)][i]
// Params: H, K, WG
@group(0) @binding(0) var<storage, read> ds: array<f32>;        // [M*K][H]
@group(0) @binding(1) var<storage, read> tkw: array<f32>;       // [M*K]
@group(0) @binding(2) var<storage, read> mprm: array<u32>;      // [1]=M
@group(0) @binding(3) var<storage, read_write> y: array<f32>;   // [M][H]
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= mprm[1] * ${H}u) { return; }
  let tok = i / ${H}u;
  let j = i % ${H}u;
  var acc: f32 = 0.0;
  for (var k: u32 = 0u; k < ${K}u; k = k + 1u) {
    acc = acc + tkw[tok * ${K}u + k] * ds[(tok * ${K}u + k) * ${H}u + j];
  }
  y[i] = acc;
}
