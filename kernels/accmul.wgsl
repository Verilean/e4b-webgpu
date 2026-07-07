// In-place residual add + scalar mul: y[i] = (y[i] + a[i]) * c[0]. Params: N, WG
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> c: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${N}u) { return; }
  y[i] = (y[i] + a[i]) * c[0];
}
