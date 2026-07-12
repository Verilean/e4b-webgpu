// One reduction, three scaled outputs (same input x): y1=rms(x)*w1, y2=..w2, y3=..w3
// Replaces the ffn-norm / router-input / pre-ffw-2 triple. Params: DIM, EPS, WG
enable subgroups;
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w1: array<f32>;
@group(0) @binding(2) var<storage, read> w2: array<f32>;
@group(0) @binding(3) var<storage, read> w3: array<f32>;
@group(0) @binding(4) var<storage, read_write> y1: array<f32>;
@group(0) @binding(5) var<storage, read_write> y2: array<f32>;
@group(0) @binding(6) var<storage, read_write> y3: array<f32>;
var<workgroup> sg8: array<f32, 8>;
@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_id) lid3: vec3<u32>) {
  let lid = lid3.x;
  var s: f32 = 0.0;
  for (var i = lid; i < ${DIM}u; i = i + ${WG}u) { s = s + x[i] * x[i]; }
  let s1 = subgroupAdd(s);
  if ((lid & 31u) == 0u) { sg8[lid / 32u] = s1; }
  workgroupBarrier();
  var tot: f32 = 0.0;
  for (var i: u32 = 0u; i < ${WG}u / 32u; i = i + 1u) { tot = tot + sg8[i]; }
  let inv = pow(tot / f32(${DIM}u) + ${EPS}, -0.5);
  for (var i = lid; i < ${DIM}u; i = i + ${WG}u) {
    let v = x[i] * inv;
    y1[i] = v * w1[i];
    y2[i] = v * w2[i];
    y3[i] = v * w3[i];
  }
}
