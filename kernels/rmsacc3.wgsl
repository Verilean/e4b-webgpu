enable f16;
// Fused: hidden += rms(t)·wAcc (post-attention residual), then ONE reduction of
// the new hidden feeding three scaled outputs (ffn-norm/router-in/pre-ffw-2).
// Params: DIM, EPS, WG
enable subgroups;
@group(0) @binding(0) var<storage, read> t: array<f32>;
@group(0) @binding(1) var<storage, read> wAcc: array<f32>;
@group(0) @binding(2) var<storage, read> w1: array<f32>;
@group(0) @binding(3) var<storage, read> w2: array<f32>;
@group(0) @binding(4) var<storage, read> w3: array<f32>;
@group(0) @binding(5) var<storage, read> hIn: array<f32>;
@group(0) @binding(9) var<storage, read_write> hOut: array<f32>;
@group(0) @binding(6) var<storage, read_write> y1: array<f16>;
@group(0) @binding(7) var<storage, read_write> y2: array<f32>;
@group(0) @binding(8) var<storage, read_write> y3: array<f16>;
var<workgroup> sg8: array<f32, 8>;
var<workgroup> hs: array<f32, ${DIM}>;
fn redAdd(lid: u32, v: f32) -> f32 {
  let s1 = subgroupAdd(v);
  if ((lid & 31u) == 0u) { sg8[lid / 32u] = s1; }
  workgroupBarrier();
  var tot: f32 = 0.0;
  for (var i: u32 = 0u; i < ${WG}u / 32u; i = i + 1u) { tot = tot + sg8[i]; }
  workgroupBarrier();
  return tot;
}
@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_id) lid3: vec3<u32>) {
  let lid = lid3.x;
  var s: f32 = 0.0;
  for (var i = lid; i < ${DIM}u; i = i + ${WG}u) {
    let v = t[i];
    s = s + v * v;
  }
  let inv1 = pow(redAdd(lid, s) / f32(${DIM}u) + ${EPS}, -0.5);
  var s2: f32 = 0.0;
  for (var i = lid; i < ${DIM}u; i = i + ${WG}u) {
    let h = hIn[i] + t[i] * inv1 * wAcc[i];
    hOut[i] = h;
    hs[i] = h;
    s2 = s2 + h * h;
  }
  workgroupBarrier();
  let inv2 = pow(redAdd(lid, s2) / f32(${DIM}u) + ${EPS}, -0.5);
  for (var i = lid; i < ${DIM}u; i = i + ${WG}u) {
    let v = hs[i] * inv2;
    y1[i] = f16(v * w1[i]);
    y2[i] = v * w2[i];
    y3[i] = f16(v * w3[i]);
  }
}
