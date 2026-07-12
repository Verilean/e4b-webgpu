// A4B layer tail, fused (replaces moecomb + rms(postFfw1) + rms(postFfw2) +
// acc + rmsaccMul = 5 dispatches):
//   mlp_i  = rms(t1)·w1_i
//   moe_i  = rms(m)·w2_i where m_i = Σ_k tkw[k]·slots[k*H+i]
//   comb_i = mlp_i + moe_i
//   hidden = (hidden + rms(comb)·wp) * MUL
// One WG; staged vectors in workgroup memory. Params: H, K, EPS, MUL, WG
enable subgroups;
@group(0) @binding(0) var<storage, read> t1: array<f32>;       // dense-down out
@group(0) @binding(1) var<storage, read> w1: array<f32>;       // post_ffw_norm_1
@group(0) @binding(2) var<storage, read> slots: array<f32>;    // [K][H] expert downs
@group(0) @binding(3) var<storage, read> tkw: array<f32>;      // [K]
@group(0) @binding(4) var<storage, read> w2: array<f32>;       // post_ffw_norm_2
@group(0) @binding(5) var<storage, read> wp: array<f32>;       // post_ffw_norm
@group(0) @binding(6) var<storage, read_write> hidden: array<f32>;
@group(0) @binding(7) var<storage, read> wNext: array<f32>;    // next attn_norm
@group(0) @binding(8) var<storage, read_write> yNext: array<f32>;
var<workgroup> sg8: array<f32, 8>;
var<workgroup> comb: array<f32, ${H}>;
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
  // rms(t1) and rms(m) — two reductions (m recomputed cheaply per element)
  var s1: f32 = 0.0;
  var s2: f32 = 0.0;
  for (var i = lid; i < ${H}u; i = i + ${WG}u) {
    let a = t1[i];
    s1 = s1 + a * a;
    var m: f32 = 0.0;
    for (var k: u32 = 0u; k < ${K}u; k = k + 1u) { m = m + tkw[k] * slots[k * ${H}u + i]; }
    comb[i] = m;                                 // stash moe-combined
    s2 = s2 + m * m;
  }
  let r1 = redAdd(lid, s1);
  let r2 = redAdd(lid, s2);
  let inv1 = pow(r1 / f32(${H}u) + ${EPS}, -0.5);
  let inv2 = pow(r2 / f32(${H}u) + ${EPS}, -0.5);
  var s3: f32 = 0.0;
  for (var i = lid; i < ${H}u; i = i + ${WG}u) {
    let c = t1[i] * inv1 * w1[i] + comb[i] * inv2 * w2[i];
    comb[i] = c;
    s3 = s3 + c * c;
  }
  let r3 = redAdd(lid, s3);
  let inv3 = pow(r3 / f32(${H}u) + ${EPS}, -0.5);
  var s4: f32 = 0.0;
  for (var i = lid; i < ${H}u; i = i + ${WG}u) {
    let hn = (hidden[i] + comb[i] * inv3 * wp[i]) * ${MUL};
    hidden[i] = hn;
    comb[i] = hn;
    s4 = s4 + hn * hn;
  }
  workgroupBarrier();
  if (${NEXT}u == 1u) {                       // next layer's input norm, fused
    let r4 = redAdd(lid, s4);
    let inv4 = pow(r4 / f32(${H}u) + ${EPS}, -0.5);
    for (var i = lid; i < ${H}u; i = i + ${WG}u) {
      yNext[i] = comb[i] * inv4 * wNext[i];
    }
  } else {
    _ = wNext[0];
    _ = yNext[0];
  }
}
