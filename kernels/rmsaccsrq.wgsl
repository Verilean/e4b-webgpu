// Fused: hidden = (hidden + rms(t)*w1) * MUL; then quantize rms(hidden)*w2
// into NS int8 regions (next block's matvec inputs). One WG, two reductions.
// Params: DIM, NS, EPS, MUL, NORM2 (0: quantize raw hidden, no w2), WG
@group(0) @binding(0) var<storage, read> t: array<f32>;
@group(0) @binding(1) var<storage, read> w1: array<f32>;
@group(0) @binding(2) var<storage, read_write> hidden: array<f32>;
@group(0) @binding(3) var<storage, read> w2: array<f32>;
@group(0) @binding(4) var<storage, read> scales: array<f32>;   // [NS]
@group(0) @binding(5) var<storage, read_write> xq: array<u32>;

var<workgroup> red: array<f32, ${WG}>;
var<workgroup> hs: array<f32, ${DIM}>;    // staged updated hidden (avoids relying
                                          // on storage-visibility within the WG)

fn reduceAdd(lid: u32, v: f32) -> f32 {
  red[lid] = v;
  workgroupBarrier();
  var stride = ${WG}u / 2u;
  while (stride > 0u) {
    if (lid < stride) { red[lid] = red[lid] + red[lid + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}

@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_id) lidv: vec3<u32>) {
  let lid = lidv.x;
  var s: f32 = 0.0;
  for (var i = lid; i < ${DIM}u; i = i + ${WG}u) {
    let v = t[i];
    s = s + v * v;
  }
  let inv1 = pow(reduceAdd(lid, s) / f32(${DIM}u) + ${EPS}, -0.5);
  var s2: f32 = 0.0;
  for (var i = lid; i < ${DIM}u; i = i + ${WG}u) {
    let h = (hidden[i] + t[i] * inv1 * w1[i]) * ${MUL};
    hidden[i] = h;
    hs[i] = h;
    s2 = s2 + h * h;
  }
  workgroupBarrier();
  _ = w2[0];
  var inv2: f32 = 1.0;
  if (${NORM2}u == 1u) {
    inv2 = pow(reduceAdd(lid, s2) / f32(${DIM}u) + ${EPS}, -0.5);
  }
  let words = ${DIM}u / 4u;
  for (var wi = lid; wi < ${NS}u * words; wi = wi + ${WG}u) {
    let sc = scales[wi / words];
    let j = (wi % words) * 4u;
    var packed: u32 = 0u;
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
      var v = hs[j + k] * inv2;
      if (${NORM2}u == 1u) { v = v * w2[j + k]; }
      let q = i32(clamp(round(v / sc), -128.0, 127.0));
      packed = packed | ((u32(q) & 0xFFu) << (k * 8u));
    }
    xq[wi] = packed;
  }
}
