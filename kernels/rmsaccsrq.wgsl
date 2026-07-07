enable subgroups;
// Fused: hidden = (hidden + rms(t)*w1) * MUL; then quantize rms(hidden)*w2
// into NS int8 regions (next block's matvec inputs). One WG, two reductions.
// Params: DIM, NS, EPS, MUL, NORM2 (0: quantize raw hidden, no w2), WG
@group(0) @binding(0) var<storage, read> t: array<f32>;
@group(0) @binding(1) var<storage, read> w1: array<f32>;
@group(0) @binding(2) var<storage, read_write> hidden: array<f32>;
@group(0) @binding(3) var<storage, read> w2: array<f32>;
@group(0) @binding(4) var<storage, read> scales: array<f32>;   // [NS]
@group(0) @binding(5) var<storage, read_write> xq: array<u32>;
@group(0) @binding(6) var<storage, read_write> xsum: array<f32>;
@group(0) @binding(7) var<storage, read_write> sumI: array<i32>;

var<workgroup> red: array<f32, ${WG}>;
var<workgroup> sgred: array<f32, 8>;
var<workgroup> hs: array<f32, ${DIM}>;    // staged updated hidden (avoids relying
                                          // on storage-visibility within the WG)

// two-level subgroup reduction: 2 barriers instead of 2*log2(WG)
fn reduceAdd(lid: u32, v: f32) -> f32 {
  let s1 = subgroupAdd(v);
  if ((lid & 31u) == 0u) { sgred[lid / 32u] = s1; }
  workgroupBarrier();
  var total: f32 = 0.0;
  for (var i: u32 = 0u; i < ${WG}u / 32u; i = i + 1u) { total = total + sgred[i]; }
  workgroupBarrier();
  return total;
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
  var rsum: array<f32, ${NS}>;
  for (var r: u32 = 0u; r < ${NS}u; r = r + 1u) { rsum[r] = 0.0; }
  for (var wi = lid; wi < ${NS}u * words; wi = wi + ${WG}u) {
    let r = wi / words;
    let sc = scales[r];
    let j = (wi % words) * 4u;
    var packed: u32 = 0u;
    var ws: f32 = 0.0;
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
      var v = hs[j + k] * inv2;
      if (${NORM2}u == 1u) { v = v * w2[j + k]; }
      let q = i32(clamp(round(v / sc), -127.0, 127.0));
      ws = ws + f32(q);
      packed = packed | ((u32(q) & 0xFFu) << (k * 8u));
    }
    rsum[r] = rsum[r] + ws;
    xq[wi] = packed;
  }
  for (var r: u32 = 0u; r < ${NS}u; r = r + 1u) {
    let t2 = reduceAdd(lid, rsum[r]);
    if (lid == 0u) { xsum[r] = t2; }
  }
  if (lid == 0u) { sumI[0] = 0; }              // zero the atomic Σq slot for the
}
