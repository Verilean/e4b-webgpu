// MoE router, one token, one WG: scale-less RMS(h) × scale × H^-0.5 → 128-dot
// (f32 weights [E rows × H]) → softmax f32 → top-K → renorm → × perExpertScale.
// Writes topkIdx u32[K] and topkW f32[K] (as two buffers).
// Params: H, E(128), K(8), EPS, WG(128)
enable subgroups;
@group(0) @binding(0) var<storage, read> h: array<f32>;
@group(0) @binding(1) var<storage, read> scaleV: array<f32>;    // [H] router input scale
@group(0) @binding(2) var<storage, read> wr: array<f32>;        // [E][H] router weights
@group(0) @binding(3) var<storage, read> pes: array<f32>;       // [E] per-expert scale
@group(0) @binding(4) var<storage, read_write> topkIdx: array<u32>;
@group(0) @binding(5) var<storage, read_write> topkW: array<f32>;

var<workgroup> sg8: array<f32, 8>;
var<workgroup> scores: array<f32, ${E}>;

@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_id) lid3: vec3<u32>) {
  let lid = lid3.x;
  // rms (scale-less)
  var s: f32 = 0.0;
  for (var i = lid; i < ${H}u; i = i + ${WG}u) { s = s + h[i] * h[i]; }
  let s1 = subgroupAdd(s);
  if ((lid & 31u) == 0u) { sg8[lid / 32u] = s1; }
  workgroupBarrier();
  var tot: f32 = 0.0;
  for (var i: u32 = 0u; i < ${WG}u / 32u; i = i + 1u) { tot = tot + sg8[i]; }
  let inv = pow(tot / f32(${H}u) + ${EPS}, -0.5) * pow(f32(${H}u), -0.5);
  // scores: each thread a few experts (serial dot over H)
  for (var e = lid; e < ${E}u; e = e + ${WG}u) {
    var acc: f32 = 0.0;
    for (var i: u32 = 0u; i < ${H}u; i = i + 1u) {
      acc = acc + h[i] * inv * scaleV[i] * wr[e * ${H}u + i];
    }
    scores[e] = acc;
  }
  workgroupBarrier();
  // thread 0: softmax + top-K (E=128, trivial serial)
  if (lid == 0u) {
    var mx: f32 = -3.0e38;
    for (var e: u32 = 0u; e < ${E}u; e = e + 1u) { mx = max(mx, scores[e]); }
    var sum: f32 = 0.0;
    for (var e: u32 = 0u; e < ${E}u; e = e + 1u) {
      let p = exp(scores[e] - mx);
      scores[e] = p;
      sum = sum + p;
    }
    var wsum: f32 = 0.0;
    for (var k: u32 = 0u; k < ${K}u; k = k + 1u) {
      var bi: u32 = 0u;
      var bv: f32 = -1.0;
      for (var e: u32 = 0u; e < ${E}u; e = e + 1u) {
        if (scores[e] > bv) { bv = scores[e]; bi = e; }
      }
      topkIdx[k] = bi;
      topkW[k] = bv / sum;
      wsum = wsum + bv / sum;
      scores[bi] = -2.0;                    // remove from candidates
    }
    for (var k: u32 = 0u; k < ${K}u; k = k + 1u) {
      topkW[k] = topkW[k] / wsum * pes[topkIdx[k]];
    }
  }
}
