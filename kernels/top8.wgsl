// Top-K over E router scores: softmax → top-K → renorm → × perExpertScale.
// One tiny WG (serial over E=128; ~µs). Params: E, K
@group(0) @binding(0) var<storage, read> scores: array<f32>;
@group(0) @binding(1) var<storage, read> pes: array<f32>;
@group(0) @binding(2) var<storage, read_write> topkIdx: array<u32>;
@group(0) @binding(3) var<storage, read_write> topkW: array<f32>;
var<workgroup> p: array<f32, ${E}>;
@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  for (var e = lid.x; e < ${E}u; e = e + 32u) { p[e] = scores[e]; }
  workgroupBarrier();
  if (lid.x == 0u) {
    var mx: f32 = -3.0e38;
    for (var e: u32 = 0u; e < ${E}u; e = e + 1u) { mx = max(mx, p[e]); }
    var sum: f32 = 0.0;
    for (var e: u32 = 0u; e < ${E}u; e = e + 1u) { p[e] = exp(p[e] - mx); sum = sum + p[e]; }
    var wsum: f32 = 0.0;
    for (var k: u32 = 0u; k < ${K}u; k = k + 1u) {
      var bi: u32 = 0u; var bv: f32 = -1.0;
      for (var e: u32 = 0u; e < ${E}u; e = e + 1u) {
        if (p[e] > bv) { bv = p[e]; bi = e; }
      }
      topkIdx[k] = bi; topkW[k] = bv / sum; wsum = wsum + bv / sum;
      p[bi] = -2.0;
    }
    for (var k: u32 = 0u; k < ${K}u; k = k + 1u) {
      topkW[k] = topkW[k] / wsum * pes[topkIdx[k]];
    }
  }
}
