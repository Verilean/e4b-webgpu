// Router scores + top-K in ONE dispatch: 8 subgroups × 16 expert rows each,
// 32-lane cooperative dots (f32 router weights), then thread-0 softmax/top-K.
// Params: H, E(128), K(8), WG(256)
enable subgroups;
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;   // routerIn (pre-scaled)
@group(0) @binding(1) var<storage, read> wr: array<vec4<f32>>;  // [E][H/4]
@group(0) @binding(2) var<storage, read> pes: array<f32>;
@group(0) @binding(3) var<storage, read_write> topkIdx: array<u32>;
@group(0) @binding(4) var<storage, read_write> topkW: array<f32>;
var<workgroup> p: array<f32, ${E}>;
@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_id) lid3: vec3<u32>) {
  let lid = lid3.x;
  let sg = lid / 32u;
  let lane = lid % 32u;
  let h4 = ${H}u / 4u;
  let perSg = ${E}u / (${WG}u / 32u);
  for (var r: u32 = 0u; r < perSg; r = r + 1u) {
    let e = sg * perSg + r;
    var acc: f32 = 0.0;
    for (var j = lane; j < h4; j = j + 32u) {
      acc = acc + dot(x[j], wr[e * h4 + j]);
    }
    let t = subgroupAdd(acc);
    if (lane == 0u) { p[e] = t; }
  }
  workgroupBarrier();
  if (lid == 0u) {
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
