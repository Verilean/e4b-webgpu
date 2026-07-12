// Router scores + top-K in ONE dispatch via the decoupled last-WG pattern:
// E workgroups each compute one expert's score (subgroup-cooperative dot) and
// atomicStore it; the LAST workgroup to finish (atomic counter) runs the
// softmax/top-K epilogue. Cross-WG visibility is via atomics only (coherent).
// Params: H, E(128), K(8), WG(64)
enable subgroups;
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;    // routerIn
@group(0) @binding(1) var<storage, read> wr: array<vec4<f32>>;   // [E][H/4]
@group(0) @binding(2) var<storage, read> pes: array<f32>;
@group(0) @binding(3) var<storage, read_write> scoresA: array<atomic<u32>>; // [E] bitcast f32
@group(0) @binding(4) var<storage, read_write> ctr: array<atomic<u32>>;     // [1]
@group(0) @binding(5) var<storage, read_write> topkIdx: array<u32>;
@group(0) @binding(6) var<storage, read_write> topkW: array<f32>;

var<workgroup> sg2: array<f32, 2>;
var<workgroup> lastFlag: u32;
var<workgroup> p: array<f32, ${E}>;

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
  let lid = lid3.x;
  let e = wid.x;
  let h4 = ${H}u / 4u;
  var acc: f32 = 0.0;
  for (var j = lid; j < h4; j = j + ${WG}u) {
    acc = acc + dot(x[j], wr[e * h4 + j]);
  }
  let s1 = subgroupAdd(acc);
  if ((lid & 31u) == 0u) { sg2[lid / 32u] = s1; }
  workgroupBarrier();
  if (lid == 0u) {
    var tot: f32 = 0.0;
    for (var i: u32 = 0u; i < ${WG}u / 32u; i = i + 1u) { tot = tot + sg2[i]; }
    atomicStore(&scoresA[e], bitcast<u32>(tot));
    // completion count; the last WG runs the epilogue
    let done = atomicAdd(&ctr[0], 1u) + 1u;
    lastFlag = select(0u, 1u, done == ${E}u);
    if (lastFlag == 1u) { atomicStore(&ctr[0], 0u); }   // reset for the next layer
  }
  workgroupBarrier();
  // load + barrier UNCONDITIONALLY (Tint uniformity); non-last WGs load
  // garbage they never use. Metal storage atomics are device-coherent.
  for (var i = lid; i < ${E}u; i = i + ${WG}u) {
    p[i] = bitcast<f32>(atomicLoad(&scoresA[i]));
  }
  workgroupBarrier();
  if (lastFlag == 1u) {
    if (lid == 0u) {
      var mx: f32 = -3.0e38;
      for (var i: u32 = 0u; i < ${E}u; i = i + 1u) { mx = max(mx, p[i]); }
      var sum: f32 = 0.0;
      for (var i: u32 = 0u; i < ${E}u; i = i + 1u) { p[i] = exp(p[i] - mx); sum = sum + p[i]; }
      var wsum: f32 = 0.0;
      for (var k: u32 = 0u; k < ${K}u; k = k + 1u) {
        var bi: u32 = 0u; var bv: f32 = -1.0;
        for (var i: u32 = 0u; i < ${E}u; i = i + 1u) {
          if (p[i] > bv) { bv = p[i]; bi = i; }
        }
        topkIdx[k] = bi; topkW[k] = bv / sum; wsum = wsum + bv / sum;
        p[bi] = -2.0;
      }
      for (var k: u32 = 0u; k < ${K}u; k = k + 1u) {
        topkW[k] = topkW[k] / wsum * pes[topkIdx[k]];
      }
    }
  }
}
