// MoE expert grouping for batched prefill (mul_mat_id-style), ONE workgroup:
// counting-sort the M×K slot draws by expert into MC-column chunk descriptors.
// chunkExp[c] = expert id of chunk c (0xFFFFFFFF = sentinel chunk);
// chunkEnt[c*MC+j] = entry (tok*K+slot, = the flat topkIdx index) or sentinel.
// Grouped kernels then read each touched expert's weights once per chunk
// instead of once per token. Params: K, MC, E(128), CMAX, WG(256)
@group(0) @binding(0) var<storage, read> topkIdx: array<u32>;   // [M][K]
@group(0) @binding(1) var<storage, read> mprm: array<u32>;      // [1]=M
@group(0) @binding(2) var<storage, read_write> chunkExp: array<u32>;
@group(0) @binding(3) var<storage, read_write> chunkEnt: array<u32>;

var<workgroup> cnt: array<atomic<u32>, ${E}>;
var<workgroup> cbase: array<u32, ${E}>;

@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_id) lid3: vec3<u32>) {
  let lid = lid3.x;
  let total = mprm[1] * ${K}u;
  // 0. clear counters + sentinel-fill the descriptors
  for (var e = lid; e < ${E}u; e = e + ${WG}u) { atomicStore(&cnt[e], 0u); }
  for (var i = lid; i < ${CMAX}u; i = i + ${WG}u) { chunkExp[i] = 0xFFFFFFFFu; }
  for (var i = lid; i < ${CMAX}u * ${MC}u; i = i + ${WG}u) { chunkEnt[i] = 0xFFFFFFFFu; }
  workgroupBarrier();
  // 1. count arrivals per expert
  for (var i = lid; i < total; i = i + ${WG}u) {
    atomicAdd(&cnt[topkIdx[i]], 1u);
  }
  workgroupBarrier();
  // 2. serial scan (128 experts, trivial): chunk bases + chunkExp labels
  if (lid == 0u) {
    var c: u32 = 0u;
    for (var e: u32 = 0u; e < ${E}u; e = e + 1u) {
      cbase[e] = c;
      let n = atomicLoad(&cnt[e]);
      let nc = (n + ${MC}u - 1u) / ${MC}u;
      for (var j: u32 = 0u; j < nc; j = j + 1u) { chunkExp[c + j] = e; }
      c = c + nc;
    }
  }
  workgroupBarrier();
  // 3. scatter entries (arrival order within an expert is irrelevant: each
  // entry's math is column-independent)
  for (var e = lid; e < ${E}u; e = e + ${WG}u) { atomicStore(&cnt[e], 0u); }
  workgroupBarrier();
  for (var i = lid; i < total; i = i + ${WG}u) {
    let e = topkIdx[i];
    let pos = atomicAdd(&cnt[e], 1u);
    chunkEnt[cbase[e] * ${MC}u + pos] = i;
  }
}
