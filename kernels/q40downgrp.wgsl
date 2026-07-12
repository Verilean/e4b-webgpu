enable f16;
enable subgroups;
// GROUPED MoE down-projection (prefill): grid.z = expert chunk (expgroup
// descriptors), WG=64 = 2 subgroups × 2 rows, weights loaded once per chunk
// and reused across MC entry columns. Output per entry goes UNWEIGHTED to
// downSlots[(tok*K+slot)][OUT] (unique per entry — race-free); the wacc pass
// applies topkW and sums the K slots. Params: IN(704), OUT(2816), K, MC, WG(64)
@group(0) @binding(0) var<storage, read> x: array<vec4<f16>>;    // gegluSlots [M*K][IN/4]
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> ws: array<u32>;
@group(0) @binding(3) var<storage, read> chunkExp: array<u32>;
@group(0) @binding(4) var<storage, read> chunkEnt: array<u32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;    // downSlots [M*K][OUT]

struct XU {
  e0: vec4f, o0: vec4f, e1: vec4f, o1: vec4f,
  e2: vec4f, o2: vec4f, e3: vec4f, o3: vec4f,
};
fn unpx(xoff: u32, jb: u32) -> XU {
  let a0 = vec4<f32>(x[xoff + jb * 8u]);      let a1 = vec4<f32>(x[xoff + jb * 8u + 1u]);
  let a2 = vec4<f32>(x[xoff + jb * 8u + 2u]); let a3 = vec4<f32>(x[xoff + jb * 8u + 3u]);
  let a4 = vec4<f32>(x[xoff + jb * 8u + 4u]); let a5 = vec4<f32>(x[xoff + jb * 8u + 5u]);
  let a6 = vec4<f32>(x[xoff + jb * 8u + 6u]); let a7 = vec4<f32>(x[xoff + jb * 8u + 7u]);
  return XU(a0, a4, a1, a5, a2, a6, a3, a7);
}
fn bdot(wv: vec4<u32>, u: XU) -> f32 {
  return dot(vec4f(vec4i(unpack4xU8(wv.x & 0x0F0F0F0Fu))) - vec4f(8.0), u.e0)
       + dot(vec4f(vec4i(unpack4xU8((wv.x >> 4u) & 0x0F0F0F0Fu))) - vec4f(8.0), u.o0)
       + dot(vec4f(vec4i(unpack4xU8(wv.y & 0x0F0F0F0Fu))) - vec4f(8.0), u.e1)
       + dot(vec4f(vec4i(unpack4xU8((wv.y >> 4u) & 0x0F0F0F0Fu))) - vec4f(8.0), u.o1)
       + dot(vec4f(vec4i(unpack4xU8(wv.z & 0x0F0F0F0Fu))) - vec4f(8.0), u.e2)
       + dot(vec4f(vec4i(unpack4xU8((wv.z >> 4u) & 0x0F0F0F0Fu))) - vec4f(8.0), u.o2)
       + dot(vec4f(vec4i(unpack4xU8(wv.w & 0x0F0F0F0Fu))) - vec4f(8.0), u.e3)
       + dot(vec4f(vec4i(unpack4xU8((wv.w >> 4u) & 0x0F0F0F0Fu))) - vec4f(8.0), u.o3);
}
fn scaleOf(base: u32, b: u32) -> f32 {
  let i = base + b;
  let two = unpack2x16float(ws[i / 2u]);
  return select(two.x, two.y, (i & 1u) == 1u);
}

var<workgroup> wgExp: u32;
var<workgroup> ents: array<u32, ${MC}>;

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
  if (lid3.x == 0u) { wgExp = chunkExp[wid.z]; }
  if (lid3.x < ${MC}u) { ents[lid3.x] = chunkEnt[wid.z * ${MC}u + lid3.x]; }
  let ce = workgroupUniformLoad(&wgExp);
  if (ce == 0xFFFFFFFFu) { return; }        // sentinel chunk (uniform exit)

  let sg = lid3.x / 32u;
  let lane = lid3.x % 32u;
  let o0 = ((wid.y * 32768u + wid.x) * (${WG}u / 32u) + sg) * 2u;
  let valid = o0 < ${OUT}u;
  let rowB = ${IN}u / 32u;
  let eb = ce * (${OUT}u * rowB);
  let b0 = eb + o0 * rowB;
  let b1 = b0 + rowB;
  var acc: array<f32, ${MC} * 2>;
  for (var c: u32 = 0u; c < ${MC}u * 2u; c = c + 1u) { acc[c] = 0.0; }
  if (valid) {
    for (var jb = lane; jb < rowB; jb = jb + 32u) {
      let wv0 = w[b0 + jb];
      let wv1 = w[b1 + jb];
      let s0 = scaleOf(b0, jb);
      let s1 = scaleOf(b1, jb);
      for (var c: u32 = 0u; c < ${MC}u; c = c + 1u) {
        let ent = ents[c];
        if (ent != 0xFFFFFFFFu) {
          let u = unpx(ent * (${IN}u / 4u), jb);   // x row = flat entry (tok*K+slot)
          acc[c * 2u] = acc[c * 2u] + s0 * bdot(wv0, u);
          acc[c * 2u + 1u] = acc[c * 2u + 1u] + s1 * bdot(wv1, u);
        }
      }
    }
  }
  for (var c: u32 = 0u; c < ${MC}u; c = c + 1u) {
    let t0 = subgroupAdd(acc[c * 2u]);
    let t1 = subgroupAdd(acc[c * 2u + 1u]);
    let ent = ents[c];
    let ok = valid && ent != 0xFFFFFFFFu;
    if (ok && lane == 0u) { y[ent * ${OUT}u + o0] = t0; }
    if (ok && lane == 1u && o0 + 1u < ${OUT}u) { y[ent * ${OUT}u + o0 + 1u] = t1; }
  }
}
