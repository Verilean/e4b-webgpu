enable f16;
enable subgroups;
// GROUPED MoE gate+up+geglu (prefill): grid.z = expert chunk (from expgroup);
// the WG loads each of its 4 gate + 4 up weight rows ONCE and reuses them
// across the chunk's MC entry columns. Per-entry math (jb order, bdot,
// subgroupAdd, gelu expression) is IDENTICAL to q40gu → bit-identical outputs.
// WG=128 = 4 subgroups: sg0-1 gate rows (R2), sg2-3 up rows, 4 FFN elems/WG.
// Params: IN, FF(704), K, MC, WG(128)
@group(0) @binding(0) var<storage, read> x: array<vec4<f16>>;    // moeIn [M][IN/4]
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> ws: array<u32>;
@group(0) @binding(3) var<storage, read> chunkExp: array<u32>;
@group(0) @binding(4) var<storage, read> chunkEnt: array<u32>;
@group(0) @binding(5) var<storage, read_write> yh: array<f16>;   // gegluSlots [M][K][FF]

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
var<workgroup> vals: array<f32, ${MC} * 8>;    // [col][gate0..3, up0..3]

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  if (lid.x == 0u) { wgExp = chunkExp[wid.z]; }
  if (lid.x < ${MC}u) { ents[lid.x] = chunkEnt[wid.z * ${MC}u + lid.x]; }
  let ce = workgroupUniformLoad(&wgExp);
  if (ce == 0xFFFFFFFFu) { return; }        // sentinel chunk (uniform exit)

  let e0 = (wid.y * 32768u + wid.x) * 4u;
  let valid = e0 < ${FF}u;
  let rowB = ${IN}u / 32u;
  let eb = ce * (2u * ${FF}u * rowB);
  let sg = lid.x / 32u;
  let lane = lid.x % 32u;
  let isUp = sg >= 2u;
  let r0 = select(e0 + (sg % 2u) * 2u, ${FF}u + e0 + (sg % 2u) * 2u, isUp);
  let b0 = eb + r0 * rowB;
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
        if (ent != 0xFFFFFFFFu) {          // sentinel columns skip ALL work
          let u = unpx((ent / ${K}u) * (${IN}u / 4u), jb);
          acc[c * 2u] = acc[c * 2u] + s0 * bdot(wv0, u);
          acc[c * 2u + 1u] = acc[c * 2u + 1u] + s1 * bdot(wv1, u);
        }
      }
    }
  }
  for (var c: u32 = 0u; c < ${MC}u; c = c + 1u) {
    let t0 = subgroupAdd(acc[c * 2u]);
    let t1 = subgroupAdd(acc[c * 2u + 1u]);
    if (lane < 2u) {
      vals[c * 8u + sg * 2u + lane] = select(t0, t1, lane == 1u);
    }
  }
  workgroupBarrier();
  // epilogue: 32 threads = MC cols × 4 elems (same gelu expression as q40gu)
  if (valid && lid.x < ${MC}u * 4u) {
    let c = lid.x / 4u;
    let k = lid.x % 4u;
    let ent = ents[c];
    if (ent != 0xFFFFFFFFu) {
      let g = vals[c * 8u + k];
      let gel = 0.5 * g * (1.0 + tanh(clamp(0.7978845608028654 * (g + 0.044715 * g*g*g), -20.0, 20.0)));
      let tok = ent / ${K}u;
      let slot = ent % ${K}u;
      yh[(tok * ${K}u + slot) * ${FF}u + e0 + k] = f16(gel * vals[c * 8u + 4u + k]);
    }
  }
}
