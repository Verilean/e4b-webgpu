enable f16;
// MoE down-projection, slot-combined: each WG owns 2 output rows and iterates
// ALL K slots (expert rows via topk, per-slot input via x offset), emitting the
// topkW-weighted SUM directly — no per-slot output buffer, no combine pass,
// and 2rows×K×blocks lane-iterations (~100% lane utilization vs 69% for the
// per-slot shape with 22-block rows). Params: IN(704), OUT(2816), K(8)
enable subgroups;
@group(0) @binding(0) var<storage, read> x: array<vec4<f16>>;    // [K][IN/4] (f16)
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> ws: array<u32>;
@group(0) @binding(3) var<storage, read> topk: array<u32>;
@group(0) @binding(4) var<storage, read> tkw: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;

struct XU {
  e0: vec4f, o0: vec4f, e1: vec4f, o1: vec4f,
  e2: vec4f, o2: vec4f, e3: vec4f, o3: vec4f,
};
fn unpx(xoff: u32, jb: u32) -> XU {
  let x0 = vec4<f32>(x[xoff + jb * 8u]);      let x1 = vec4<f32>(x[xoff + jb * 8u + 1u]);
  let x2 = vec4<f32>(x[xoff + jb * 8u + 2u]); let x3 = vec4<f32>(x[xoff + jb * 8u + 3u]);
  let x4 = vec4<f32>(x[xoff + jb * 8u + 4u]); let x5 = vec4<f32>(x[xoff + jb * 8u + 5u]);
  let x6 = vec4<f32>(x[xoff + jb * 8u + 6u]); let x7 = vec4<f32>(x[xoff + jb * 8u + 7u]);
  return XU(x0, x4, x1, x5, x2, x6, x3, x7);
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

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
  let sg = lid3.x / 32u;
  let lid = vec3<u32>(lid3.x % 32u, 0u, 0u);
  // BATCH=1 (prefill): wid.z = token; per-token topk/tkw/x/y blocks
  let bTok = select(0u, wid.z, ${BATCH}u == 1u);
  let tb = bTok * ${K}u;
  let o0 = ((wid.y * 32768u + wid.x) * 2u + sg) * 2u;
  let valid = o0 < ${OUT}u;
  let rowB = ${IN}u / 32u;
  var acc0: f32 = 0.0;
  var acc1: f32 = 0.0;
  for (var k: u32 = 0u; k < select(0u, ${K}u, valid); k = k + 1u) {
    let eb = topk[tb + k] * (${OUT}u * rowB);
    let xoff = (tb + k) * (${IN}u / 4u);
    let b0 = eb + o0 * rowB;
    let b1 = b0 + rowB;
    var s0: f32 = 0.0;
    var s1: f32 = 0.0;
    for (var jb = lid.x; jb < rowB; jb = jb + 32u) {
      let u = unpx(xoff, jb);
      s0 = s0 + scaleOf(b0, jb) * bdot(w[b0 + jb], u);
      s1 = s1 + scaleOf(b1, jb) * bdot(w[b1 + jb], u);
    }
    let tw = tkw[tb + k];
    acc0 = acc0 + tw * s0;
    acc1 = acc1 + tw * s1;
  }
  let t0 = subgroupAdd(acc0);
  let t1 = subgroupAdd(acc1);
  let ob = bTok * ${OUT}u;
  if (valid && lid.x == 0u) { y[ob + o0] = t0; }
  if (valid && lid.x == 1u && o0 + 1u < ${OUT}u) { y[ob + o0 + 1u] = t1; }
}
