enable f16;
// Fused gate+up+geglu over q4_0 (f32 out): WG=128 = 4 subgroups; the WG owns 4
// FFN elements e0..e0+3 (gate rows via sg 0-1, up rows via sg 2-3, R2 each).
// EXPERT=1: expert base from topk[wid.z]. (A top8-absorbing prologue variant
// was measured NEUTRAL — redundant per-WG top8 ≈ the saved fence — REJECTED.)
// Params: IN, FF (expert rows), FF2 (dense rows), E, K, EXPERT(2 = merged:
// grid z==0 dense from x2/w2→yh2, z>=1 expert slot z-1 from x/w→yh)
enable subgroups;
@group(0) @binding(0) var<storage, read> x: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> ws: array<u32>;
@group(0) @binding(3) var<storage, read> topk: array<u32>;
@group(0) @binding(6) var<storage, read> x2: array<vec4<f16>>;   // dense input (EXPERT=2)
@group(0) @binding(7) var<storage, read> w2: array<vec4<u32>>;
@group(0) @binding(8) var<storage, read> ws2: array<u32>;
@group(0) @binding(9) var<storage, read_write> yh2: array<f16>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;      // EXPERT=0
@group(0) @binding(5) var<storage, read_write> yh: array<f16>;     // EXPERT=1 (halves
                                                                   // the moedown x traffic)

struct XU {
  e0: vec4f, o0: vec4f, e1: vec4f, o1: vec4f,
  e2: vec4f, o2: vec4f, e3: vec4f, o3: vec4f,
};
fn unpx(dense: bool, xoff: u32, jb: u32) -> XU {
  if (dense) {
    let a0 = vec4<f32>(x2[xoff + jb * 8u]);      let a1 = vec4<f32>(x2[xoff + jb * 8u + 1u]);
    let a2 = vec4<f32>(x2[xoff + jb * 8u + 2u]); let a3 = vec4<f32>(x2[xoff + jb * 8u + 3u]);
    let a4 = vec4<f32>(x2[xoff + jb * 8u + 4u]); let a5 = vec4<f32>(x2[xoff + jb * 8u + 5u]);
    let a6 = vec4<f32>(x2[xoff + jb * 8u + 6u]); let a7 = vec4<f32>(x2[xoff + jb * 8u + 7u]);
    return XU(a0, a4, a1, a5, a2, a6, a3, a7);
  }
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
fn scaleOf2(base: u32, b: u32) -> f32 {
  let i = base + b;
  let two = unpack2x16float(ws2[i / 2u]);
  return select(two.x, two.y, (i & 1u) == 1u);
}

var<workgroup> vals: array<f32, 8>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  _ = topk[0]; _ = y[0]; _ = yh[0]; _ = x[0]; _ = x2[0]; _ = w2[0]; _ = ws2[0]; _ = yh2[0];
  // EXPERT=2 merged grid: z==0 = dense (FF2 rows, x2/w2 → yh2); z>=1 = expert
  // slot z-1 (FF rows, x/w → yh). Other EXPERT values: single-role dispatch.
  // BATCH=1 (prefill, EXPERT=2 only): wid.z = token*(1+K)+slot; per-token rows.
  var bTok: u32 = 0u;
  var zz = wid.z;
  if (${BATCH}u == 1u) { bTok = wid.z / (1u + ${K}u); zz = wid.z % (1u + ${K}u); }
  let dense = ${EXPERT}u == 0u || (${EXPERT}u == 2u && zz == 0u);
  let slot = select(zz - 1u, zz, ${EXPERT}u == 1u);   // expert slot index
  let xoff = bTok * (${IN}u / 4u);       // x and x2 share the IN(=hidden) row stride
  let ff = select(${FF}u, ${FF2}u, dense);
  let e0 = (wid.y * 32768u + wid.x) * 4u;
  let valid = e0 < ff;
  let rowB = ${IN}u / 32u;
  var eb: u32 = 0u;
  var yb: u32 = 0u;
  if (!dense) {
    eb = topk[bTok * ${K}u + slot] * (2u * ${FF}u * rowB);
    yb = bTok * ${K}u * ${FF}u + slot * ${FF}u;
  }
  let sg = lid.x / 32u;
  let lane = lid.x % 32u;
  let isUp = sg >= 2u;
  let r0 = select(e0 + (sg % 2u) * 2u, ff + e0 + (sg % 2u) * 2u, isUp);
  let b0 = eb + r0 * rowB;
  let b1 = b0 + rowB;
  var acc0: f32 = 0.0;
  var acc1: f32 = 0.0;
  if (valid) {
    for (var jb = lane; jb < rowB; jb = jb + 32u) {
      let u = unpx(dense, xoff, jb);
      if (dense) {
        acc0 = acc0 + scaleOf2(b0, jb) * bdot(w2[b0 + jb], u);
        acc1 = acc1 + scaleOf2(b1, jb) * bdot(w2[b1 + jb], u);
      } else {
        acc0 = acc0 + scaleOf(b0, jb) * bdot(w[b0 + jb], u);
        acc1 = acc1 + scaleOf(b1, jb) * bdot(w[b1 + jb], u);
      }
    }
  }
  let t0 = subgroupAdd(acc0);
  let t1 = subgroupAdd(acc1);
  if (lane < 2u) {
    vals[sg * 2u + lane] = select(t0, t1, lane == 1u);
  }
  workgroupBarrier();
  if (valid && lid.x == 0u) {
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
      let g = vals[k];
      let gel = 0.5 * g * (1.0 + tanh(clamp(0.7978845608028654 * (g + 0.044715 * g*g*g), -20.0, 20.0)));
      if (dense) { yh2[bTok * ${FF2}u + e0 + k] = f16(gel * vals[4u + k]); }
      else { yh[yb + e0 + k] = f16(gel * vals[4u + k]); }
    }
  }
}
