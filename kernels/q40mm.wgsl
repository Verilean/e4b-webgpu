enable f16;
// Q4_0 batched matvec (prefill): each WG loads its 2 rows' weight blocks ONCE
// and reuses them across MCOLS token columns (grid.z = token chunk). Per-token
// accumulation order (jb ascending, subgroupAdd) is IDENTICAL to q40mv →
// bit-identical activations vs the token-by-token path (the P5 correctness bar).
// mprm[1] = M (token count); tokens ≥ M read stale x but never write.
// Params: IN, OUT, XF16(0/1), WG (32/64: WG/16 rows per WG), MCOLS
enable subgroups;
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> xh: array<vec4<f16>>;   // XF16=1 input
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;     // nibble plane
@group(0) @binding(2) var<storage, read> ws: array<u32>;          // f16 scales, 2/word
@group(0) @binding(3) var<storage, read> mprm: array<u32>;        // [1]=M
@group(0) @binding(4) var<storage, read_write> y: array<f32>;     // [M][OUT]

struct XU {
  e0: vec4f, o0: vec4f, e1: vec4f, o1: vec4f,
  e2: vec4f, o2: vec4f, e3: vec4f, o3: vec4f,
};

// activation lanes matched to the nibble order (lo nibbles = elems 0..15)
fn unpx(xoff: u32, jb: u32) -> XU {
  if (${XF16}u == 1u) {
    let x0 = vec4<f32>(xh[xoff + jb * 8u]);      let x1 = vec4<f32>(xh[xoff + jb * 8u + 1u]);
    let x2 = vec4<f32>(xh[xoff + jb * 8u + 2u]); let x3 = vec4<f32>(xh[xoff + jb * 8u + 3u]);
    let x4 = vec4<f32>(xh[xoff + jb * 8u + 4u]); let x5 = vec4<f32>(xh[xoff + jb * 8u + 5u]);
    let x6 = vec4<f32>(xh[xoff + jb * 8u + 6u]); let x7 = vec4<f32>(xh[xoff + jb * 8u + 7u]);
    return XU(x0, x4, x1, x5, x2, x6, x3, x7);
  }
  let x0 = x[xoff + jb * 8u];      let x1 = x[xoff + jb * 8u + 1u];
  let x2 = x[xoff + jb * 8u + 2u]; let x3 = x[xoff + jb * 8u + 3u];
  let x4 = x[xoff + jb * 8u + 4u]; let x5 = x[xoff + jb * 8u + 5u];
  let x6 = x[xoff + jb * 8u + 6u]; let x7 = x[xoff + jb * 8u + 7u];
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

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
  _ = mprm[0]; _ = x[0]; _ = xh[0];
  let sg = lid3.x / 32u;
  let lane = lid3.x % 32u;
  let o0 = ((wid.y * 32768u + wid.x) * (${WG}u / 32u) + sg) * 2u;
  let valid = o0 < ${OUT}u;                 // no early return (subgroupAdd uniformity)
  let rowB = ${IN}u / 32u;
  let b0 = o0 * rowB;
  let b1 = b0 + rowB;
  let mTot = mprm[1];
  let m0 = wid.z * ${MCOLS}u;
  var acc: array<f32, ${MCOLS} * 2>;
  for (var m: u32 = 0u; m < ${MCOLS}u * 2u; m = m + 1u) { acc[m] = 0.0; }
  if (valid) {
    for (var jb = lane; jb < rowB; jb = jb + 32u) {
      let wv0 = w[b0 + jb];
      let wv1 = w[b1 + jb];
      let s0 = scaleOf(b0, jb);
      let s1 = scaleOf(b1, jb);
      for (var m: u32 = 0u; m < ${MCOLS}u; m = m + 1u) {
        let u = unpx((m0 + m) * (${IN}u / 4u), jb);
        acc[m * 2u] = acc[m * 2u] + s0 * bdot(wv0, u);
        acc[m * 2u + 1u] = acc[m * 2u + 1u] + s1 * bdot(wv1, u);
      }
    }
  }
  for (var m: u32 = 0u; m < ${MCOLS}u; m = m + 1u) {
    let t0 = subgroupAdd(acc[m * 2u]);
    let t1 = subgroupAdd(acc[m * 2u + 1u]);
    let tokOk = m0 + m < mTot;
    if (valid && tokOk && lane == 0u) { y[(m0 + m) * ${OUT}u + o0] = t0; }
    if (valid && tokOk && lane == 1u && o0 + 1u < ${OUT}u) { y[(m0 + m) * ${OUT}u + o0 + 1u] = t1; }
  }
}
