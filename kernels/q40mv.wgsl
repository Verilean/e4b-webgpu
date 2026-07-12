// Q4_0 matvec, f32 activations: y[o] = Σ_blocks d_b · Σ_{i∈b}(v_i−8)·x_i
// Repacked layout: nibble plane (16B per 32-elem block; one vec4<u32> load = one
// block) + f16 scale plane (2 per u32, unpack2x16float). 2-row interleaved
// subgroup shape (campaign-1 matvec4). EXPERT=1: weight/scale bases offset by
// topk[wid.z]·stride (MoE slot indirection, no CPU readback).
// Params: IN, OUT, EXPERT(0/1), XSLOT(0/1: per-slot x offset), WG (32/64/128: WG/16 rows per WG)
enable subgroups;
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;     // nibble plane
@group(0) @binding(2) var<storage, read> ws: array<u32>;          // f16 scales, 2/word
@group(0) @binding(3) var<storage, read> topk: array<u32>;        // [8] expert ids (EXPERT=1)
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

struct XU {
  e0: vec4f, o0: vec4f, e1: vec4f, o1: vec4f,
  e2: vec4f, o2: vec4f, e3: vec4f, o3: vec4f,
};

// activation lanes matched to the nibble order (lo nibbles = even elements)
fn unpx(xoff: u32, jb: u32) -> XU {
  let x0 = x[xoff + jb * 8u];      let x1 = x[xoff + jb * 8u + 1u];
  let x2 = x[xoff + jb * 8u + 2u]; let x3 = x[xoff + jb * 8u + 3u];
  let x4 = x[xoff + jb * 8u + 4u]; let x5 = x[xoff + jb * 8u + 5u];
  let x6 = x[xoff + jb * 8u + 6u]; let x7 = x[xoff + jb * 8u + 7u];
  // q4_0 within-block order: nibble k of byte j = elements j (lo) and j+16 (hi)
  // byte j of word m covers elements 4m+j… lo-plane = elems 0..15, hi = 16..31
  return XU(vec4f(x0.x, x0.y, x0.z, x0.w), vec4f(x4.x, x4.y, x4.z, x4.w),
            vec4f(x1.x, x1.y, x1.z, x1.w), vec4f(x5.x, x5.y, x5.z, x5.w),
            vec4f(x2.x, x2.y, x2.z, x2.w), vec4f(x6.x, x6.y, x6.z, x6.w),
            vec4f(x3.x, x3.y, x3.z, x3.w), vec4f(x7.x, x7.y, x7.z, x7.w));
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
  let word = ws[i / 2u];
  let two = unpack2x16float(word);
  return select(two.x, two.y, (i & 1u) == 1u);
}

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
  _ = topk[0];                              // keep binding when EXPERT=0 (DCE)
  let sg = lid3.x / 32u;
  let lane = lid3.x % 32u;
  let o0 = ((wid.y * 32768u + wid.x) * (${WG}u / 32u) + sg) * 2u;
  let valid = o0 < ${OUT}u;                 // no early return: keeps subgroupAdd
                                            // in (Tint-provable) uniform flow
  let rowB = ${IN}u / 32u;                    // blocks per row
  var eb: u32 = 0u;                           // expert offset in blocks
  var yb: u32 = 0u;                           // output offset
  if (${EXPERT}u == 1u) {
    let e = topk[wid.z];
    eb = e * (${OUT}u * rowB);
    yb = wid.z * ${OUT}u;
  }
  var xoff: u32 = 0u;
  if (${XSLOT}u == 1u) { xoff = wid.z * (${IN}u / 4u); }
  let b0 = eb + o0 * rowB;
  let b1 = b0 + rowB;
  var acc0: f32 = 0.0;
  var acc1: f32 = 0.0;
  if (valid) {
    for (var jb = lane; jb < rowB; jb = jb + 32u) {
      let u = unpx(xoff, jb);
      acc0 = acc0 + scaleOf(b0, jb) * bdot(w[b0 + jb], u);
      acc1 = acc1 + scaleOf(b1, jb) * bdot(w[b1 + jb], u);
    }
  }
  let t0 = subgroupAdd(acc0);
  let t1 = subgroupAdd(acc1);
  if (valid && lane == 0u) { y[yb + o0] = t0; }
  if (valid && lane == 1u && o0 + 1u < ${OUT}u) { y[yb + o0 + 1u] = t1; }
}
