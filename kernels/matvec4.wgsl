// Subgroup matvec, R rows interleaved per 32-thread subgroup: one xq load feeds
// R weight rows (cuts x traffic, multiplies loads in flight). No barriers.
// Params: BITS(4/8), IN, OUT, R, SOFTCAP
enable subgroups;
@group(0) @binding(0) var<storage, read> xq: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> wscale: array<f32>;
@group(0) @binding(3) var<uniform> srq: vec2f;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<storage, read> sumI: array<i32>;   // [0] = Σ int8 acts (atomic-produced)

struct XU {
  e0: vec4f, o0: vec4f, e1: vec4f, o1: vec4f,
  e2: vec4f, o2: vec4f, e3: vec4f, o3: vec4f,
};

// Unpack 32 int8 activations (two vec4<u32>) into nibble-order-matched lanes.
// Hoisted to the caller so BOTH row accumulations share ONE set of loads —
// the in-function loads were not CSE'd across calls (LSU carried 2x redundant
// activation traffic, capping weight streaming at ~335 GB/s).
fn unpx(xv0: vec4<u32>, xv1: vec4<u32>) -> XU {
  if (${ZP}u == 1u) {
    let a0 = unpack4x8snorm(xv0.x);
    let a1 = unpack4x8snorm(xv0.y);
    let a2 = unpack4x8snorm(xv0.z);
    let a3 = unpack4x8snorm(xv0.w);
    let b0 = unpack4x8snorm(xv1.x);
    let b1 = unpack4x8snorm(xv1.y);
    let b2 = unpack4x8snorm(xv1.z);
    let b3 = unpack4x8snorm(xv1.w);
    return XU(vec4f(a0.x, a0.z, a1.x, a1.z), vec4f(a0.y, a0.w, a1.y, a1.w),
              vec4f(a2.x, a2.z, a3.x, a3.z), vec4f(a2.y, a2.w, a3.y, a3.w),
              vec4f(b0.x, b0.z, b1.x, b1.z), vec4f(b0.y, b0.w, b1.y, b1.w),
              vec4f(b2.x, b2.z, b3.x, b3.z), vec4f(b2.y, b2.w, b3.y, b3.w));
  }
  let a0 = vec4f(unpack4xI8(xv0.x));
  let a1 = vec4f(unpack4xI8(xv0.y));
  let a2 = vec4f(unpack4xI8(xv0.z));
  let a3 = vec4f(unpack4xI8(xv0.w));
  let b0 = vec4f(unpack4xI8(xv1.x));
  let b1 = vec4f(unpack4xI8(xv1.y));
  let b2 = vec4f(unpack4xI8(xv1.z));
  let b3 = vec4f(unpack4xI8(xv1.w));
  return XU(vec4f(a0.x, a0.z, a1.x, a1.z), vec4f(a0.y, a0.w, a1.y, a1.w),
            vec4f(a2.x, a2.z, a3.x, a3.z), vec4f(a2.y, a2.w, a3.y, a3.w),
            vec4f(b0.x, b0.z, b1.x, b1.z), vec4f(b0.y, b0.w, b1.y, b1.w),
            vec4f(b2.x, b2.z, b3.x, b3.z), vec4f(b2.y, b2.w, b3.y, b3.w));
}

fn wdot(wv: vec4<u32>, x: XU) -> f32 {
  if (${ZP}u == 1u) {
    return dot(unpack4x8unorm(wv.x & 0x0F0F0F0Fu), x.e0)
         + dot(unpack4x8unorm((wv.x >> 4u) & 0x0F0F0F0Fu), x.o0)
         + dot(unpack4x8unorm(wv.y & 0x0F0F0F0Fu), x.e1)
         + dot(unpack4x8unorm((wv.y >> 4u) & 0x0F0F0F0Fu), x.o1)
         + dot(unpack4x8unorm(wv.z & 0x0F0F0F0Fu), x.e2)
         + dot(unpack4x8unorm((wv.z >> 4u) & 0x0F0F0F0Fu), x.o2)
         + dot(unpack4x8unorm(wv.w & 0x0F0F0F0Fu), x.e3)
         + dot(unpack4x8unorm((wv.w >> 4u) & 0x0F0F0F0Fu), x.o3);
  }
  return dot(vec4f(vec4i(unpack4xU8(wv.x & 0x0F0F0F0Fu))) - vec4f(8.0), x.e0)
       + dot(vec4f(vec4i(unpack4xU8((wv.x >> 4u) & 0x0F0F0F0Fu))) - vec4f(8.0), x.o0)
       + dot(vec4f(vec4i(unpack4xU8(wv.y & 0x0F0F0F0Fu))) - vec4f(8.0), x.e1)
       + dot(vec4f(vec4i(unpack4xU8((wv.y >> 4u) & 0x0F0F0F0Fu))) - vec4f(8.0), x.o1)
       + dot(vec4f(vec4i(unpack4xU8(wv.z & 0x0F0F0F0Fu))) - vec4f(8.0), x.e2)
       + dot(vec4f(vec4i(unpack4xU8((wv.z >> 4u) & 0x0F0F0F0Fu))) - vec4f(8.0), x.o2)
       + dot(vec4f(vec4i(unpack4xU8(wv.w & 0x0F0F0F0Fu))) - vec4f(8.0), x.e3)
       + dot(vec4f(vec4i(unpack4xU8((wv.w >> 4u) & 0x0F0F0F0Fu))) - vec4f(8.0), x.o3);
}

fn finish(total: f32, o: u32) {
  var out = total * srq.x * wscale[o];
  if (srq.y != 0.0) { out = clamp(round(out / srq.y), -128.0, 127.0) * srq.y; }
  if (${SOFTCAP} != 0.0) { out = ${SOFTCAP} * tanh(out / ${SOFTCAP}); }
  y[o] = out;
}

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  // SER serial passes of the proven 2-row interleave (2*SER rows per WG):
  // longer per-thread streams pipeline better than many short WGs.
  _ = sumI[0];                              // keep binding in BITS8 layouts (DCE)
  let og = (wid.y * 32768u + wid.x) * (2u * ${SER}u);
  if (og >= ${OUT}u) { return; }
  let rowW = select(${IN}u / 16u, ${IN}u / 32u, ${BITS}u == 4u);
  for (var rr: u32 = 0u; rr < ${SER}u; rr = rr + 1u) {
    let o0 = og + rr * 2u;
    if (o0 >= ${OUT}u) { continue; }
    let b0 = o0 * rowW;
    let b1 = b0 + rowW;
    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    for (var jw = lid.x; jw < rowW; jw = jw + 32u) {
      if (${BITS}u == 4u) {
        let x = unpx(xq[jw * 2u], xq[jw * 2u + 1u]);
        acc0 = acc0 + wdot(w[b0 + jw], x);
        acc1 = acc1 + wdot(w[b1 + jw], x);
      } else {
        let xv = xq[jw];
        let x0 = vec4f(unpack4xI8(xv.x));
        let x1 = vec4f(unpack4xI8(xv.y));
        let x2 = vec4f(unpack4xI8(xv.z));
        let x3 = vec4f(unpack4xI8(xv.w));
        let wa = w[b0 + jw];
        let wb = w[b1 + jw];
        acc0 = acc0 + dot(x0, vec4f(unpack4xI8(wa.x))) + dot(x1, vec4f(unpack4xI8(wa.y)))
                    + dot(x2, vec4f(unpack4xI8(wa.z))) + dot(x3, vec4f(unpack4xI8(wa.w)));
        acc1 = acc1 + dot(x0, vec4f(unpack4xI8(wb.x))) + dot(x1, vec4f(unpack4xI8(wb.y)))
                    + dot(x2, vec4f(unpack4xI8(wb.z))) + dot(x3, vec4f(unpack4xI8(wb.w)));
      }
    }
    let t0 = subgroupAdd(acc0);
    let t1 = subgroupAdd(acc1);
    if (lid.x == 0u) { finish(t0, o0); }
    if (lid.x == 1u && o0 + 1u < ${OUT}u) { finish(t1, o0 + 1u); }
  }
}
