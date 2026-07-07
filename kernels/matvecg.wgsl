// Grouped subgroup matvec (concat of up to 3 linears sharing one norm+quant
// dispatch): rows [0,B0) = group 0, [B0,B1) = group 1, [B1,OUT) = group 2.
// Group g reads xq region g and srqs[g] = (inS, outS). 2 rows/WG interleaved.
// Params: BITS(4), IN, OUT, B0, B1
enable subgroups;
@group(0) @binding(0) var<storage, read> xq: array<vec4<u32>>;   // regions of IN/16
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> wscale: array<f32>;
@group(0) @binding(3) var<storage, read> srqs: array<vec2<f32>>; // [group] (inS,outS)
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

struct XU {
  e0: vec4f, o0: vec4f, e1: vec4f, o1: vec4f,
  e2: vec4f, o2: vec4f, e3: vec4f, o3: vec4f,
};

// Unpack 32 int8 activations (two vec4<u32>) into nibble-order-matched lanes.
// Hoisted to the caller so BOTH row accumulations share ONE set of loads —
// the in-function loads were not CSE'd across calls (LSU carried 2x redundant
// activation traffic, capping weight streaming at ~335 GB/s).
fn unpx(xv0: vec4<u32>, xv1: vec4<u32>) -> XU {
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
  return dot(vec4f(vec4i(unpack4xU8(wv.x & 0x0F0F0F0Fu))) - vec4f(8.0), x.e0)
       + dot(vec4f(vec4i(unpack4xU8((wv.x >> 4u) & 0x0F0F0F0Fu))) - vec4f(8.0), x.o0)
       + dot(vec4f(vec4i(unpack4xU8(wv.y & 0x0F0F0F0Fu))) - vec4f(8.0), x.e1)
       + dot(vec4f(vec4i(unpack4xU8((wv.y >> 4u) & 0x0F0F0F0Fu))) - vec4f(8.0), x.o1)
       + dot(vec4f(vec4i(unpack4xU8(wv.z & 0x0F0F0F0Fu))) - vec4f(8.0), x.e2)
       + dot(vec4f(vec4i(unpack4xU8((wv.z >> 4u) & 0x0F0F0F0Fu))) - vec4f(8.0), x.o2)
       + dot(vec4f(vec4i(unpack4xU8(wv.w & 0x0F0F0F0Fu))) - vec4f(8.0), x.e3)
       + dot(vec4f(vec4i(unpack4xU8((wv.w >> 4u) & 0x0F0F0F0Fu))) - vec4f(8.0), x.o3);
}

fn grp(o: u32) -> u32 {
  if (o < ${B0}u) { return 0u; }
  if (o < ${B1}u) { return 1u; }
  return 2u;
}

fn finish(total: f32, o: u32) {
  let sq = srqs[grp(o)];
  var out = total * sq.x * wscale[o];
  if (sq.y != 0.0) { out = clamp(round(out / sq.y), -128.0, 127.0) * sq.y; }
  y[o] = out;
}

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let o0 = (wid.y * 32768u + wid.x) * 2u;
  if (o0 >= ${OUT}u) { return; }
  let rowW = ${IN}u / 32u;
  let xr = ${IN}u / 16u;                 // xq region stride in vec4s
  let b0 = o0 * rowW;
  // group boundaries are even → a row pair never straddles groups; ONE xq
  // stream for both rows (two streams would defeat load CSE — measured 25% loss)
  let xoff = grp(o0) * xr;
  var acc0: f32 = 0.0;
  var acc1: f32 = 0.0;
  for (var jw = lid.x; jw < rowW; jw = jw + 32u) {
    let x = unpx(xq[xoff + jw * 2u], xq[xoff + jw * 2u + 1u]);
    acc0 = acc0 + wdot(w[b0 + jw], x);
    acc1 = acc1 + wdot(w[b0 + rowW + jw], x);
  }
  let t0 = subgroupAdd(acc0);
  let t1 = subgroupAdd(acc1);
  if (lid.x == 0u) { finish(t0, o0); }
  if (lid.x == 1u && o0 + 1u < ${OUT}u) { finish(t1, o0 + 1u); }
}
