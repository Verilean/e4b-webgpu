// Fused gate+up+geglu+quant v2: WG = 128 = 4 subgroups; subgroups 0-1 compute
// gate rows e0..e0+3 (2 rows each, proven R2 interleave), subgroups 2-3 the
// matching up rows. Epilogue does output-SRQ + geglu + int8 pack of the
// down-proj input word. Removes the separate geglu dispatch and its fences.
// Params: IN, OUT (= inter; weights are 2*OUT rows: [gate; up])
enable subgroups;
@group(0) @binding(0) var<storage, read> xq: array<vec4<u32>>;   // 2 regions of IN/16
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> wscale: array<f32>;     // [2*OUT]
@group(0) @binding(3) var<storage, read> srqs: array<f32>;       // [gIn,gOut,uIn,uOut,dIn]
@group(0) @binding(4) var<storage, read_write> xqOut: array<u32>;

var<workgroup> vals: array<f32, 8>;   // gate y0..y3, up y0..y3

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

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let e0 = (wid.y * 32768u + wid.x) * 4u;      // first FFN element of this WG
  if (e0 >= ${OUT}u) { return; }
  let sg = lid.x / 32u;                        // subgroup index (Apple: size 32)
  let lane = lid.x % 32u;
  let rowW = ${IN}u / 32u;
  let xr = ${IN}u / 16u;
  let isUp = sg >= 2u;
  let r0 = select(e0 + (sg % 2u) * 2u, ${OUT}u + e0 + (sg % 2u) * 2u, isUp);
  let xoff = select(0u, xr, isUp);
  let b0 = r0 * rowW;
  var acc0: f32 = 0.0;
  var acc1: f32 = 0.0;
  for (var jw = lane; jw < rowW; jw = jw + 32u) {
    let x = unpx(xq[xoff + jw * 2u], xq[xoff + jw * 2u + 1u]);
    acc0 = acc0 + wdot(w[b0 + jw], x);
    acc1 = acc1 + wdot(w[b0 + rowW + jw], x);
  }
  let t0 = subgroupAdd(acc0);
  let t1 = subgroupAdd(acc1);
  if (lane < 2u) {
    let t = select(t0, t1, lane == 1u);
    let o = r0 + lane;
    let inS = select(srqs[0], srqs[2], isUp);
    let outS = select(srqs[1], srqs[3], isUp);
    var y = t * inS * wscale[o];
    if (outS != 0.0) { y = clamp(round(y / outS), -128.0, 127.0) * outS; }
    vals[sg * 2u + lane] = y;                  // [g0,g1,g2,g3? no: sg0→0,1 sg1→2,3 sg2→4,5 sg3→6,7]
  }
  workgroupBarrier();
  if (lid.x == 0u) {
    var packed: u32 = 0u;
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
      let x = vals[k];
      let v = 0.5 * x * (1.0 + tanh(clamp(0.7978845608028654 * (x + 0.044715 * x*x*x), -20.0, 20.0))) * vals[4u + k];
      let qv = i32(clamp(round(v / srqs[4]), -128.0, 127.0));
      packed = packed | ((u32(qv) & 0xFFu) << (k * 8u));
    }
    xqOut[(wid.y * 32768u + wid.x)] = packed;
  }
}
