// Workgroup-cooperative quantized matvec (M4): one WG per output row.
// Input = pre-quantized int8 activation (xq, 4/u32 from srq8.wgsl).
// y[o] = SRQ_out( inS * wscale[o] * sum_i xq[i] * W[o,i] )
// Integer products accumulated in f32 (exact below 2^24).
// ${BITS}: 4 (2 vals/byte, low nibble first, -8) or 8 (i8).
// Params: BITS, IN, OUT, WG, SOFTCAP
@group(0) @binding(0) var<storage, read> xq: array<vec4<u32>>;   // int8×16 per elem
@group(0) @binding(1) var<storage, read> w: array<vec2<u32>>;    // 8-byte loads
@group(0) @binding(2) var<storage, read> wscale: array<f32>;
@group(0) @binding(3) var<uniform> srq: vec2f;             // [inS, outS]
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

var<workgroup> partial: array<f32, ${WG}>;



@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let o = wid.y * 32768u + wid.x;          // 2D grid: rows can exceed 65535
  if (o >= ${OUT}u) { return; }            // whole-WG uniform exit (excess grid)
  var acc: f32 = 0.0;

  if (${BITS}u == 4u) {
    // 8B weight load = 16 int4 values ↔ one vec4<u32> xq load (16 int8).
    let rowW2 = ${IN}u / 16u;
    let base = o * rowW2;
    for (var jw = lid.x; jw < rowW2; jw = jw + ${WG}u) {
      let wv = w[base + jw];
      let xv = xq[jw];
      let a0 = vec4f(unpack4xI8(xv.x));
      let a1 = vec4f(unpack4xI8(xv.y));
      let a2 = vec4f(unpack4xI8(xv.z));
      let a3 = vec4f(unpack4xI8(xv.w));
      let lo0 = vec4f(vec4i(unpack4xU8(wv.x & 0x0F0F0F0Fu))) - vec4f(8.0);
      let hi0 = vec4f(vec4i(unpack4xU8((wv.x >> 4u) & 0x0F0F0F0Fu))) - vec4f(8.0);
      let lo1 = vec4f(vec4i(unpack4xU8(wv.y & 0x0F0F0F0Fu))) - vec4f(8.0);
      let hi1 = vec4f(vec4i(unpack4xU8((wv.y >> 4u) & 0x0F0F0F0Fu))) - vec4f(8.0);
      acc = acc + dot(lo0, vec4f(a0.x, a0.z, a1.x, a1.z))
                + dot(hi0, vec4f(a0.y, a0.w, a1.y, a1.w))
                + dot(lo1, vec4f(a2.x, a2.z, a3.x, a3.z))
                + dot(hi1, vec4f(a2.y, a2.w, a3.y, a3.w));
    }
  } else if (${BITS}u == 2u) {
    // 8B weight load = 32 int2 values ↔ two vec4<u32> xq loads
    let rowW2 = ${IN}u / 32u;
    let base = o * rowW2;
    for (var jw = lid.x; jw < rowW2; jw = jw + ${WG}u) {
      let wv = w[base + jw];
      for (var h: u32 = 0u; h < 2u; h = h + 1u) {
        let ww = wv[h];
        let xv = xq[jw * 2u + h];
        for (var g: u32 = 0u; g < 4u; g = g + 1u) {
          let b = (ww >> (g * 8u)) & 0xFFu;
          let w4 = vec4f(f32(b & 3u), f32((b >> 2u) & 3u), f32((b >> 4u) & 3u), f32((b >> 6u) & 3u)) - vec4f(2.0);
          acc = acc + dot(w4, vec4f(unpack4xI8(xv[g])));
        }
      }
    }
  } else { // BITS == 8: 8B load = 8 int8 ↔ half a vec4<u32> xq
    let rowW2 = ${IN}u / 8u;
    let base = o * rowW2;
    for (var jw = lid.x; jw < rowW2; jw = jw + ${WG}u) {
      let wv = w[base + jw];
      let xv = xq[jw / 2u];
      let k = (jw & 1u) * 2u;
      acc = acc + dot(vec4f(unpack4xI8(xv[k])), vec4f(unpack4xI8(wv.x)))
                + dot(vec4f(unpack4xI8(xv[k + 1u])), vec4f(unpack4xI8(wv.y)));
    }
  }

  partial[lid.x] = acc;
  workgroupBarrier();
  var stride = ${WG}u / 2u;
  while (stride > 0u) {
    if (lid.x < stride) { partial[lid.x] = partial[lid.x] + partial[lid.x + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (lid.x == 0u) {
    var out = partial[0] * srq.x * wscale[o];
    if (srq.y != 0.0) {
      out = clamp(round(out / srq.y), -128.0, 127.0) * srq.y;
    }
    if (${SOFTCAP} != 0.0) {
      out = ${SOFTCAP} * tanh(out / ${SOFTCAP});
    }
    y[o] = out;
  }
}
