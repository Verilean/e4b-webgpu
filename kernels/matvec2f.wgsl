// Workgroup-cooperative matvec, RAW f32 input (for SRQ-identity linears: lm_head
// inS==0, and the f32 per_layer_model_projection). One WG per output row.
// BITS: 2 (int2, 16/word) or 32 (f32 via bitcast, 1/word).
// y[o] = softcap( SRQ_out( wscale[o] * sum_i x[i]*W[o,i] ) )
// Params: BITS, IN, OUT, WG, SOFTCAP
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;   // IN must be /4
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> wscale: array<f32>;
@group(0) @binding(3) var<uniform> srq: vec2f;             // [inS(unused), outS]
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

var<workgroup> partial: array<f32, ${WG}>;

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let o = wid.y * 32768u + wid.x;
  if (o >= ${OUT}u) { return; }
  var acc: f32 = 0.0;

  if (${BITS}u == 2u) {
    let rowWords = ${IN}u / 16u;                           // 16 int2 per u32 = 4 vec4
    let base = o * rowWords;
    for (var jw = lid.x; jw < rowWords; jw = jw + ${WG}u) {
      let wv = w[base + jw];
      for (var g: u32 = 0u; g < 4u; g = g + 1u) {
        let b = (wv >> (g * 8u)) & 0xFFu;
        let w4 = vec4f(f32(b & 3u), f32((b >> 2u) & 3u), f32((b >> 4u) & 3u), f32((b >> 6u) & 3u)) - vec4f(2.0);
        acc = acc + dot(w4, x[jw * 4u + g]);
      }
    }
  } else if (${BITS}u == 16u) {                            // f16 weights, native unpack
    let rowW = ${IN}u / 8u;                                // vec4<u32> = 8 f16
    let base = o * rowW;
    for (var j = lid.x; j < rowW; j = j + ${WG}u) {
      let wb = (base + j) * 4u;
      let w0 = vec4f(unpack2x16float(w[wb]), unpack2x16float(w[wb + 1u]));
      let w1 = vec4f(unpack2x16float(w[wb + 2u]), unpack2x16float(w[wb + 3u]));
      acc = acc + dot(x[j * 2u], w0) + dot(x[j * 2u + 1u], w1);
    }
  } else {                                                 // BITS == 32 (f32 weights)
    let rowV = ${IN}u / 4u;
    let base = o * rowV;
    for (var j = lid.x; j < rowV; j = j + ${WG}u) {
      let wb = (base + j) * 4u;
      acc = acc + dot(x[j], vec4f(bitcast<f32>(w[wb]), bitcast<f32>(w[wb + 1u]),
                                  bitcast<f32>(w[wb + 2u]), bitcast<f32>(w[wb + 3u])));
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
    var out = partial[0] * wscale[o];
    if (srq.y != 0.0) { out = clamp(round(out / srq.y), -128.0, 127.0) * srq.y; }
    if (${SOFTCAP} != 0.0) { out = ${SOFTCAP} * tanh(out / ${SOFTCAP}); }
    y[o] = out;
  }
}
