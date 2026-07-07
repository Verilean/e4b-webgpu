// Quantized matvec: y[o] = SRQ_out( sum_i SRQ_in(x[i]) * W[o,i] * wscale[o] )
// W storage by ${BITS}: 2 -> 4 vals/byte (v-2), 4 -> 2 vals/byte (v-8, low nibble first),
// 8 -> i8, 32 -> f32 plain. Weights row-major [OUT, packed_in]. f32 math (bring-up).
// Params: BITS, IN, OUT, WG, SOFTCAP (0 = none, else softcap value like 30.0)
struct Scales { inS: f32, outS: f32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;        // packed bytes as u32 words (or f32 bits when BITS=32)
@group(0) @binding(2) var<storage, read> wscale: array<f32>;   // [OUT]
@group(0) @binding(3) var<uniform> srq: Scales;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

fn srq_in(v: f32) -> f32 {
  if (srq.inS == 0.0) { return v; }
  return clamp(round(v / srq.inS), -128.0, 127.0) * srq.inS;
}

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let o = gid.x;
  if (o >= ${OUT}u) { return; }
  var acc: f32 = 0.0;
  let IN: u32 = ${IN}u;

  if (${BITS}u == 32u) {
    let base = o * IN;
    for (var i: u32 = 0u; i < IN; i = i + 1u) {
      acc = acc + srq_in(x[i]) * bitcast<f32>(w[base + i]);
    }
  } else if (${BITS}u == 8u) {
    let rowBytes = IN;                       // 1 byte per value
    let base = o * rowBytes;
    for (var i: u32 = 0u; i < IN; i = i + 1u) {
      let byteIdx = base + i;
      let word = w[byteIdx >> 2u];
      let b = (word >> ((byteIdx & 3u) * 8u)) & 0xFFu;
      // sign-extend i8
      let v = f32(i32(b << 24u) >> 24u);
      acc = acc + srq_in(x[i]) * v;
    }
  } else if (${BITS}u == 4u) {
    let rowBytes = IN / 2u;
    let base = o * rowBytes;
    for (var j: u32 = 0u; j < rowBytes; j = j + 1u) {
      let byteIdx = base + j;
      let word = w[byteIdx >> 2u];
      let b = (word >> ((byteIdx & 3u) * 8u)) & 0xFFu;
      let lo = f32(i32(b & 0xFu)) - 8.0;
      let hi = f32(i32(b >> 4u)) - 8.0;
      acc = acc + srq_in(x[2u*j]) * lo + srq_in(x[2u*j + 1u]) * hi;
    }
  } else { // BITS == 2
    let rowBytes = IN / 4u;
    let base = o * rowBytes;
    for (var j: u32 = 0u; j < rowBytes; j = j + 1u) {
      let byteIdx = base + j;
      let word = w[byteIdx >> 2u];
      let b = (word >> ((byteIdx & 3u) * 8u)) & 0xFFu;
      let v0 = f32(i32(b & 3u)) - 2.0;
      let v1 = f32(i32((b >> 2u) & 3u)) - 2.0;
      let v2 = f32(i32((b >> 4u) & 3u)) - 2.0;
      let v3 = f32(i32(b >> 6u)) - 2.0;
      acc = acc + srq_in(x[4u*j]) * v0 + srq_in(x[4u*j+1u]) * v1
                + srq_in(x[4u*j+2u]) * v2 + srq_in(x[4u*j+3u]) * v3;
    }
  }

  var out = acc * wscale[o];
  if (srq.outS != 0.0) {
    out = clamp(round(out / srq.outS), -128.0, 127.0) * srq.outS;
  }
  if (${SOFTCAP} != 0.0) {
    out = ${SOFTCAP} * tanh(out / ${SOFTCAP});
  }
  y[o] = out;
}
