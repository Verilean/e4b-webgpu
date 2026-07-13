enable f16;
// MoE down-projection with TERNARY experts, slot-combined (t2 analog of
// q40moedown): 2-bit planes (16 vals/u32, shift-grouped), ONE f16 scale per
// expert row. Params: IN(704), OUT(2816), K(8)
enable subgroups;
@group(0) @binding(0) var<storage, read> x: array<vec4<f16>>;    // [K][IN/4]
@group(0) @binding(1) var<storage, read> w: array<u32>;          // t2 plane
@group(0) @binding(2) var<storage, read> ws: array<u32>;         // row scales f16 x2
@group(0) @binding(3) var<storage, read> topk: array<u32>;
@group(0) @binding(4) var<storage, read> tkw: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;

fn rowScale(i: u32) -> f32 {
  let two = unpack2x16float(ws[i / 2u]);
  return select(two.x, two.y, (i & 1u) == 1u);
}
fn t2dot16x(wv: u32, q: u32) -> f32 {
  return dot(vec4f(unpack4xU8(wv & 0x03030303u)) - vec4f(1.0), vec4<f32>(x[q]))
       + dot(vec4f(unpack4xU8((wv >> 2u) & 0x03030303u)) - vec4f(1.0), vec4<f32>(x[q + 1u]))
       + dot(vec4f(unpack4xU8((wv >> 4u) & 0x03030303u)) - vec4f(1.0), vec4<f32>(x[q + 2u]))
       + dot(vec4f(unpack4xU8((wv >> 6u) & 0x03030303u)) - vec4f(1.0), vec4<f32>(x[q + 3u]));
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
  let sg = lid3.x / 32u;
  let lane = lid3.x % 32u;
  let o0 = ((wid.y * 32768u + wid.x) * 2u + sg) * 2u;
  let valid = o0 < ${OUT}u;
  let words = ${IN}u / 16u;
  var acc0: f32 = 0.0;
  var acc1: f32 = 0.0;
  for (var k: u32 = 0u; k < select(0u, ${K}u, valid); k = k + 1u) {
    let rbase = topk[k] * ${OUT}u;
    let xoff = k * (${IN}u / 4u);
    let b0 = (rbase + o0) * words;
    let b1 = b0 + words;
    var s0: f32 = 0.0;
    var s1: f32 = 0.0;
    for (var jw = lane; jw < words; jw = jw + 32u) {
      s0 = s0 + t2dot16x(w[b0 + jw], xoff + jw * 4u);
      s1 = s1 + t2dot16x(w[b1 + jw], xoff + jw * 4u);
    }
    let tw = tkw[k];
    acc0 = acc0 + tw * s0 * rowScale(rbase + o0);
    acc1 = acc1 + tw * s1 * rowScale(rbase + o0 + 1u);
  }
  let t0 = subgroupAdd(acc0);
  let t1 = subgroupAdd(acc1);
  if (valid && lane == 0u) { y[o0] = t0; }
  if (valid && lane == 1u && o0 + 1u < ${OUT}u) { y[o0 + 1u] = t1; }
}
