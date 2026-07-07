// Pre-quantize activation once per matvec input: xq[i] = clamp(round(x[i]/s), -128, 127)
// packed 4×i8 per u32 (little-endian lanes). s==0 → identity is NOT representable in
// int8, so the caller must only use this for calibrated layers (all E4B layers are).
// Params: N, WG
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<uniform> srq: vec2f;            // [inS, outS] (uses inS)
@group(0) @binding(2) var<storage, read_write> xq: array<u32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = gid.x;                       // one u32 word = 4 values
  let n4 = (${N}u + 3u) / 4u;
  if (w >= n4) { return; }
  var packed: u32 = 0u;
  for (var k: u32 = 0u; k < 4u; k = k + 1u) {
    let i = w * 4u + k;
    var q: i32 = 0;
    if (i < ${N}u) {
      q = i32(clamp(round(x[i] / srq.x), -128.0, 127.0));
    }
    packed = packed | ((u32(q) & 0xFFu) << (k * 8u));
  }
  xq[w] = packed;
}
