// Fused geglu + SRQ: xq = quant( gelu_pytorch_tanh(gate)*up , srq.x )
// One thread per packed word (4 elems). Params: N (mult of 4), WG
@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<uniform> srq: vec2f;
@group(0) @binding(3) var<storage, read_write> xq: array<u32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = gid.x;
  if (w >= ${N}u / 4u) { return; }
  var packed: u32 = 0u;
  for (var k: u32 = 0u; k < 4u; k = k + 1u) {
    let i = w * 4u + k;
    let x = gate[i];
    let v = 0.5 * x * (1.0 + tanh(clamp(0.7978845608028654 * (x + 0.044715 * x*x*x), -20.0, 20.0))) * up[i];
    let q = i32(clamp(round(v / srq.x), -128.0, 127.0));
    packed = packed | ((u32(q) & 0xFFu) << (k * 8u));
  }
  xq[w] = packed;
}
