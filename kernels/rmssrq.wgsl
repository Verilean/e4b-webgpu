// Fused RMSNorm + SRQ int8 pre-quantization for NS downstream linears with
// distinct input scales. One WG. Writes NS regions of DIM/4 packed words.
// xq[s][i] = clamp(round( x[i]*inv*weight[i] / scales[s] ), -128, 127)
// Params: DIM, NS, EPS, WG
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;   // [NS]
@group(0) @binding(3) var<storage, read_write> xq: array<u32>;

var<workgroup> partial: array<f32, ${WG}>;

@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  var s: f32 = 0.0;
  for (var i = lid.x; i < ${DIM}u; i = i + ${WG}u) {
    let v = x[i];
    s = s + v * v;
  }
  partial[lid.x] = s;
  workgroupBarrier();
  var stride = ${WG}u / 2u;
  while (stride > 0u) {
    if (lid.x < stride) { partial[lid.x] = partial[lid.x] + partial[lid.x + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let inv = pow(partial[0] / f32(${DIM}u) + ${EPS}, -0.5);
  let words = ${DIM}u / 4u;
  for (var wi = lid.x; wi < ${NS}u * words; wi = wi + ${WG}u) {
    let sc = scales[wi / words];
    let j = (wi % words) * 4u;
    var packed: u32 = 0u;
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
      let q = i32(clamp(round(x[j + k] * inv * weight[j + k] / sc), -128.0, 127.0));
      packed = packed | ((u32(q) & 0xFFu) << (k * 8u));
    }
    xq[wi] = packed;
  }
}
