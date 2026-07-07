// Fused post-norm + residual add: hidden = (hidden + rms(x)*weight) * MUL
// (MUL = layer_scalar literal, or 1.0). One WG. Params: DIM, EPS, MUL, WG
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> hidden: array<f32>;

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
  for (var i = lid.x; i < ${DIM}u; i = i + ${WG}u) {
    hidden[i] = (hidden[i] + x[i] * inv * weight[i]) * ${MUL};
  }
}
