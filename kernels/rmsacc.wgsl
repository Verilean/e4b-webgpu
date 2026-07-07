enable subgroups;
// Fused post-norm + residual add: hidden = (hidden + rms(x)*weight) * MUL
// (MUL = layer_scalar literal, or 1.0). One WG. Params: DIM, EPS, MUL, WG
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> hidden: array<f32>;

var<workgroup> sg8: array<f32, 8>;

@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  var s: f32 = 0.0;
  for (var i = lid.x; i < ${DIM}u; i = i + ${WG}u) {
    let v = x[i];
    s = s + v * v;
  }
  let s1 = subgroupAdd(s);
  if ((lid.x & 31u) == 0u) { sg8[lid.x / 32u] = s1; }
  workgroupBarrier();
  var tot: f32 = 0.0;
  for (var i: u32 = 0u; i < ${WG}u / 32u; i = i + 1u) { tot = tot + sg8[i]; }
  let inv = pow(tot / f32(${DIM}u) + ${EPS}, -0.5);
  for (var i = lid.x; i < ${DIM}u; i = i + ${WG}u) {
    hidden[i] = (hidden[i] + x[i] * inv * weight[i]) * ${MUL};
  }
}
