// Per-head RMSNorm over HEAD_DIM, IN-PLACE (q_norm/k_norm; v_norm via WITH_SCALE=0).
// One workgroup per head; reads complete before writes (barrier). Params: HEAD_DIM, WITH_SCALE, EPS, WG
@group(0) @binding(0) var<storage, read> weight: array<f32>;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
var<workgroup> partial: array<f32, ${WG}>;
@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let base = wid.x * ${HEAD_DIM}u;
  var s: f32 = 0.0;
  for (var i = lid.x; i < ${HEAD_DIM}u; i = i + ${WG}u) { let v = x[base+i]; s = s + v*v; }
  partial[lid.x] = s;
  workgroupBarrier();
  var stride = ${WG}u / 2u;
  while (stride > 0u) {
    if (lid.x < stride) { partial[lid.x] = partial[lid.x] + partial[lid.x+stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let inv = pow(partial[0] / f32(${HEAD_DIM}u) + ${EPS}, -0.5);
  for (var i = lid.x; i < ${HEAD_DIM}u; i = i + ${WG}u) {
    var v = x[base+i] * inv;
    if (${WITH_SCALE}u == 1u) { v = v * weight[i]; }
    x[base+i] = v;
  }
}
