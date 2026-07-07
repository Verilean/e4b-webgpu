// Fused per-head RMSNorm + RoPE for q (in place). One WG per head.
// Params: HEADS, HEAD_DIM, ROPE_ANGLES, THETA, EPS, WG
@group(0) @binding(0) var<storage, read> weight: array<f32>;   // [HEAD_DIM]
@group(0) @binding(1) var<storage, read> params: array<u32>;   // [0]=pos
@group(0) @binding(2) var<storage, read_write> x: array<f32>;

var<workgroup> red: array<f32, ${WG}>;

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let base = wid.x * ${HEAD_DIM}u;
  var s: f32 = 0.0;
  for (var d = lid.x; d < ${HEAD_DIM}u; d = d + ${WG}u) {
    let v = x[base + d];
    s = s + v * v;
  }
  red[lid.x] = s;
  workgroupBarrier();
  var stride = ${WG}u / 2u;
  while (stride > 0u) {
    if (lid.x < stride) { red[lid.x] = red[lid.x] + red[lid.x + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let inv = pow(red[0] / f32(${HEAD_DIM}u) + ${EPS}, -0.5);
  let half = ${HEAD_DIM}u / 2u;
  let pos = f32(params[0]);
  for (var p = lid.x; p < half; p = p + ${WG}u) {
    let i0 = base + p;
    let i1 = i0 + half;
    let x0 = x[i0] * inv * weight[p];
    let x1 = x[i1] * inv * weight[p + half];
    if (p < ${ROPE_ANGLES}u) {
      let th = pos * pow(${THETA}, -2.0 * f32(p) / f32(${HEAD_DIM}u));
      let c = cos(th); let sn = sin(th);
      x[i0] = x0 * c - x1 * sn;
      x[i1] = x0 * sn + x1 * c;
    } else {
      x[i0] = x0;
      x[i1] = x1;
    }
  }
}
