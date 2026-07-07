// Fused K/V prep: WGs [0,KV_HEADS) do K (weighted RMS + RoPE + kcache write at
// pos), WGs [KV_HEADS, 2*KV_HEADS) do V (scale-less RMS + vcache write).
// Cache layout [MAXSEQ, KV_HEADS, HEAD_DIM]. Params: KV_HEADS, HEAD_DIM,
// ROPE_ANGLES, THETA, EPS, WG
@group(0) @binding(0) var<storage, read> kIn: array<f32>;
@group(0) @binding(1) var<storage, read> vIn: array<f32>;
@group(0) @binding(2) var<storage, read> kw: array<f32>;       // k_norm weight
@group(0) @binding(3) var<storage, read> params: array<u32>;   // [0]=pos
@group(0) @binding(4) var<storage, read_write> kcache: array<f32>;
@group(0) @binding(5) var<storage, read_write> vcache: array<f32>;

var<workgroup> red: array<f32, ${WG}>;

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let isV = wid.x >= ${KV_HEADS}u;
  let h = wid.x % ${KV_HEADS}u;
  let base = h * ${HEAD_DIM}u;
  var s: f32 = 0.0;
  for (var d = lid.x; d < ${HEAD_DIM}u; d = d + ${WG}u) {
    var v: f32;
    if (isV) { v = vIn[base + d]; } else { v = kIn[base + d]; }
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
  let pos = params[0];
  let cBase = (pos * ${KV_HEADS}u + h) * ${HEAD_DIM}u;
  let half = ${HEAD_DIM}u / 2u;
  if (isV) {
    for (var d = lid.x; d < ${HEAD_DIM}u; d = d + ${WG}u) {
      vcache[cBase + d] = vIn[base + d] * inv;               // v_norm is scale-less
    }
  } else {
    let fp = f32(pos);
    for (var p = lid.x; p < half; p = p + ${WG}u) {
      let x0 = kIn[base + p] * inv * kw[p];
      let x1 = kIn[base + p + half] * inv * kw[p + half];
      if (p < ${ROPE_ANGLES}u) {
        let th = fp * pow(${THETA}, -2.0 * f32(p) / f32(${HEAD_DIM}u));
        let c = cos(th); let sn = sin(th);
        kcache[cBase + p] = x0 * c - x1 * sn;
        kcache[cBase + p + half] = x0 * sn + x1 * c;
      } else {
        kcache[cBase + p] = x0;
        kcache[cBase + p + half] = x1;
      }
    }
  }
}
