// Fused head prep, one dispatch: WGs [0,QH) q-heads (weighted RMS + RoPE in
// place on the concat qkv buffer), [QH, QH+KVH) k-heads (weighted RMS + RoPE +
// kcache write), [QH+KVH, QH+2*KVH) v-heads (scale-less RMS + vcache write).
// Shared layers dispatch only QH workgroups. Cache layout [MAXSEQ, KVH, HD].
// Params: QH, KVH, HEAD_DIM, ROPE_ANGLES, THETA, EPS, WG
@group(0) @binding(0) var<storage, read_write> qkv: array<f32>;  // [QH+2*KVH][HD]
@group(0) @binding(1) var<storage, read> qw: array<f32>;
@group(0) @binding(2) var<storage, read> kw: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;     // [0]=pos
@group(0) @binding(4) var<storage, read_write> kcache: array<f32>;
@group(0) @binding(5) var<storage, read_write> vcache: array<f32>;

var<workgroup> red: array<f32, ${WG}>;

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let base = wid.x * ${HEAD_DIM}u;
  var s: f32 = 0.0;
  for (var d = lid.x; d < ${HEAD_DIM}u; d = d + ${WG}u) {
    let v = qkv[base + d];
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
  let pos = params[0];
  let fp = f32(pos);

  if (wid.x < ${QH}u) {                                    // q: in place
    for (var p = lid.x; p < half; p = p + ${WG}u) {
      let x0 = qkv[base + p] * inv * qw[p];
      let x1 = qkv[base + p + half] * inv * qw[p + half];
      if (p < ${ROPE_ANGLES}u) {
        let th = fp * pow(${THETA}, -2.0 * f32(p) / f32(${HEAD_DIM}u));
        let c = cos(th); let sn = sin(th);
        qkv[base + p] = x0 * c - x1 * sn;
        qkv[base + p + half] = x0 * sn + x1 * c;
      } else {
        qkv[base + p] = x0;
        qkv[base + p + half] = x1;
      }
    }
  } else if (wid.x < ${QH}u + ${KVH}u) {                   // k: rope → kcache
    let h = wid.x - ${QH}u;
    let cBase = (pos * ${KVH}u + h) * ${HEAD_DIM}u;
    for (var p = lid.x; p < half; p = p + ${WG}u) {
      let x0 = qkv[base + p] * inv * kw[p];
      let x1 = qkv[base + p + half] * inv * kw[p + half];
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
  } else {                                                 // v: scale-less → vcache
    let h = wid.x - ${QH}u - ${KVH}u;
    let cBase = (pos * ${KVH}u + h) * ${HEAD_DIM}u;
    for (var d = lid.x; d < ${HEAD_DIM}u; d = d + ${WG}u) {
      vcache[cBase + d] = qkv[base + d] * inv;
    }
  }
}
