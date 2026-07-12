enable f16;
enable subgroups;
// Fused head prep, one dispatch: WGs [0,QH) q-heads (weighted RMS + RoPE in
// place on the concat qkv buffer), [QH, QH+KVH) k-heads (weighted RMS + RoPE +
// kcache write), [QH+KVH, QH+2*KVH) v-heads (scale-less RMS + vcache write).
// Shared layers dispatch only QH workgroups. Cache layout [MAXSEQ, KVH, HD].
// Params: QH, KVH, HEAD_DIM, ROPE_ANGLES, THETA, EPS, KEQV(0/1), WG
@group(0) @binding(0) var<storage, read> qkv: array<f32>;       // [QH+(1|2)*KVH][HD]
@group(0) @binding(7) var<storage, read_write> qOut: array<f32>; // prepped q
@group(0) @binding(1) var<storage, read> qw: array<f32>;
@group(0) @binding(2) var<storage, read> kw: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;     // [0]=pos
@group(0) @binding(4) var<storage, read_write> kcache: array<f16>;
@group(0) @binding(5) var<storage, read_write> vcache: array<f16>;
@group(0) @binding(6) var<storage, read_write> sumI: array<i32>;

var<workgroup> sg8: array<f32, 8>;

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  if (wid.x == 0u && lid.x == 0u) { sumI[0] = 0; }   // zero the Σq slot for att1
  var srcHead = wid.x;
  if (${KEQV}u == 1u && wid.x >= ${QH}u + ${KVH}u) { srcHead = wid.x - ${KVH}u; }  // v reads the k slice
  // BATCH=1 (prefill): wid.y = token; qkv/qOut rows per token, pos advances
  let bTok = select(0u, wid.y, ${BATCH}u == 1u);
  let qkvS = (${QH}u + ${KVH}u * (2u - ${KEQV}u)) * ${HEAD_DIM}u;
  let base = bTok * qkvS + srcHead * ${HEAD_DIM}u;
  let qb = bTok * ${QH}u * ${HEAD_DIM}u + wid.x * ${HEAD_DIM}u;
  var s: f32 = 0.0;
  for (var d = lid.x; d < ${HEAD_DIM}u; d = d + ${WG}u) {
    let v = qkv[base + d];
    s = s + v * v;
  }
  let s1 = subgroupAdd(s);
  if ((lid.x & 31u) == 0u) { sg8[lid.x / 32u] = s1; }
  workgroupBarrier();
  var tot: f32 = 0.0;
  for (var i: u32 = 0u; i < ${WG}u / 32u; i = i + 1u) { tot = tot + sg8[i]; }
  let inv = pow(tot / f32(${HEAD_DIM}u) + ${EPS}, -0.5);
  let half = ${HEAD_DIM}u / 2u;
  let pos = params[0] + bTok;
  let fp = f32(pos);

  if (wid.x < ${QH}u) {                                    // q: in place
    for (var p = lid.x; p < half; p = p + ${WG}u) {
      let x0 = qkv[base + p] * inv * qw[p];
      let x1 = qkv[base + p + half] * inv * qw[p + half];
      if (p < ${ROPE_ANGLES}u) {
        let th = fp * pow(${THETA}, -2.0 * f32(p) / f32(${HEAD_DIM}u));
        let c = cos(th); let sn = sin(th);
        qOut[qb + p] = x0 * c - x1 * sn;
        qOut[qb + p + half] = x0 * sn + x1 * c;
      } else {
        qOut[qb + p] = x0;
        qOut[qb + p + half] = x1;
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
        kcache[cBase + p] = f16(x0 * c - x1 * sn);
        kcache[cBase + p + half] = f16(x0 * sn + x1 * c);
      } else {
        kcache[cBase + p] = f16(x0);
        kcache[cBase + p + half] = f16(x1);
      }
    }
  } else {                                                 // v: scale-less → vcache
    let h = wid.x - ${QH}u - ${KVH}u;
    let cBase = (pos * ${KVH}u + h) * ${HEAD_DIM}u;
    for (var d = lid.x; d < ${HEAD_DIM}u; d = d + ${WG}u) {
      vcache[cBase + d] = f16(qkv[base + d] * inv);
    }
  }
}
