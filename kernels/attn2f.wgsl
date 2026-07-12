// A4B decode attention with head-prep ABSORBED (viable at DT=1: no tile
// redundancy; k/v prep duplicated only QH/KVH× per kv head — cheap):
// per WG (q-head h): q norm+rope → qs; k norm+rope → ks; v = v_norm(k slice if
// KEQV else v slice) → vs; designated WG writes the caches at pos; scores use
// LOCAL ks/vs for t==pos and the caches for t<pos. f32 output.
// Params: Q_HEADS, KV_HEADS, HEAD_DIM, MAXSEQ, WINDOW, ROPE_ANGLES, THETA,
//         EPS, KEQV, WG
enable subgroups;
@group(0) @binding(0) var<storage, read> qkv: array<f32>;      // [QH+(1|2)*KVH][HD]
@group(0) @binding(1) var<storage, read> qw: array<f32>;       // q_norm
@group(0) @binding(2) var<storage, read> kw: array<f32>;       // k_norm
@group(0) @binding(3) var<storage, read> params: array<u32>;   // [0]=pos [1]=len
@group(0) @binding(4) var<storage, read_write> kcache: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> vcache: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> outv: array<vec4<f32>>;

var<workgroup> probs: array<f32, ${MAXSEQ}>;
var<workgroup> sg8: array<f32, 8>;
var<workgroup> sgm: array<f32, 8>;
var<workgroup> qs: array<vec4<f32>, ${HEAD_DIM} / 4>;
var<workgroup> ks: array<vec4<f32>, ${HEAD_DIM} / 4>;
var<workgroup> vs: array<vec4<f32>, ${HEAD_DIM} / 4>;
var<workgroup> vpart: array<vec4<f32>, ${WG}>;

fn rmsInv(base: u32, lid: u32) -> f32 {
  var s: f32 = 0.0;
  for (var d = lid; d < ${HEAD_DIM}u; d = d + ${WG}u) {
    let v = qkv[base + d];
    s = s + v * v;
  }
  let s1 = subgroupAdd(s);
  if ((lid & 31u) == 0u) { sg8[lid / 32u] = s1; }
  workgroupBarrier();
  var tot: f32 = 0.0;
  for (var i: u32 = 0u; i < ${WG}u / 32u; i = i + 1u) { tot = tot + sg8[i]; }
  workgroupBarrier();
  return pow(tot / f32(${HEAD_DIM}u) + ${EPS}, -0.5);
}

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
  let h = wid.x;
  let lid = lid3.x;
  let kvh = h / (${Q_HEADS}u / ${KV_HEADS}u);
  let pos = params[0];
  let len = params[1];
  let fp = f32(pos);
  let hd4 = ${HEAD_DIM}u / 4u;
  let half = ${HEAD_DIM}u / 2u;

  // q prep
  let qBase = h * ${HEAD_DIM}u;
  let qInv = rmsInv(qBase, lid);
  for (var p = lid; p < half; p = p + ${WG}u) {
    let x0 = qkv[qBase + p] * qInv * qw[p];
    let x1 = qkv[qBase + p + half] * qInv * qw[p + half];
    var y0 = x0; var y1 = x1;
    if (p < ${ROPE_ANGLES}u) {
      let th = fp * pow(${THETA}, -2.0 * f32(p) / f32(${HEAD_DIM}u));
      let c = cos(th); let sn = sin(th);
      y0 = x0 * c - x1 * sn;
      y1 = x0 * sn + x1 * c;
    }
    qs[p / 4u][p % 4u] = y0;
    qs[(p + half) / 4u][(p + half) % 4u] = y1;
  }
  // k prep (norm + rope)
  let kBase = (${Q_HEADS}u + kvh) * ${HEAD_DIM}u;
  let kInv = rmsInv(kBase, lid);
  for (var p = lid; p < half; p = p + ${WG}u) {
    let x0 = qkv[kBase + p] * kInv * kw[p];
    let x1 = qkv[kBase + p + half] * kInv * kw[p + half];
    var y0 = x0; var y1 = x1;
    if (p < ${ROPE_ANGLES}u) {
      let th = fp * pow(${THETA}, -2.0 * f32(p) / f32(${HEAD_DIM}u));
      let c = cos(th); let sn = sin(th);
      y0 = x0 * c - x1 * sn;
      y1 = x0 * sn + x1 * c;
    }
    ks[p / 4u][p % 4u] = y0;
    ks[(p + half) / 4u][(p + half) % 4u] = y1;
  }
  // v prep (scale-less norm, no rope); KEQV: source = the k slice
  let vBase = select((${Q_HEADS}u + ${KV_HEADS}u + kvh) * ${HEAD_DIM}u, kBase, ${KEQV}u == 1u);
  let vInv = rmsInv(vBase, lid);
  for (var d = lid; d < ${HEAD_DIM}u; d = d + ${WG}u) {
    vs[d / 4u][d % 4u] = qkv[vBase + d] * vInv;
  }
  workgroupBarrier();
  // one WG per kv head writes the caches at pos (for FUTURE tokens)
  if (h == kvh * (${Q_HEADS}u / ${KV_HEADS}u)) {
    let cBase = (pos * ${KV_HEADS}u + kvh) * hd4;
    for (var d = lid; d < hd4; d = d + ${WG}u) {
      kcache[cBase + d] = ks[d];
      vcache[cBase + d] = vs[d];
    }
  }

  // scores (t == pos from LOCAL ks)
  var start: u32 = 0u;
  if (${WINDOW}u != 0u && len > ${WINDOW}u) { start = len - ${WINDOW}u; }
  var m: f32 = -3.0e38;
  for (var t = start + lid; t < len; t = t + ${WG}u) {
    var s: f32 = 0.0;
    if (t == pos) {
      for (var d: u32 = 0u; d < hd4; d = d + 1u) { s = s + dot(qs[d], ks[d]); }
    } else {
      let kB = (t * ${KV_HEADS}u + kvh) * hd4;
      for (var d: u32 = 0u; d < hd4; d = d + 1u) { s = s + dot(qs[d], kcache[kB + d]); }
    }
    probs[t] = s;
    m = max(m, s);
  }
  let m1 = subgroupMax(m);
  if ((lid & 31u) == 0u) { sgm[lid / 32u] = m1; }
  workgroupBarrier();
  var mx: f32 = -3.0e38;
  for (var i: u32 = 0u; i < ${WG}u / 32u; i = i + 1u) { mx = max(mx, sgm[i]); }
  var sm: f32 = 0.0;
  for (var t = start + lid; t < len; t = t + ${WG}u) {
    let e = exp(probs[t] - mx);
    probs[t] = e;
    sm = sm + e;
  }
  let s1 = subgroupAdd(sm);
  if ((lid & 31u) == 0u) { sg8[lid / 32u] = s1; }
  workgroupBarrier();
  var denom: f32 = 0.0;
  for (var i: u32 = 0u; i < ${WG}u / 32u; i = i + 1u) { denom = denom + sg8[i]; }

  // V (t == pos from LOCAL vs), t-partitioned
  let tw = hd4;
  let tp = ${WG}u / tw;
  let dl = lid % tw;
  let part = lid / tw;
  var acc = vec4f(0.0);
  for (var t = start + part; t < len; t = t + tp) {
    if (t == pos) { acc = acc + probs[t] * vs[dl]; }
    else { acc = acc + probs[t] * vcache[(t * ${KV_HEADS}u + kvh) * hd4 + dl]; }
  }
  vpart[lid] = acc;
  workgroupBarrier();
  if (lid < tw) {
    var v = vpart[lid];
    for (var p: u32 = 1u; p < tp; p = p + 1u) { v = v + vpart[lid + p * tw]; }
    outv[h * hd4 + lid] = v / denom;
  }
}
