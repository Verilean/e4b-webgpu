// Decode attention v2 — absorbs head-prep: each WG (q-head h, dim-tile) does
// q norm+RoPE for its head and k/v norm+RoPE for its kv-head LOCALLY (registers
// /shared); designated WGs also write the caches for future tokens. The current
// position's k/v never round-trips memory, so the separate headprep dispatch
// and its fence disappear. Scores/softmax in f32; epilogue writes the o-proj
// input SRQ-quantized. Shared layers (SHARED=1): q-prep only, caches read-only.
// Params: Q_HEADS, KV_HEADS, HEAD_DIM, MAXSEQ, WINDOW, ROPE_ANGLES, THETA, EPS,
//         SHARED, DT, WG
enable subgroups;
@group(0) @binding(0) var<storage, read> qkv: array<f32>;      // [QH+2*KVH][HD] (or [QH][HD] shared)
@group(0) @binding(1) var<storage, read> qw: array<f32>;       // q_norm weight
@group(0) @binding(2) var<storage, read> kw: array<f32>;       // k_norm weight
@group(0) @binding(3) var<storage, read> params: array<u32>;   // [0]=pos [1]=len
@group(0) @binding(4) var<storage, read_write> kcache: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> vcache: array<vec4<f32>>;
@group(0) @binding(6) var<uniform> srq: vec2f;                 // o-proj (inS, outS)
@group(0) @binding(7) var<storage, read_write> xq: array<u32>;

var<workgroup> probs: array<f32, ${MAXSEQ}>;
var<workgroup> redM: array<f32, ${WG}>;
var<workgroup> redS: array<f32, ${WG}>;
var<workgroup> qs: array<vec4<f32>, ${HEAD_DIM} / 4>;   // prepped q (this head)
var<workgroup> ks: array<vec4<f32>, ${HEAD_DIM} / 4>;   // prepped k[pos] (kv head)
var<workgroup> vs: array<vec4<f32>, ${HEAD_DIM} / 4>;   // prepped v[pos]

fn rmsInv(base: u32, lid: u32) -> f32 {
  var s: f32 = 0.0;
  for (var d = lid; d < ${HEAD_DIM}u; d = d + ${WG}u) {
    let v = qkv[base + d];
    s = s + v * v;
  }
  redM[lid] = s;
  workgroupBarrier();
  var st = ${WG}u / 2u;
  while (st > 0u) {
    if (lid < st) { redM[lid] = redM[lid] + redM[lid + st]; }
    workgroupBarrier();
    st = st / 2u;
  }
  let r = redM[0];
  workgroupBarrier();
  return pow(r / f32(${HEAD_DIM}u) + ${EPS}, -0.5);
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

  // ---- local q prep (norm + rope) into qs ----
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
    qs[p / 4u][p % 4u] = y0;                 // scalar writes into vec4 shared
    qs[(p + half) / 4u][(p + half) % 4u] = y1;
  }

  // ---- local k/v prep for this WG's kv head (non-shared layers only) ----
  if (${SHARED}u == 0u) {
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
    let vBase = (${Q_HEADS}u + ${KV_HEADS}u + kvh) * ${HEAD_DIM}u;
    let vInv = rmsInv(vBase, lid);
    for (var d = lid; d < ${HEAD_DIM}u; d = d + ${WG}u) {
      vs[d / 4u][d % 4u] = qkv[vBase + d] * vInv;   // v_norm is scale-less
    }
    workgroupBarrier();
    // designated WG per kv head writes the caches for FUTURE tokens
    if (wid.y == 0u && h == kvh * (${Q_HEADS}u / ${KV_HEADS}u)) {
      let cBase = (pos * ${KV_HEADS}u + kvh) * hd4;
      for (var d = lid; d < hd4; d = d + ${WG}u) {
        kcache[cBase + d] = ks[d];
        vcache[cBase + d] = vs[d];
      }
    }
  }
  workgroupBarrier();

  // ---- scores: t == pos uses LOCAL ks; t < pos from cache ----
  var start: u32 = 0u;
  if (${WINDOW}u != 0u && len > ${WINDOW}u) { start = len - ${WINDOW}u; }
  var m: f32 = -3.0e38;
  var sm: f32 = 0.0;
  for (var t = start + lid; t < len; t = t + ${WG}u) {
    var s: f32 = 0.0;
    if (${SHARED}u == 0u && t == pos) {
      for (var d: u32 = 0u; d < hd4; d = d + 1u) { s = s + dot(qs[d], ks[d]); }
    } else {
      let kB = (t * ${KV_HEADS}u + kvh) * hd4;
      for (var d: u32 = 0u; d < hd4; d = d + 1u) { s = s + dot(qs[d], kcache[kB + d]); }
    }
    probs[t] = s;
    m = max(m, s);
  }
  // one fused (max, scaled-sum) reduction — online softmax
  for (var t = start + lid; t < len; t = t + ${WG}u) { sm = sm + exp(probs[t] - m); }
  redM[lid] = m;
  redS[lid] = sm;
  workgroupBarrier();
  var st = ${WG}u / 2u;
  while (st > 0u) {
    if (lid < st) {
      let m2 = max(redM[lid], redM[lid + st]);
      redS[lid] = redS[lid] * exp(redM[lid] - m2) + redS[lid + st] * exp(redM[lid + st] - m2);
      redM[lid] = m2;
    }
    workgroupBarrier();
    st = st / 2u;
  }
  let mx = redM[0];
  let denom = redS[0];
  workgroupBarrier();
  for (var t = start + lid; t < len; t = t + ${WG}u) { probs[t] = exp(probs[t] - mx); }
  workgroupBarrier();

  // ---- V for my dim tile (t == pos from LOCAL vs); quantized epilogue ----
  let tw = hd4 / ${DT}u;
  let d0 = wid.y * tw;
  for (var d = d0 + lid; d < d0 + tw; d = d + ${WG}u) {
    var acc = vec4f(0.0);
    for (var t = start; t < len; t = t + 1u) {
      if (${SHARED}u == 0u && t == pos) {
        acc = acc + probs[t] * vs[d];
      } else {
        acc = acc + probs[t] * vcache[(t * ${KV_HEADS}u + kvh) * hd4 + d];
      }
    }
    let v = acc / denom;
    let qv = vec4<i32>(clamp(round(v / srq.x), vec4f(-128.0), vec4f(127.0)));
    xq[h * hd4 + d] = (u32(qv.x) & 0xFFu) | ((u32(qv.y) & 0xFFu) << 8u)
                    | ((u32(qv.z) & 0xFFu) << 16u) | ((u32(qv.w) & 0xFFu) << 24u);
  }
}
