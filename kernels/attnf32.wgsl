enable f16;
enable subgroups;
// Single-dispatch decode attention (A4B variant: f32 output, no SRQ): grid (Q_HEADS, DT dim-tiles). Each WG
// recomputes its head's scores (redundant ×DT — cheap vs a second fenced
// dispatch) then accumulates V for its HEAD_DIM/DT dims. Epilogue writes the
// o-proj input SRQ-quantized (packed int8). Softmax in f32, scaling = 1.0.
// Params: Q_HEADS, KV_HEADS, HEAD_DIM, MAXSEQ, WINDOW, DT, WG
@group(0) @binding(0) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> kcache: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> vcache: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> params: array<u32>;   // [1]=cacheLen
@group(0) @binding(4) var<storage, read_write> outv: array<vec4<f16>>;
@group(0) @binding(5) var<storage, read_write> score: array<f32>;   // SCORE=1: [QH][MAXSEQ]

var<workgroup> probs: array<f32, ${MAXSEQ}>;
var<workgroup> red: array<f32, ${WG}>;
var<workgroup> sgm: array<f32, 8>;
var<workgroup> sgs: array<f32, 8>;
var<workgroup> vpart: array<vec4<f32>, ${WG}>;   // V-phase t-partition partials

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x;
  let kvh = h / (${Q_HEADS}u / ${KV_HEADS}u);
  // BATCH=1 (prefill): wid.z = token; causal len = basePos(params[0]) + tok + 1
  let bTok = select(0u, wid.z, ${BATCH}u == 1u);
  var len = select(params[1], params[0] + bTok + 1u, ${BATCH}u == 1u);
  var start: u32 = 0u;
  if (${CACHEMODE}u == 0u && ${WINDOW}u != 0u && len > ${WINDOW}u) { start = len - ${WINDOW}u; }
  // ring (CACHEMODE 1): RING = window + chunk slack, so a prefill chunk's own
  // writes never clobber entries its earlier tokens still need. Each slot's
  // TRUE position is recovered from the newest written position (maxW) and
  // window/causality are enforced per slot below.
  var qp: u32 = 0u;      // this query's position
  var maxW: u32 = 0u;    // newest position written into the ring
  if (${CACHEMODE}u == 1u) {
    start = 0u;
    qp = select(params[1] - 1u, params[0] + bTok, ${BATCH}u == 1u);
    maxW = select(params[1] - 1u, params[0] + params[1] - 1u, ${BATCH}u == 1u);
    len = min(maxW + 1u, ${RING}u);
  }
  if (${CACHEMODE}u == 2u) { len = select(params[5], params[4] + bTok + 1u, ${BATCH}u == 1u); }
  let hd4 = ${HEAD_DIM}u / 4u;
  let qBase = bTok * ${Q_HEADS}u * hd4 + h * hd4;

  // scores (each thread strided over positions/slots)
  var m: f32 = -3.0e38;
  for (var t = start + lid.x; t < len; t = t + ${WG}u) {
    var dead = false;
    if (${CACHEMODE}u == 1u) {
      let ps = maxW - ((maxW + ${RING}u - t) % ${RING}u);   // slot t's true position
      dead = ps > qp || ps + ${WINDOW}u <= qp;              // causality + window
    }
    let kBase = (t * ${KV_HEADS}u + kvh) * hd4;
    var s: f32 = -3.0e38;
    if (!dead) {
      s = 0.0;
      for (var d: u32 = 0u; d < hd4; d = d + 1u) {
        s = s + dot(q[qBase + d], vec4<f32>(kcache[kBase + d]));
      }
    }
    probs[t] = s;
    m = max(m, s);
  }
  let m1 = subgroupMax(m);
  if ((lid.x & 31u) == 0u) { sgm[lid.x / 32u] = m1; }
  workgroupBarrier();
  var mx: f32 = -3.0e38;
  for (var i: u32 = 0u; i < ${WG}u / 32u; i = i + 1u) { mx = max(mx, sgm[i]); }
  var sm: f32 = 0.0;
  for (var t = start + lid.x; t < len; t = t + ${WG}u) {
    let e = exp(probs[t] - mx);
    probs[t] = e;
    sm = sm + e;
  }
  let s1 = subgroupAdd(sm);
  if ((lid.x & 31u) == 0u) { sgs[lid.x / 32u] = s1; }
  workgroupBarrier();
  var denom: f32 = 0.0;
  for (var i: u32 = 0u; i < ${WG}u / 32u; i = i + 1u) { denom = denom + sgs[i]; }

  // V for my dim tile: TP t-partitions per dim; f32 output (no SRQ in q4_0 land)
  let tw = hd4 / ${DT}u;
  let d0 = wid.y * tw;
  let tp = ${WG}u / tw;
  let dl = lid.x % tw;
  let part = lid.x / tw;
  var acc = vec4f(0.0);
  for (var t = start + part; t < len; t = t + tp) {
    acc = acc + probs[t] * vec4<f32>(vcache[(t * ${KV_HEADS}u + kvh) * hd4 + d0 + dl]);
  }
  vpart[lid.x] = acc;
  workgroupBarrier();
  if (lid.x < tw) {
    var v = vpart[lid.x];
    for (var p: u32 = 1u; p < tp; p = p + 1u) { v = v + vpart[lid.x + p * tw]; }
    outv[qBase + d0 + lid.x] = vec4<f16>(v / denom);
  }
  if (${SCORE}u == 1u) {                     // per-(head, slot) attention mass
    workgroupBarrier();                      // for the budget compactor
    for (var t = start + lid.x; t < len; t = t + ${WG}u) {
      score[h * ${MAXSEQ}u + t] = score[h * ${MAXSEQ}u + t] + probs[t] / denom;
    }
  } else { _ = score[0]; }
}
