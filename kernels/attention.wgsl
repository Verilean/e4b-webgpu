// Naive decode attention (batch 1, one new token), f32, scaling = 1.0 (Gemma4).
// One workgroup per q-head. Cache layout: [MAXSEQ, KV_HEADS, HEAD_DIM].
// GQA: kv head = q_head / (Q_HEADS / KV_HEADS).
// WINDOW = 0 → full causal; else attend positions [max(0, len-WINDOW), len).
// params[1] = cacheLen (positions INCLUDING current).
// Params: Q_HEADS, KV_HEADS, HEAD_DIM, MAXSEQ, WINDOW, WG
@group(0) @binding(0) var<storage, read> q: array<f32>;        // [Q_HEADS*HEAD_DIM]
@group(0) @binding(1) var<storage, read> kcache: array<f32>;
@group(0) @binding(2) var<storage, read> vcache: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@group(0) @binding(4) var<storage, read_write> outv: array<f32>; // [Q_HEADS*HEAD_DIM]

var<workgroup> scores: array<f32, ${MAXSEQ}>;
var<workgroup> redbuf: array<f32, ${WG}>;

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x;
  let kvh = h / (${Q_HEADS}u / ${KV_HEADS}u);
  let len = params[1];
  var start: u32 = 0u;
  if (${WINDOW}u != 0u && len > ${WINDOW}u) { start = len - ${WINDOW}u; }
  let qBase = h * ${HEAD_DIM}u;

  // scores[t] = dot(q, k_t)   (scaling = 1.0 in Gemma4)
  for (var t = start + lid.x; t < len; t = t + ${WG}u) {
    let kBase = (t * ${KV_HEADS}u + kvh) * ${HEAD_DIM}u;
    var s: f32 = 0.0;
    for (var d: u32 = 0u; d < ${HEAD_DIM}u; d = d + 1u) {
      s = s + q[qBase + d] * kcache[kBase + d];
    }
    scores[t] = s;
  }
  workgroupBarrier();

  // max (f32 softmax)
  var m: f32 = -3.0e38;
  for (var t = start + lid.x; t < len; t = t + ${WG}u) { m = max(m, scores[t]); }
  redbuf[lid.x] = m;
  workgroupBarrier();
  var stride = ${WG}u / 2u;
  while (stride > 0u) {
    if (lid.x < stride) { redbuf[lid.x] = max(redbuf[lid.x], redbuf[lid.x + stride]); }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let mx = redbuf[0];
  workgroupBarrier();

  // exp + sum
  var sm: f32 = 0.0;
  for (var t = start + lid.x; t < len; t = t + ${WG}u) {
    let e = exp(scores[t] - mx);
    scores[t] = e;
    sm = sm + e;
  }
  redbuf[lid.x] = sm;
  workgroupBarrier();
  stride = ${WG}u / 2u;
  while (stride > 0u) {
    if (lid.x < stride) { redbuf[lid.x] = redbuf[lid.x] + redbuf[lid.x + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let denom = redbuf[0];

  // out[d] = sum_t p_t * v_t[d]
  for (var d = lid.x; d < ${HEAD_DIM}u; d = d + ${WG}u) {
    var acc: f32 = 0.0;
    for (var t = start; t < len; t = t + 1u) {
      let vBase = (t * ${KV_HEADS}u + kvh) * ${HEAD_DIM}u;
      acc = acc + scores[t] * vcache[vBase + d];
    }
    outv[qBase + d] = acc / denom;
  }
}
