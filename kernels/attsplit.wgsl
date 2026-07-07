// Flash-decode phase A: grid (Q_HEADS, CHUNKS). Each WG handles CS positions of
// one q-head: scores (1 pos/thread), local softmax stats, partial V-accumulation.
// partO is stored UNNORMALIZED with local max m_c; phase B combines.
// Params: Q_HEADS, KV_HEADS, HEAD_DIM, MAXSEQ, WINDOW, CS(=WG), CHUNKS
enable subgroups;
@group(0) @binding(0) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> kcache: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> vcache: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> params: array<u32>;    // [1]=cacheLen
@group(0) @binding(4) var<storage, read_write> partO: array<vec4<f32>>; // [H][CHUNKS][HD/4]
@group(0) @binding(5) var<storage, read_write> partME: array<vec2<f32>>; // [H][CHUNKS] (m, e)

var<workgroup> probs: array<f32, ${CS}>;
var<workgroup> red: array<f32, ${CS}>;

@compute @workgroup_size(${CS})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x;
  let c = wid.y;
  let kvh = h / (${Q_HEADS}u / ${KV_HEADS}u);
  let len = params[1];
  var start: u32 = 0u;
  if (${WINDOW}u != 0u && len > ${WINDOW}u) { start = len - ${WINDOW}u; }
  let lo = start + c * ${CS}u;
  let hd4 = ${HEAD_DIM}u / 4u;
  let pi = h * ${CHUNKS}u + c;
  if (lo >= len) {                       // empty chunk
    if (lid.x == 0u) { partME[pi] = vec2f(-3.0e38, 0.0); }
    return;
  }
  let qBase = h * hd4;

  // score for my position
  let t = lo + lid.x;
  var s: f32 = -3.0e38;
  if (t < len) {
    let kBase = (t * ${KV_HEADS}u + kvh) * hd4;
    s = 0.0;
    for (var d: u32 = 0u; d < hd4; d = d + 1u) {
      s = s + dot(q[qBase + d], kcache[kBase + d]);
    }
  }
  // chunk max
  red[lid.x] = s;
  workgroupBarrier();
  var stride = ${CS}u / 2u;
  while (stride > 0u) {
    if (lid.x < stride) { red[lid.x] = max(red[lid.x], red[lid.x + stride]); }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let m = red[0];
  workgroupBarrier();
  // probs + sum
  var p: f32 = 0.0;
  if (t < len) { p = exp(s - m); }
  probs[lid.x] = p;
  red[lid.x] = p;
  workgroupBarrier();
  stride = ${CS}u / 2u;
  while (stride > 0u) {
    if (lid.x < stride) { red[lid.x] = red[lid.x] + red[lid.x + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (lid.x == 0u) { partME[pi] = vec2f(m, red[0]); }
  // partial V: out[d4] = sum_t p_t v_t[d4]
  let hi = min(lo + ${CS}u, len);
  for (var d = lid.x; d < hd4; d = d + ${CS}u) {
    var acc = vec4f(0.0);
    for (var tt = lo; tt < hi; tt = tt + 1u) {
      acc = acc + probs[tt - lo] * vcache[(tt * ${KV_HEADS}u + kvh) * hd4 + d];
    }
    partO[pi * hd4 + d] = acc;
  }
}
