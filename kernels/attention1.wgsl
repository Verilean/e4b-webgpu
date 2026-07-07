enable subgroups;
// Single-dispatch decode attention: grid (Q_HEADS, DT dim-tiles). Each WG
// recomputes its head's scores (redundant ×DT — cheap vs a second fenced
// dispatch) then accumulates V for its HEAD_DIM/DT dims. Epilogue writes the
// o-proj input SRQ-quantized (packed int8). Softmax in f32, scaling = 1.0.
// Params: Q_HEADS, KV_HEADS, HEAD_DIM, MAXSEQ, WINDOW, DT, WG
@group(0) @binding(0) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> kcache: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> vcache: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> params: array<u32>;   // [1]=cacheLen
@group(0) @binding(4) var<uniform> srq: vec2f;                 // o-proj (inS, outS)
@group(0) @binding(5) var<storage, read_write> xq: array<u32>;
@group(0) @binding(6) var<storage, read_write> sumOut: array<atomic<i32>>;

var<workgroup> probs: array<f32, ${MAXSEQ}>;
var<workgroup> red: array<f32, ${WG}>;
var<workgroup> sgm: array<f32, 8>;
var<workgroup> sgs: array<f32, 8>;
var<workgroup> vpart: array<vec4<f32>, ${WG}>;   // V-phase t-partition partials

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x;
  let kvh = h / (${Q_HEADS}u / ${KV_HEADS}u);
  let len = params[1];
  var start: u32 = 0u;
  if (${WINDOW}u != 0u && len > ${WINDOW}u) { start = len - ${WINDOW}u; }
  let hd4 = ${HEAD_DIM}u / 4u;
  let qBase = h * hd4;

  // scores (each thread strided over positions)
  var m: f32 = -3.0e38;
  for (var t = start + lid.x; t < len; t = t + ${WG}u) {
    let kBase = (t * ${KV_HEADS}u + kvh) * hd4;
    var s: f32 = 0.0;
    for (var d: u32 = 0u; d < hd4; d = d + 1u) {
      s = s + dot(q[qBase + d], kcache[kBase + d]);
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

  // V for my dim tile; quantized o-proj input epilogue
  let tw = hd4 / ${DT}u;                       // vec4-dims per tile
  let d0 = wid.y * tw;
  for (var d = d0 + lid.x; d < d0 + tw; d = d + ${WG}u) {
    var acc = vec4f(0.0);
    for (var t = start; t < len; t = t + 1u) {
      acc = acc + probs[t] * vcache[(t * ${KV_HEADS}u + kvh) * hd4 + d];
    }
    let v = acc / denom;
    let qv = vec4<i32>(clamp(round(v / srq.x), vec4f(-127.0), vec4f(127.0)));
    xq[qBase + d] = (u32(qv.x) & 0xFFu) | ((u32(qv.y) & 0xFFu) << 8u)
                  | ((u32(qv.z) & 0xFFu) << 16u) | ((u32(qv.w) & 0xFFu) << 24u);
    atomicAdd(&sumOut[0], qv.x + qv.y + qv.z + qv.w);
  }
}
