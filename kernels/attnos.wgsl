enable f16;
enable subgroups;
// Online-softmax attention for FULL layers — no probs[] workgroup array, so
// no context cap (M3-v2). WG=256 = 8 subgroups × 32 lanes; each lane owns
// HD/128 vec4 strips; each subgroup streams its share of positions with a
// running (max, sum, acc) and rescale; log-sum-exp merge across subgroups.
// SCORE=1 (decode probe/maintenance): a second pass re-derives normalized
// probs and accumulates per-slot mass into score[h*SCAP + t].
// CACHEMODE 2 semantics: len = params[5] (decode) / params[4]+bTok+1 (batch).
// Params: Q_HEADS, KV_HEADS, HEAD_DIM, SCAP, BATCH, SCORE, WG(256)
@group(0) @binding(0) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> kcache: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> vcache: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@group(0) @binding(4) var<storage, read_write> outv: array<vec4<f16>>;
@group(0) @binding(5) var<storage, read_write> score: array<f32>;

var<workgroup> sgm: array<f32, 8>;
var<workgroup> sgs: array<f32, 8>;
var<workgroup> stage: array<vec4<f32>, 1024>;   // 8 sg × 128 vec4 strips

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x;
  let kvh = h / (${Q_HEADS}u / ${KV_HEADS}u);
  let bTok = select(0u, wid.z, ${BATCH}u == 1u);
  let len = select(params[5], params[4] + bTok + 1u, ${BATCH}u == 1u);
  let hd4 = ${HEAD_DIM}u / 4u;
  let ns = hd4 / 32u;                        // vec4 strips per lane
  let sg = lid.x / 32u;
  let lane = lid.x % 32u;
  let qBase = bTok * ${Q_HEADS}u * hd4 + h * hd4;

  var qv: array<vec4<f32>, 4>;
  for (var j: u32 = 0u; j < ns; j = j + 1u) { qv[j] = q[qBase + lane * ns + j]; }

  var m: f32 = -3.0e38;
  var s: f32 = 0.0;
  var acc: array<vec4<f32>, 4>;
  for (var j: u32 = 0u; j < ns; j = j + 1u) { acc[j] = vec4<f32>(0.0); }

  let iters = (len + 7u) / 8u;               // workgroup-uniform trip count
  for (var it: u32 = 0u; it < iters; it = it + 1u) {
    let t = it * 8u + sg;
    let ok = t < len;                        // uniform within the subgroup
    let tt = min(t, len - 1u);
    let kBase = (tt * ${KV_HEADS}u + kvh) * hd4;
    var partial: f32 = 0.0;
    for (var j: u32 = 0u; j < ns; j = j + 1u) {
      partial = partial + dot(qv[j], vec4<f32>(kcache[kBase + lane * ns + j]));
    }
    let sc = subgroupAdd(partial);           // unguarded: Tint-uniform
    if (ok) {
      let mN = max(m, sc);
      let scale = exp(m - mN);
      let e = exp(sc - mN);
      for (var j: u32 = 0u; j < ns; j = j + 1u) {
        acc[j] = acc[j] * scale + e * vec4<f32>(vcache[kBase + lane * ns + j]);
      }
      s = s * scale + e;
      m = mN;
    }
  }
  if (lane == 0u) { sgm[sg] = m; sgs[sg] = s; }
  workgroupBarrier();
  var gm: f32 = -3.0e38;
  for (var i: u32 = 0u; i < 8u; i = i + 1u) { gm = max(gm, sgm[i]); }
  let f = exp(m - gm);
  for (var j: u32 = 0u; j < ns; j = j + 1u) {
    stage[sg * 128u + lane * ns + j] = acc[j] * f;
  }
  workgroupBarrier();
  var denom: f32 = 0.0;
  for (var i: u32 = 0u; i < 8u; i = i + 1u) { denom = denom + sgs[i] * exp(sgm[i] - gm); }
  denom = max(denom, 1e-20);
  // out: 128 vec4 strips, threads 0..127 each sum 8 subgroup stages
  if (lid.x < hd4) {
    var v = vec4<f32>(0.0);
    for (var i: u32 = 0u; i < 8u; i = i + 1u) { v = v + stage[i * 128u + lid.x]; }
    outv[qBase + lid.x] = vec4<f16>(v / denom);
  }
  if (${SCORE}u == 1u) {                     // probe/maintenance mass pass
    let iters2 = (len + 7u) / 8u;
    for (var it: u32 = 0u; it < iters2; it = it + 1u) {
      let t = it * 8u + sg;
      let tt = min(t, len - 1u);
      let kBase = (tt * ${KV_HEADS}u + kvh) * hd4;
      var partial: f32 = 0.0;
      for (var j: u32 = 0u; j < ns; j = j + 1u) {
        partial = partial + dot(qv[j], vec4<f32>(kcache[kBase + lane * ns + j]));
      }
      let sc = subgroupAdd(partial);
      if (lane == 0u && t < len) {
        score[h * ${SCAP}u + t] = score[h * ${SCAP}u + t] + exp(sc - gm) / denom;
      }
    }
  } else { _ = score[0]; }
}
