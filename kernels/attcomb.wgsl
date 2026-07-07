// Flash-decode phase B: one WG per q-head combines chunk partials.
// Params: Q_HEADS, HEAD_DIM, WINDOW, CS, CHUNKS, WG
@group(0) @binding(0) var<storage, read> partO: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> partME: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@group(0) @binding(3) var<storage, read_write> outv: array<vec4<f32>>;

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x;
  let len = params[1];
  var start: u32 = 0u;
  if (${WINDOW}u != 0u && len > ${WINDOW}u) { start = len - ${WINDOW}u; }
  let n = min(${CHUNKS}u, (len - start + ${CS}u - 1u) / ${CS}u);
  let hd4 = ${HEAD_DIM}u / 4u;
  var M: f32 = -3.0e38;
  for (var c: u32 = 0u; c < n; c = c + 1u) { M = max(M, partME[h * ${CHUNKS}u + c].x); }
  var denom: f32 = 0.0;
  for (var c: u32 = 0u; c < n; c = c + 1u) {
    let me = partME[h * ${CHUNKS}u + c];
    denom = denom + me.y * exp(me.x - M);
  }
  for (var d = lid.x; d < hd4; d = d + ${WG}u) {
    var acc = vec4f(0.0);
    for (var c: u32 = 0u; c < n; c = c + 1u) {
      let me = partME[h * ${CHUNKS}u + c];
      if (me.y > 0.0) { acc = acc + partO[(h * ${CHUNKS}u + c) * hd4 + d] * exp(me.x - M); }
    }
    outv[h * hd4 + d] = acc / denom;
  }
}
