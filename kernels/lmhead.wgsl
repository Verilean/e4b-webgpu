// lm_head int2 matvec, tiled: weights repacked at load into [tile of 32 rows]
// blocks laid out [vec4word j][row t] so a 32-thread subgroup streams fully
// coalesced 512B lines. x (2560 f32) is staged in workgroup memory once per WG.
// Thread t owns row (wg*32+t) entirely — no reduction.
// y[o] = softcap( wscale[o] * dot(x, w_row) )   (lm_head SRQ is identity)
// Params: IN, OUT, SOFTCAP
enable subgroups;
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;  // tiled layout
@group(0) @binding(2) var<storage, read> wscale: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> xs: array<vec4<f32>, ${IN} / 4>;

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  if ((wid.y * 32768u + wid.x) * 32u >= ${OUT}u) { return; }   // excess-grid guard
  let nx = ${IN}u / 4u;
  for (var i = lid.x; i < nx; i = i + 32u) { xs[i] = x[i]; }
  workgroupBarrier();

  let tile = wid.y * 32768u + wid.x;
  let o = tile * 32u + lid.x;
  let vwords = ${IN}u / 64u;                 // vec4<u32> words per row (64 int2 each)
  let base = tile * (vwords * 32u);
  var acc: f32 = 0.0;
  for (var j: u32 = 0u; j < vwords; j = j + 1u) {
    let wv = w[base + j * 32u + lid.x];      // coalesced across the subgroup
    for (var h: u32 = 0u; h < 4u; h = h + 1u) {
      let ww = wv[h];
      let xb = j * 16u + h * 4u;             // 16 int2 per u32 = 4 vec4f of x
      for (var g: u32 = 0u; g < 4u; g = g + 1u) {
        let b = (ww >> (g * 8u)) & 0xFFu;
        let w4 = vec4f(f32(b & 3u), f32((b >> 2u) & 3u), f32((b >> 4u) & 3u), f32((b >> 6u) & 3u)) - vec4f(2.0);
        acc = acc + dot(w4, xs[xb + g]);
      }
    }
  }
  if (o < ${OUT}u) {
    var out = acc * wscale[o];
    if (${SOFTCAP} != 0.0) { out = ${SOFTCAP} * tanh(out / ${SOFTCAP}); }
    y[o] = out;
  }
}
