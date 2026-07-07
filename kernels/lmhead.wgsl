// lm_head int2 matvec, tiled: weights repacked at load into [tile of 32 rows]
// blocks laid out [vec4word j][row t] so a 32-thread subgroup streams fully
// coalesced 512B lines. x (2560 f32) is staged in workgroup memory once per WG.
// Thread t owns row (wg*32+t) entirely — no reduction.
// y[o] = softcap( wscale[o] * (255*dot_unorm - 2*S) )   (lm_head SRQ identity)
// int2 decode via native unpack4x8unorm (v/255, refolded); -2 zero point uses
// S = Σ normed x (from the final rmsnorm's SUMOUT). Params: IN, OUT, SOFTCAP
enable subgroups;
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;  // tiled layout
@group(0) @binding(2) var<storage, read> wscale: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read> sums: array<f32>;   // [0] = Σ x

var<workgroup> xs: array<vec4<f32>, ${IN} / 4>;

@compute @workgroup_size(${TILE})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  if ((wid.y * 32768u + wid.x) * ${TILE}u >= ${OUT}u) { return; }   // excess-grid guard
  let nx = ${IN}u / 4u;
  for (var i = lid.x; i < nx; i = i + ${TILE}u) { xs[i] = x[i]; }
  workgroupBarrier();

  let tile = wid.y * 32768u + wid.x;
  let o = tile * ${TILE}u + lid.x;
  let vwords = ${IN}u / 64u;                 // vec4<u32> words per row (64 int2 each)
  let base = tile * (vwords * ${TILE}u);
  var acc: f32 = 0.0;
  for (var j: u32 = 0u; j < vwords; j = j + 1u) {
    let wv = w[base + j * ${TILE}u + lid.x]; // coalesced across the WG
    for (var h: u32 = 0u; h < 4u; h = h + 1u) {
      let ww = wv[h];
      let xb = j * 16u + h * 4u;             // 16 int2 per u32 = 4 vec4f of x
      let q0 = xs[xb]; let q1 = xs[xb + 1u]; let q2 = xs[xb + 2u]; let q3 = xs[xb + 3u];
      // native unorm unpacks give STRIDED values (v0,v4,v8,v12)/255 etc — pair
      // with x columns; the /255 and the -2 zero point are refolded in finish
      acc = acc + dot(unpack4x8unorm(ww & 0x03030303u), vec4f(q0.x, q1.x, q2.x, q3.x))
                + dot(unpack4x8unorm((ww >> 2u) & 0x03030303u), vec4f(q0.y, q1.y, q2.y, q3.y))
                + dot(unpack4x8unorm((ww >> 4u) & 0x03030303u), vec4f(q0.z, q1.z, q2.z, q3.z))
                + dot(unpack4x8unorm((ww >> 6u) & 0x03030303u), vec4f(q0.w, q1.w, q2.w, q3.w));
    }
  }
  if (o < ${OUT}u) {
    var out = (255.0 * acc - 2.0 * sums[0]) * wscale[o];
    if (${SOFTCAP} != 0.0) { out = ${SOFTCAP} * tanh(out / ${SOFTCAP}); }
    y[o] = out;
  }
}
