// Q6_K kernels over repacked planes: ql (128B/block of 256), qh (64B), sc (16
// int8), d (f16, 2/u32). Decode (llama.cpp dequant_row_q6_K): per 128-half,
// for l in 0..31: elems l, l+32, l+64, l+96 from ql[l]/ql[l+32] nibbles + qh[l]
// 2-bit fields, q−32, × d × sc[sub16].
// MODE 0: dequant ONE row (token embedding) × MULT. Row index = params[2].
// MODE 1: tiled matvec (lm_head): TILE rows/WG, thread-per-row, x staged in
//         workgroup memory, softcap epilogue.
// Params: N (row len), OUT, MODE, TILE, MULT, SOFTCAP
enable subgroups;
@group(0) @binding(0) var<storage, read> ql: array<vec4<u32>>;   // 8 vec4 per half
@group(0) @binding(1) var<storage, read> qh: array<vec4<u32>>;   // 4 vec4 per half
@group(0) @binding(2) var<storage, read> sc: array<u32>;         // 4 i8 per word
@group(0) @binding(3) var<storage, read> dd: array<u32>;         // f16 ×2 per word
@group(0) @binding(4) var<storage, read> params: array<u32>;     // [2]=token (MODE 0)
@group(0) @binding(5) var<storage, read> x: array<vec4<f32>>;    // MODE 1 input
@group(0) @binding(6) var<storage, read_write> y: array<f32>;

var<workgroup> xs: array<vec4<f32>, ${N} / 4>;
var<workgroup> gs: array<f32, ${N} / 16>;   // per-16-elem group sums (−32 refold)

fn i8of(word: u32, k: u32) -> f32 {
  return f32((i32(word << ((3u - k) * 8u))) >> 24u);
}
fn dOf(b: u32) -> f32 {
  let two = unpack2x16float(dd[b / 2u]);
  return select(two.x, two.y, (b & 1u) == 1u);
}

// accumulate dot(row-block b of row `row`, xs[...]) or write dequant (MODE 0)
fn doBlock(row: u32, b: u32, blocksPerRow: u32, mult: f32) -> f32 {
  let blk = row * blocksPerRow + b;
  let d = dOf(blk);
  var acc: f32 = 0.0;
  for (var h: u32 = 0u; h < 2u; h = h + 1u) {          // two 128-halves
    let qlB = blk * 8u + h * 4u;                       // vec4<u32> index (16B each)
    let qhB = blk * 4u + h * 2u;
    let scB = blk * 4u + h * 2u;                       // sc words (4 i8 each)
    for (var l: u32 = 0u; l < 32u; l = l + 1u) {
      let qlLo = (ql[qlB + (l / 16u)][(l / 4u) % 4u] >> ((l % 4u) * 8u)) & 0xFFu;
      let qlHi = (ql[qlB + 2u + (l / 16u)][(l / 4u) % 4u] >> ((l % 4u) * 8u)) & 0xFFu;
      let qhB8 = (qh[qhB + (l / 16u)][(l / 4u) % 4u] >> ((l % 4u) * 8u)) & 0xFFu;
      let is = l / 16u;
      let base = b * 256u + h * 128u;
      let s0 = i8of(sc[scB + ((is + 0u) / 4u)], (is + 0u) % 4u);
      let s2 = i8of(sc[scB + ((is + 2u) / 4u)], (is + 2u) % 4u);
      let s4 = i8of(sc[scB + ((is + 4u) / 4u)], (is + 4u) % 4u);
      let s6 = i8of(sc[scB + ((is + 6u) / 4u)], (is + 6u) % 4u);
      let q1 = f32(i32((qlLo & 0xFu) | ((qhB8 & 3u) << 4u))) - 32.0;
      let q2 = f32(i32((qlHi & 0xFu) | (((qhB8 >> 2u) & 3u) << 4u))) - 32.0;
      let q3 = f32(i32((qlLo >> 4u) | (((qhB8 >> 4u) & 3u) << 4u))) - 32.0;
      let q4 = f32(i32((qlHi >> 4u) | (((qhB8 >> 6u) & 3u) << 4u))) - 32.0;
      if (${MODE}u == 0u) {
        y[base + l]        = d * s0 * q1 * mult;
        y[base + l + 32u]           = d * s2 * q2 * mult;
        y[base + l + 64u]           = d * s4 * q3 * mult;
        y[base + l + 96u]           = d * s6 * q4 * mult;
      } else {
        let xb = (base) / 4u;
        acc = acc + d * (s0 * q1 * xs[xb + (l / 4u)][l % 4u]
                       + s2 * q2 * xs[xb + 8u + (l / 4u)][l % 4u]
                       + s4 * q3 * xs[xb + 16u + (l / 4u)][l % 4u]
                       + s6 * q4 * xs[xb + 24u + (l / 4u)][l % 4u]);
      }
    }
  }
  return acc;
}

@compute @workgroup_size(${TILE})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  _ = x[0];        // keep bindings across MODE variants (layout:"auto" DCE)
  _ = params[0];
  let blocksPerRow = ${N}u / 256u;
  if (${MODE}u == 0u) {
    // one row dequant: WG covers blocks; TILE threads, blocksPerRow blocks
    let row = params[2];
    for (var b = lid.x; b < blocksPerRow; b = b + ${TILE}u) {
      _ = doBlock(row, b, blocksPerRow, ${MULT});
    }
    return;
  }
  // MODE 1: matvec — stage x + group sums, thread-per-row tile, native unpacks.
  // Per 128-half: q_i = qlnib_i + 16*qh2_i; Σ(q−32)x over 16-elem scale groups:
  //   y = d·[ 255·Σ_w sc(g)·(dot(unorm(ql),x4) + 16·dot(unorm(qh2),x4)) − 32·Σ_j sc_j·S_j ]
  if ((wid.y * 32768u + wid.x) * ${TILE}u >= ${OUT}u) { return; }
  for (var i = lid.x; i < ${N}u / 4u; i = i + ${TILE}u) { xs[i] = x[i]; }
  workgroupBarrier();
  for (var g = lid.x; g < ${N}u / 16u; g = g + ${TILE}u) {
    let q0 = xs[g * 4u]; let q1 = xs[g * 4u + 1u]; let q2 = xs[g * 4u + 2u]; let q3 = xs[g * 4u + 3u];
    gs[g] = dot(q0, vec4f(1.0)) + dot(q1, vec4f(1.0)) + dot(q2, vec4f(1.0)) + dot(q3, vec4f(1.0));
  }
  workgroupBarrier();
  let o = (wid.y * 32768u + wid.x) * ${TILE}u + lid.x;
  if (o >= ${OUT}u) { return; }
  var acc: f32 = 0.0;
  for (var b: u32 = 0u; b < blocksPerRow; b = b + 1u) {
    let blk = o * blocksPerRow + b;
    let d = dOf(blk);
    var bacc: f32 = 0.0;                      // 255-scaled dots
    var gsum: f32 = 0.0;                      // Σ_j sc_j S_j
    for (var h: u32 = 0u; h < 2u; h = h + 1u) {
      let qlB = blk * 8u + h * 4u;            // vec4<u32> (16B) units
      let qhB = blk * 4u + h * 2u;
      let scB = blk * 4u + h * 2u;            // u32 (4×i8) units
      let e0 = b * 64u + h * 32u;             // first x-vec4 of this half (256 elems = 64 v4)
      for (var w: u32 = 0u; w < 8u; w = w + 1u) {
        let qlA = ql[qlB + (w / 4u)][w % 4u];         // bytes l=4w..4w+3
        let qlBv = ql[qlB + 2u + (w / 4u)][w % 4u];   // bytes l+32
        let qhw = qh[qhB + (w / 4u)][w % 4u];
        let is = w / 4u;                               // sc group 0 or 1
        let s0 = i8of(sc[scB + ((is + 0u) / 4u)], (is + 0u) % 4u);
        let s2 = i8of(sc[scB + ((is + 2u) / 4u)], (is + 2u) % 4u);
        let s4 = i8of(sc[scB + ((is + 4u) / 4u)], (is + 4u) % 4u);
        let s6 = i8of(sc[scB + ((is + 6u) / 4u)], (is + 6u) % 4u);
        let x0 = xs[e0 + w];          // elems l..l+3
        let x1 = xs[e0 + 8u + w];     // elems l+32..
        let x2 = xs[e0 + 16u + w];    // elems l+64..
        let x3 = xs[e0 + 24u + w];    // elems l+96..
        bacc = bacc
          + s0 * (dot(unpack4x8unorm(qlA & 0x0F0F0F0Fu), x0)
                + 16.0 * dot(unpack4x8unorm(qhw & 0x03030303u), x0))
          + s2 * (dot(unpack4x8unorm(qlBv & 0x0F0F0F0Fu), x1)
                + 16.0 * dot(unpack4x8unorm((qhw >> 2u) & 0x03030303u), x1))
          + s4 * (dot(unpack4x8unorm((qlA >> 4u) & 0x0F0F0F0Fu), x2)
                + 16.0 * dot(unpack4x8unorm((qhw >> 4u) & 0x03030303u), x2))
          + s6 * (dot(unpack4x8unorm((qlBv >> 4u) & 0x0F0F0F0Fu), x3)
                + 16.0 * dot(unpack4x8unorm((qhw >> 6u) & 0x03030303u), x3));
      }
      // −32 refold: Σ_j sc_j S_j over this half's 8 groups (elems h*128 + j*16)
      let g0 = b * 16u + h * 8u;
      for (var j: u32 = 0u; j < 8u; j = j + 1u) {
        let sj = i8of(sc[scB + (j / 4u)], j % 4u);
        gsum = gsum + sj * gs[g0 + j];
      }
    }
    acc = acc + d * (255.0 * bacc - 32.0 * gsum);
  }
  var out = acc;
  if (${SOFTCAP} != 0.0) { out = ${SOFTCAP} * tanh(out / ${SOFTCAP}); }
  y[o] = out;
}
