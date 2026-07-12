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
  // MODE 1: matvec — stage x, thread-per-row tile
  if ((wid.y * 32768u + wid.x) * ${TILE}u >= ${OUT}u) { return; }
  for (var i = lid.x; i < ${N}u / 4u; i = i + ${TILE}u) { xs[i] = x[i]; }
  workgroupBarrier();
  let o = (wid.y * 32768u + wid.x) * ${TILE}u + lid.x;
  if (o >= ${OUT}u) { return; }
  var acc: f32 = 0.0;
  for (var b: u32 = 0u; b < blocksPerRow; b = b + 1u) {
    acc = acc + doBlock(o, b, blocksPerRow, 1.0);
  }
  var out = acc;
  if (${SOFTCAP} != 0.0) { out = ${SOFTCAP} * tanh(out / ${SOFTCAP}); }
  y[o] = out;
}
