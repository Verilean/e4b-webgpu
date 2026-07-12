// Q6_K kernels over TILE-TRANSPOSED planes (128-row tiles; unit u of row o at
// (tile*unitsPerRow + u)*128 + o%128 — fully coalesced across a 128-thread WG).
// Decode: q_i = qlnib_i + 16*qh2_i; both via native unpack4x8unorm (/255 refold);
// −32 zero point deferred through per-16-elem x group sums.
// MODE 0: dequant ONE row (token embedding) × MULT; row = params[2].
// MODE 1: matvec (lm_head): thread-per-row (TILE=128), x staged in shared, softcap.
// Params: N (row len), OUT (rows), MODE, TILE(=128), MULT, SOFTCAP, TOKSRC
// (TOKSRC=1: MODE-0 row comes from tok[0] (the argmax buffer) instead of params[2])
enable subgroups;
@group(0) @binding(0) var<storage, read> ql: array<u32>;
@group(0) @binding(1) var<storage, read> qh: array<u32>;
@group(0) @binding(2) var<storage, read> sc: array<u32>;       // 4 i8 per word
@group(0) @binding(3) var<storage, read> dd: array<f32>;
@group(0) @binding(4) var<storage, read> params: array<u32>;   // [2]=token (MODE 0)
@group(0) @binding(7) var<storage, read> tok: array<u32>;      // [0]=token (TOKSRC=1)
@group(0) @binding(5) var<storage, read> x: array<vec4<f32>>;  // MODE 1 input
@group(0) @binding(6) var<storage, read_write> y: array<f32>;

var<workgroup> xs: array<vec4<f32>, ${N} / 4>;
var<workgroup> gs: array<f32, ${N} / 16>;    // per-16-elem x group sums

fn i8of(word: u32, k: u32) -> f32 {
  return f32((i32(word << ((3u - k) * 8u))) >> 24u);
}

@compute @workgroup_size(${TILE})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
  _ = x[0];
  _ = params[0];
  _ = tok[0];
  let lid = lid3.x;
  let bpr = ${N}u / 256u;
  let qlU = bpr * 32u;
  let qhU = bpr * 16u;
  let scU = bpr * 4u;

  if (${MODE}u == 0u) {
    let row = select(params[2], tok[0], ${TOKSRC}u == 1u);
    let tile = row / ${TILE}u;
    let t = row % ${TILE}u;
    for (var b = lid; b < bpr; b = b + ${TILE}u) {
      let d = dd[(tile * bpr + b) * ${TILE}u + t];
      for (var h: u32 = 0u; h < 2u; h = h + 1u) {
        let scW0 = sc[(tile * scU + b * 4u + h * 2u) * ${TILE}u + t];
        let scW1 = sc[(tile * scU + b * 4u + h * 2u + 1u) * ${TILE}u + t];
        for (var w: u32 = 0u; w < 8u; w = w + 1u) {
          let qlA = ql[(tile * qlU + b * 32u + h * 16u + w) * ${TILE}u + t];
          let qlB = ql[(tile * qlU + b * 32u + h * 16u + 8u + w) * ${TILE}u + t];
          let qhw = qh[(tile * qhU + b * 16u + h * 8u + w) * ${TILE}u + t];
          let is = w / 4u;
          let s0 = i8of(select(scW0, scW1, (is + 0u) >= 4u), (is + 0u) % 4u);
          let s2 = i8of(select(scW0, scW1, (is + 2u) >= 4u), (is + 2u) % 4u);
          let s4 = i8of(select(scW0, scW1, (is + 4u) >= 4u), (is + 4u) % 4u);
          let s6 = i8of(select(scW0, scW1, (is + 6u) >= 4u), (is + 6u) % 4u);
          let base = b * 256u + h * 128u + w * 4u;
          for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let q1 = f32(i32(((qlA >> (k * 8u)) & 0xFu) | (((qhw >> (k * 8u)) & 3u) << 4u))) - 32.0;
            let q2 = f32(i32(((qlB >> (k * 8u)) & 0xFu) | (((qhw >> (k * 8u + 2u)) & 3u) << 4u))) - 32.0;
            let q3 = f32(i32(((qlA >> (k * 8u + 4u)) & 0xFu) | (((qhw >> (k * 8u + 4u)) & 3u) << 4u))) - 32.0;
            let q4 = f32(i32(((qlB >> (k * 8u + 4u)) & 0xFu) | (((qhw >> (k * 8u + 6u)) & 3u) << 4u))) - 32.0;
            y[base + k]        = d * s0 * q1 * ${MULT};
            y[base + k + 32u]  = d * s2 * q2 * ${MULT};
            y[base + k + 64u]  = d * s4 * q3 * ${MULT};
            y[base + k + 96u]  = d * s6 * q4 * ${MULT};
          }
        }
      }
    }
    return;
  }

  // MODE 1: matvec — coalesced tiled reads, thread-per-row
  let tile = wid.y * 32768u + wid.x;
  if (tile * ${TILE}u >= ${OUT}u) { return; }
  for (var i = lid; i < ${N}u / 4u; i = i + ${TILE}u) { xs[i] = x[i]; }
  workgroupBarrier();
  for (var g = lid; g < ${N}u / 16u; g = g + ${TILE}u) {
    let q0 = xs[g * 4u]; let q1 = xs[g * 4u + 1u]; let q2 = xs[g * 4u + 2u]; let q3 = xs[g * 4u + 3u];
    gs[g] = dot(q0, vec4f(1.0)) + dot(q1, vec4f(1.0)) + dot(q2, vec4f(1.0)) + dot(q3, vec4f(1.0));
  }
  workgroupBarrier();
  let o = tile * ${TILE}u + lid;
  if (o >= ${OUT}u) { return; }
  let t = lid;
  var acc: f32 = 0.0;
  for (var b: u32 = 0u; b < bpr; b = b + 1u) {
    let d = dd[(tile * bpr + b) * ${TILE}u + t];
    var bacc: f32 = 0.0;
    var gsum: f32 = 0.0;
    for (var h: u32 = 0u; h < 2u; h = h + 1u) {
      let scW0 = sc[(tile * scU + b * 4u + h * 2u) * ${TILE}u + t];
      let scW1 = sc[(tile * scU + b * 4u + h * 2u + 1u) * ${TILE}u + t];
      let e0 = b * 64u + h * 32u;
      for (var w: u32 = 0u; w < 8u; w = w + 1u) {
        let qlA = ql[(tile * qlU + b * 32u + h * 16u + w) * ${TILE}u + t];
        let qlB = ql[(tile * qlU + b * 32u + h * 16u + 8u + w) * ${TILE}u + t];
        let qhw = qh[(tile * qhU + b * 16u + h * 8u + w) * ${TILE}u + t];
        let is = w / 4u;
        let s0 = i8of(select(scW0, scW1, (is + 0u) >= 4u), (is + 0u) % 4u);
        let s2 = i8of(select(scW0, scW1, (is + 2u) >= 4u), (is + 2u) % 4u);
        let s4 = i8of(select(scW0, scW1, (is + 4u) >= 4u), (is + 4u) % 4u);
        let s6 = i8of(select(scW0, scW1, (is + 6u) >= 4u), (is + 6u) % 4u);
        let x0 = xs[e0 + w];
        let x1 = xs[e0 + 8u + w];
        let x2 = xs[e0 + 16u + w];
        let x3 = xs[e0 + 24u + w];
        bacc = bacc
          + s0 * (dot(unpack4x8unorm(qlA & 0x0F0F0F0Fu), x0)
                + 16.0 * dot(unpack4x8unorm(qhw & 0x03030303u), x0))
          + s2 * (dot(unpack4x8unorm(qlB & 0x0F0F0F0Fu), x1)
                + 16.0 * dot(unpack4x8unorm((qhw >> 2u) & 0x03030303u), x1))
          + s4 * (dot(unpack4x8unorm((qlA >> 4u) & 0x0F0F0F0Fu), x2)
                + 16.0 * dot(unpack4x8unorm((qhw >> 4u) & 0x03030303u), x2))
          + s6 * (dot(unpack4x8unorm((qlB >> 4u) & 0x0F0F0F0Fu), x3)
                + 16.0 * dot(unpack4x8unorm((qhw >> 6u) & 0x03030303u), x3));
      }
      let g0 = b * 16u + h * 8u;
      for (var j: u32 = 0u; j < 8u; j = j + 1u) {
        let sj = i8of(select(scW0, scW1, j >= 4u), j % 4u);
        gsum = gsum + sj * gs[g0 + j];
      }
    }
    acc = acc + d * (255.0 * bacc - 32.0 * gsum);
  }
  var out = acc;
  if (${SOFTCAP} != 0.0) { out = ${SOFTCAP} * tanh(out / ${SOFTCAP}); }
  y[o] = out;
}
