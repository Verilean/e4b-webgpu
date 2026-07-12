enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;
diagnostic(off, chromium.subgroup_matrix_uniformity);
// Q4_0 batched-prefill GEMM on subgroup matrices (k13-style geometry, read as
// reference — implementation is ours): TILE = 32 M(tokens) × 64 N(rows) × 32 K,
// WG = 128 = 4 subgroups, each owning a 16M×32N subtile = 2×4 result mats
// accumulated in f32. Weight tiles are JIT-dequanted from the q4_0 nibble
// plane with the per-32-block scale folded (TPREC=f32: dequant exact — the
// only difference vs q40mv is the summation order). One q4_0 block = one
// K-tile (TILE_K = 32). Activations are f16 rows ([M][IN]); output f32.
// OUT must be a 64-multiple, IN a 32-multiple (all A4B sites qualify).
// Params: IN, OUT, TPREC (f32|f16), MTILES (dispatch y = ceil(M/32))
@group(0) @binding(0) var<storage, read> xh: array<vec4<f16>>;   // [M][IN/4]
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;     // nibble plane
@group(0) @binding(2) var<storage, read> ws: array<u32>;          // f16 scales, 2/word
@group(0) @binding(3) var<storage, read> mprm: array<u32>;        // [1]=M
@group(0) @binding(4) var<storage, read_write> y: array<f32>;     // [M][OUT]

var<workgroup> tA: array<${TPREC}, 32 * 32>;   // [m][k], stride 32
var<workgroup> tB: array<${TPREC}, 64 * 32>;   // [n][k], stride 32
var<workgroup> scr: array<array<f32, 64>, 4>;

fn scaleOf(base: u32, b: u32) -> f32 {
  let i = base + b;
  let two = unpack2x16float(ws[i / 2u]);
  return select(two.x, two.y, (i & 1u) == 1u);
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32,
        @builtin(subgroup_invocation_id) sgid: u32, @builtin(subgroup_size) sgsz: u32) {
  let mBase = wid.y * 32u;
  let nBase = wid.x * 64u;
  let mTot = mprm[1];
  let rowB = ${IN}u / 32u;                 // q4_0 blocks per row = K tiles
  let sub = li / sgsz;                     // subgroup 0..3
  let subx = sub / 2u;                     // N half (0/1): 32 cols
  let suby = sub % 2u;                     // M half (0/1): 16 rows

  var c00: subgroup_matrix_result<f32, 8, 8>;
  var c01: subgroup_matrix_result<f32, 8, 8>;
  var c02: subgroup_matrix_result<f32, 8, 8>;
  var c03: subgroup_matrix_result<f32, 8, 8>;
  var c10: subgroup_matrix_result<f32, 8, 8>;
  var c11: subgroup_matrix_result<f32, 8, 8>;
  var c12: subgroup_matrix_result<f32, 8, 8>;
  var c13: subgroup_matrix_result<f32, 8, 8>;

  // A-tile fill indices (thread-invariant across kb): row = li/4, 8 elems each
  let aRow = li / 4u;
  let aCol = (li % 4u) * 8u;
  // B-tile fill: n = li/2, half = li%2 (lo nibbles = elems 0..15, hi = 16..31)
  let bN = li / 2u;
  let bHalf = li % 2u;
  let bRowBase = (nBase + bN) * rowB;

  for (var kb: u32 = 0u; kb < rowB; kb = kb + 1u) {
    // ---- stage A (activations, f16 → TPREC; rows ≥ M zero-filled) ----
    let am = mBase + aRow;
    let ax = (am * ${IN}u + kb * 32u + aCol) / 4u;
    if (am < mTot) {
      let v0 = xh[ax];
      let v1 = xh[ax + 1u];
      tA[aRow * 32u + aCol]      = ${TPREC}(v0.x); tA[aRow * 32u + aCol + 1u] = ${TPREC}(v0.y);
      tA[aRow * 32u + aCol + 2u] = ${TPREC}(v0.z); tA[aRow * 32u + aCol + 3u] = ${TPREC}(v0.w);
      tA[aRow * 32u + aCol + 4u] = ${TPREC}(v1.x); tA[aRow * 32u + aCol + 5u] = ${TPREC}(v1.y);
      tA[aRow * 32u + aCol + 6u] = ${TPREC}(v1.z); tA[aRow * 32u + aCol + 7u] = ${TPREC}(v1.w);
    } else {
      for (var j: u32 = 0u; j < 8u; j = j + 1u) { tA[aRow * 32u + aCol + j] = ${TPREC}(0.0); }
    }
    // ---- stage B (JIT dequant, scale folded) ----
    let wv = w[bRowBase + kb];
    let d = scaleOf(bRowBase, kb);
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
      let word = select(select(select(wv.x, wv.y, i >= 4u), wv.z, i >= 8u), wv.w, i >= 12u);
      let nib = (word >> ((i % 4u) * 8u + bHalf * 4u)) & 0xFu;
      tB[bN * 32u + bHalf * 16u + i] = ${TPREC}(d * (f32(nib) - 8.0));
    }
    workgroupBarrier();

    for (var st: u32 = 0u; st < 32u; st = st + 8u) {
      let aOff = suby * 16u * 32u + st;
      let a0 = subgroupMatrixLoad<subgroup_matrix_left<${TPREC}, 8, 8>>(&tA, aOff, false, 32u);
      let a1 = subgroupMatrixLoad<subgroup_matrix_left<${TPREC}, 8, 8>>(&tA, aOff + 8u * 32u, false, 32u);
      let bOff = subx * 32u * 32u + st;
      let b0 = subgroupMatrixLoad<subgroup_matrix_right<${TPREC}, 8, 8>>(&tB, bOff, true, 32u);
      let b1 = subgroupMatrixLoad<subgroup_matrix_right<${TPREC}, 8, 8>>(&tB, bOff + 8u * 32u, true, 32u);
      let b2 = subgroupMatrixLoad<subgroup_matrix_right<${TPREC}, 8, 8>>(&tB, bOff + 16u * 32u, true, 32u);
      let b3 = subgroupMatrixLoad<subgroup_matrix_right<${TPREC}, 8, 8>>(&tB, bOff + 24u * 32u, true, 32u);
      c00 = subgroupMatrixMultiplyAccumulate(a0, b0, c00);
      c01 = subgroupMatrixMultiplyAccumulate(a0, b1, c01);
      c02 = subgroupMatrixMultiplyAccumulate(a0, b2, c02);
      c03 = subgroupMatrixMultiplyAccumulate(a0, b3, c03);
      c10 = subgroupMatrixMultiplyAccumulate(a1, b0, c10);
      c11 = subgroupMatrixMultiplyAccumulate(a1, b1, c11);
      c12 = subgroupMatrixMultiplyAccumulate(a1, b2, c12);
      c13 = subgroupMatrixMultiplyAccumulate(a1, b3, c13);
    }
    workgroupBarrier();
  }

  // epilogue: stage each 8x8 through this subgroup's scratch, 2 elems/lane
  let row = sgid / 4u;
  let col = (sgid % 4u) * 2u;
  let mSub = mBase + suby * 16u;
  let nSub = nBase + subx * 32u;
  for (var s: u32 = 0u; s < 8u; s = s + 1u) {
    let strip = s / 4u;              // M strip (0/1)
    let bcol = s % 4u;               // N strip (0..3)
    var cm: subgroup_matrix_result<f32, 8, 8>;
    switch (s) {
      case 0u: { cm = c00; } case 1u: { cm = c01; } case 2u: { cm = c02; } case 3u: { cm = c03; }
      case 4u: { cm = c10; } case 5u: { cm = c11; } case 6u: { cm = c12; } default: { cm = c13; }
    }
    subgroupMatrixStore(&scr[sub], 0u, cm, false, 8u);
    let gm = mSub + strip * 8u + row;
    if (gm < mTot) {
      let gn = nSub + bcol * 8u + col;
      y[gm * ${OUT}u + gn] = scr[sub][row * 8u + col];
      y[gm * ${OUT}u + gn + 1u] = scr[sub][row * 8u + col + 1u];
    }
  }
}
