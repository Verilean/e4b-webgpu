enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;
diagnostic(off, chromium.subgroup_matrix_uniformity);
// GROUPED MoE GEMM, MC=32 chunks (large-M prefill): q40sg tile geometry
// (32M×64N×32K, WG=128 = 4 subgroups × 8 result mats) with the M rows gathered
// through expgroup MC=32 descriptors and the weight base from the chunk's
// expert. ENTROW=0: x row = ent/K (gate|up over moeIn, OUT = 2·FF);
// ENTROW=1: x row = ent (down over gegluSlots). Unweighted per-entry output to
// y[ent][OUT]. Each expert's weights are read ~once per layer at M·K/E ≥ 32.
// Params: IN, OUT, K, ENTROW, TPREC
@group(0) @binding(0) var<storage, read> xh: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;     // nibble plane
@group(0) @binding(2) var<storage, read> ws: array<u32>;          // f16 scales, 2/word
@group(0) @binding(3) var<storage, read> chunkExp: array<u32>;
@group(0) @binding(4) var<storage, read> chunkEnt: array<u32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;     // [M*K][OUT]

var<workgroup> tA: array<${TPREC}, 32 * 32>;   // [m][k], stride 32
var<workgroup> tB: array<${TPREC}, 64 * 32>;   // [n][k], stride 32
var<workgroup> scr: array<array<f32, 64>, 4>;
var<workgroup> wgExp: u32;
var<workgroup> ents: array<u32, 32>;

fn scaleOf(base: u32, b: u32) -> f32 {
  let i = base + b;
  let two = unpack2x16float(ws[i / 2u]);
  return select(two.x, two.y, (i & 1u) == 1u);
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32,
        @builtin(subgroup_invocation_id) sgid: u32, @builtin(subgroup_size) sgsz: u32) {
  if (li == 0u) { wgExp = chunkExp[wid.z]; }
  if (li < 32u) { ents[li] = chunkEnt[wid.z * 32u + li]; }
  let ce = workgroupUniformLoad(&wgExp);
  if (ce == 0xFFFFFFFFu) { return; }        // sentinel chunk (uniform exit)

  let nBase = wid.x * 64u;
  let rowB = ${IN}u / 32u;
  let eb = ce * (${OUT}u * rowB);            // expert weight base (blocks)
  let sub = li / sgsz;
  let subx = sub / 2u;                       // N half
  let suby = sub % 2u;                       // M half

  var c00: subgroup_matrix_result<f32, 8, 8>;
  var c01: subgroup_matrix_result<f32, 8, 8>;
  var c02: subgroup_matrix_result<f32, 8, 8>;
  var c03: subgroup_matrix_result<f32, 8, 8>;
  var c10: subgroup_matrix_result<f32, 8, 8>;
  var c11: subgroup_matrix_result<f32, 8, 8>;
  var c12: subgroup_matrix_result<f32, 8, 8>;
  var c13: subgroup_matrix_result<f32, 8, 8>;

  let aRow = li / 4u;
  let aCol = (li % 4u) * 8u;
  let aEnt = ents[aRow];
  var aX: u32 = 0u;
  if (aEnt != 0xFFFFFFFFu) {
    let r = select(aEnt / ${K}u, aEnt, ${ENTROW}u == 1u);
    aX = (r * ${IN}u + aCol) / 4u;
  }
  let bN = li / 2u;
  let bHalf = li % 2u;
  let bRowBase = eb + (nBase + bN) * rowB;

  for (var kb: u32 = 0u; kb < rowB; kb = kb + 1u) {
    if (aEnt != 0xFFFFFFFFu) {
      let ax = aX + kb * (32u / 4u);
      let v0 = xh[ax];
      let v1 = xh[ax + 1u];
      tA[aRow * 32u + aCol]      = ${TPREC}(v0.x); tA[aRow * 32u + aCol + 1u] = ${TPREC}(v0.y);
      tA[aRow * 32u + aCol + 2u] = ${TPREC}(v0.z); tA[aRow * 32u + aCol + 3u] = ${TPREC}(v0.w);
      tA[aRow * 32u + aCol + 4u] = ${TPREC}(v1.x); tA[aRow * 32u + aCol + 5u] = ${TPREC}(v1.y);
      tA[aRow * 32u + aCol + 6u] = ${TPREC}(v1.z); tA[aRow * 32u + aCol + 7u] = ${TPREC}(v1.w);
    } else {
      for (var j: u32 = 0u; j < 8u; j = j + 1u) { tA[aRow * 32u + aCol + j] = ${TPREC}(0.0); }
    }
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

  let row = sgid / 4u;
  let col = (sgid % 4u) * 2u;
  let nSub = nBase + subx * 32u;
  for (var s: u32 = 0u; s < 8u; s = s + 1u) {
    let strip = s / 4u;
    let bcol = s % 4u;
    var cm: subgroup_matrix_result<f32, 8, 8>;
    switch (s) {
      case 0u: { cm = c00; } case 1u: { cm = c01; } case 2u: { cm = c02; } case 3u: { cm = c03; }
      case 4u: { cm = c10; } case 5u: { cm = c11; } case 6u: { cm = c12; } default: { cm = c13; }
    }
    subgroupMatrixStore(&scr[sub], 0u, cm, false, 8u);
    let tr = suby * 16u + strip * 8u + row;       // tile row 0..31
    let ent = ents[tr];
    if (ent != 0xFFFFFFFFu) {
      let gn = nSub + bcol * 8u + col;
      y[ent * ${OUT}u + gn] = scr[sub][row * 8u + col];
      y[ent * ${OUT}u + gn + 1u] = scr[sub][row * 8u + col + 1u];
    }
  }
}
