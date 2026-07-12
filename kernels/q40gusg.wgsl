enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;
diagnostic(off, chromium.subgroup_matrix_uniformity);
// GROUPED MoE gate|up GEMM on subgroup matrices: grid.z = expert chunk
// (expgroup descriptors), M-tile = the chunk's ≤MC(=8) entries, N = the
// expert's 2·FF gate|up rows in 128-row strips (grid.x), K = IN in q4_0
// block tiles (32). WG=128 = 4 subgroups, each owning 8M×32N = 4 result mats
// (f32 accum; TPREC=f32 keeps the dequant exact). Raw output per entry goes to
// yraw[(tok*K+slot)][2*FF]; geglu pairing is a separate elementwise pass.
// Params: IN, FF(704), K, MC(8), TPREC
@group(0) @binding(0) var<storage, read> xh: array<vec4<f16>>;   // moeIn [M][IN/4]
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> ws: array<u32>;
@group(0) @binding(3) var<storage, read> chunkExp: array<u32>;
@group(0) @binding(4) var<storage, read> chunkEnt: array<u32>;
@group(0) @binding(5) var<storage, read_write> yraw: array<f32>; // [M*K][2*FF]

var<workgroup> tA: array<${TPREC}, 8 * 32>;     // [entry][k], stride 32
var<workgroup> tB: array<${TPREC}, 128 * 32>;   // [n][k], stride 32
var<workgroup> scr: array<array<f32, 64>, 4>;
var<workgroup> wgExp: u32;
var<workgroup> ents: array<u32, ${MC}>;

fn scaleOf(base: u32, b: u32) -> f32 {
  let i = base + b;
  let two = unpack2x16float(ws[i / 2u]);
  return select(two.x, two.y, (i & 1u) == 1u);
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32,
        @builtin(subgroup_invocation_id) sgid: u32, @builtin(subgroup_size) sgsz: u32) {
  if (li == 0u) { wgExp = chunkExp[wid.z]; }
  if (li < ${MC}u) { ents[li] = chunkEnt[wid.z * ${MC}u + li]; }
  let ce = workgroupUniformLoad(&wgExp);
  if (ce == 0xFFFFFFFFu) { return; }        // sentinel chunk (uniform exit)

  let nBase = wid.x * 128u;                  // strip within the 2*FF rows
  let rowB = ${IN}u / 32u;
  let eb = ce * (2u * ${FF}u * rowB);        // expert base (blocks)
  let sub = li / sgsz;                       // subgroup 0..3, owns 32 N cols
  var c0: subgroup_matrix_result<f32, 8, 8>;
  var c1: subgroup_matrix_result<f32, 8, 8>;
  var c2: subgroup_matrix_result<f32, 8, 8>;
  var c3: subgroup_matrix_result<f32, 8, 8>;

  // A fill: threads 0..31 → entry li/4, 8 elems; B fill: 1 row per thread
  let aEnt = li / 4u;
  let aCol = (li % 4u) * 8u;
  let bRow = eb + (nBase + li) * rowB;

  for (var kb: u32 = 0u; kb < rowB; kb = kb + 1u) {
    if (li < 32u) {
      let ent = ents[aEnt];
      if (ent != 0xFFFFFFFFu) {
        let ax = ((ent / ${K}u) * ${IN}u + kb * 32u + aCol) / 4u;
        let v0 = xh[ax];
        let v1 = xh[ax + 1u];
        tA[aEnt * 32u + aCol]      = ${TPREC}(v0.x); tA[aEnt * 32u + aCol + 1u] = ${TPREC}(v0.y);
        tA[aEnt * 32u + aCol + 2u] = ${TPREC}(v0.z); tA[aEnt * 32u + aCol + 3u] = ${TPREC}(v0.w);
        tA[aEnt * 32u + aCol + 4u] = ${TPREC}(v1.x); tA[aEnt * 32u + aCol + 5u] = ${TPREC}(v1.y);
        tA[aEnt * 32u + aCol + 6u] = ${TPREC}(v1.z); tA[aEnt * 32u + aCol + 7u] = ${TPREC}(v1.w);
      } else {
        for (var j: u32 = 0u; j < 8u; j = j + 1u) { tA[aEnt * 32u + aCol + j] = ${TPREC}(0.0); }
      }
    }
    let wv = w[bRow + kb];
    let d = scaleOf(bRow, kb);
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
      let word = select(select(select(wv.x, wv.y, i == 1u), wv.z, i == 2u), wv.w, i == 3u);
      for (var b: u32 = 0u; b < 4u; b = b + 1u) {
        let e = i * 4u + b;
        tB[li * 32u + e]        = ${TPREC}(d * (f32((word >> (b * 8u)) & 0xFu) - 8.0));
        tB[li * 32u + 16u + e]  = ${TPREC}(d * (f32((word >> (b * 8u + 4u)) & 0xFu) - 8.0));
      }
    }
    workgroupBarrier();

    for (var st: u32 = 0u; st < 32u; st = st + 8u) {
      let a0 = subgroupMatrixLoad<subgroup_matrix_left<${TPREC}, 8, 8>>(&tA, st, false, 32u);
      let bOff = sub * 32u * 32u + st;
      let b0 = subgroupMatrixLoad<subgroup_matrix_right<${TPREC}, 8, 8>>(&tB, bOff, true, 32u);
      let b1 = subgroupMatrixLoad<subgroup_matrix_right<${TPREC}, 8, 8>>(&tB, bOff + 8u * 32u, true, 32u);
      let b2 = subgroupMatrixLoad<subgroup_matrix_right<${TPREC}, 8, 8>>(&tB, bOff + 16u * 32u, true, 32u);
      let b3 = subgroupMatrixLoad<subgroup_matrix_right<${TPREC}, 8, 8>>(&tB, bOff + 24u * 32u, true, 32u);
      c0 = subgroupMatrixMultiplyAccumulate(a0, b0, c0);
      c1 = subgroupMatrixMultiplyAccumulate(a0, b1, c1);
      c2 = subgroupMatrixMultiplyAccumulate(a0, b2, c2);
      c3 = subgroupMatrixMultiplyAccumulate(a0, b3, c3);
    }
    workgroupBarrier();
  }

  // epilogue: rows = chunk entries, scatter to yraw[entry][2*FF]
  let row = sgid / 4u;
  let col = (sgid % 4u) * 2u;
  let nSub = nBase + sub * 32u;
  for (var s: u32 = 0u; s < 4u; s = s + 1u) {
    var cm: subgroup_matrix_result<f32, 8, 8>;
    switch (s) {
      case 0u: { cm = c0; } case 1u: { cm = c1; } case 2u: { cm = c2; } default: { cm = c3; }
    }
    subgroupMatrixStore(&scr[sub], 0u, cm, false, 8u);
    let ent = ents[row];                     // row < 8 = MC
    if (ent != 0xFFFFFFFFu) {
      let gn = nSub + s * 8u + col;
      yraw[ent * 2u * ${FF}u + gn] = scr[sub][row * 8u + col];
      yraw[ent * 2u * ${FF}u + gn + 1u] = scr[sub][row * 8u + col + 1u];
    }
  }
}
