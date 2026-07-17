enable f16;
enable chromium_experimental_subgroup_matrix;
diagnostic(off, chromium.subgroup_matrix_uniformity);

// Hand-WGSL twin of the hand-MSL q4k_grouped_reg_indexed (metal_replace.mm):
// IDENTICAL algorithm/tiling to the DSL-generated kernel k5394720408…, authored
// with MSL-style hygiene: hoisted locals, loops instead of macro-expansion,
// unpack2x16float for the d/dmin f16 pair (vs the generated exp2-chain manual
// decode), ONE u32 load per qs byte-pair (vs two), staged dequant locals.
// M-Metal Stage 1: measures what WGSL authorship alone recovers of the 1.61x.

@group(0) @binding(0)
var<storage, read_write> src: array<f32, 785664>;
@group(0) @binding(1)
var<storage, read_write> idx: array<u32, 6336>;
@group(0) @binding(2)
var<storage, read_write> b: array<u32, 71368704>;
@group(0) @binding(3)
var<storage, read_write> c: array<f32, 8921088>;
@group(0) @binding(4)
var<storage, read_write> tileExpert: array<u32, 198>;
@group(0) @binding(5)
var<storage, read_write> tileRows: array<u32, 198>;

var<workgroup> shared_A: array<f16, 1024>;
var<workgroup> shared_B: array<f16, 1024>;
var<workgroup> shared_dq: array<f32, 576>;

fn smByte(p: u32, s0: u32, s1: u32, s2: u32) -> u32 {
  if (p < 4u) { return (s0 >> (p * 8u)) & 0xFFu; }
  if (p < 8u) { return (s1 >> ((p - 4u) * 8u)) & 0xFFu; }
  return (s2 >> ((p - 8u) * 8u)) & 0xFFu;
}

fn scaleMin(j: u32, s0: u32, s1: u32, s2: u32) -> vec2<f32> {
  if (j < 4u) {
    return vec2<f32>(f32(smByte(j, s0, s1, s2) & 63u), f32(smByte(j + 4u, s0, s1, s2) & 63u));
  }
  let sc = (smByte(j + 4u, s0, s1, s2) & 0xFu) | ((smByte(j - 4u, s0, s1, s2) >> 6u) << 4u);
  let m  = (smByte(j + 4u, s0, s1, s2) >> 4u)  | ((smByte(j, s0, s1, s2) >> 6u) << 4u);
  return vec2<f32>(f32(sc), f32(m));
}

@compute
@workgroup_size(128, 1, 1)
fn main(@builtin(local_invocation_id) local_invocation_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>) {
  var Ax: array<subgroup_matrix_left<f16, 8, 8>, 2>;
  Ax[0u] = subgroup_matrix_left<f16, 8, 8>(0);
  Ax[1u] = subgroup_matrix_left<f16, 8, 8>(0);
  var Bx: array<subgroup_matrix_right<f16, 8, 8>, 2>;
  Bx[0u] = subgroup_matrix_right<f16, 8, 8>(0);
  Bx[1u] = subgroup_matrix_right<f16, 8, 8>(0);
  var Cx: array<subgroup_matrix_result<f32, 8, 8>, 4>;
  Cx[0u] = subgroup_matrix_result<f32, 8, 8>(0);
  Cx[1u] = subgroup_matrix_result<f32, 8, 8>(0);
  Cx[2u] = subgroup_matrix_result<f32, 8, 8>(0);
  Cx[3u] = subgroup_matrix_result<f32, 8, 8>(0);

  let tid = local_invocation_id.x;
  let sg = tid / 32u;
  let sgRow = sg % 2u;
  let sgCol = sg / 2u;
  let rowBase = workgroup_id.y * 32u;
  let colBase = workgroup_id.x * 32u;
  let teRaw = tileExpert[workgroup_id.y];
  let isActive = teRaw < 128u;
  let e = select(127u, teRaw, isActive);
  let wro = e * 1408u;
  let trRaw = tileRows[workgroup_id.y];
  let frag0 = isActive && ((sgRow * 16u) < trRaw);
  let frag1 = isActive && ((sgRow * 16u + 8u) < trRaw);
  let tr8 = ((trRaw + 7u) / 8u) * 8u;

  for (var blockIdx: u32 = 0u; blockIdx < 11u; blockIdx = blockIdx + 1u) {
    if (tid < 32u && isActive) {
      let row = wro + colBase + tid;
      let bb = row * 396u + blockIdx * 36u;
      let ddm = unpack2x16float(b[bb]);
      let s0 = b[bb + 1u];
      let s1 = b[bb + 2u];
      let s2 = b[bb + 3u];
      let base = tid * 18u;
      shared_dq[base] = ddm.x;
      shared_dq[base + 1u] = ddm.y;
      for (var j: u32 = 0u; j < 8u; j = j + 1u) {
        let sm = scaleMin(j, s0, s1, s2);
        shared_dq[base + 2u + j] = sm.x;
        shared_dq[base + 10u + j] = sm.y;
      }
    }
    workgroupBarrier();
    for (var jSub: u32 = 0u; jSub < 8u; jSub = jSub + 1u) {
      let kBase = blockIdx * 256u + jSub * 32u;
      let chunk = jSub / 2u;
      let isHigh = (jSub % 2u) == 1u;
      if (isActive) {
        for (var s: u32 = 0u; s < 8u; s = s + 1u) {
          let flat = tid + s * 128u;
          let m = flat / 32u;
          let k = flat % 32u;
          if (m < tr8) {
            let tok = idx[rowBase + m];
            let x = src[tok * 2816u + kBase + k];
            let blk = (m / 8u) * 4u + k / 8u;
            let within = (m % 8u) * 8u + (k % 8u);
            shared_A[blk * 64u + within] = f16(x);
          }
        }
        for (var s: u32 = 0u; s < 4u; s = s + 1u) {
          let u = tid + s * 128u;
          let n = u / 16u;
          let kpair = u % 16u;
          let row = wro + colBase + n;
          let bbase = row * 396u + blockIdx * 36u;
          let dqb = n * 18u;
          let d = shared_dq[dqb];
          let dmin = shared_dq[dqb + 1u];
          let sc = shared_dq[dqb + 2u + jSub];
          let mv = shared_dq[dqb + 10u + jSub];
          let k0 = kpair * 2u;
          let byteIdx = chunk * 32u + k0;
          let w = b[bbase + 4u + byteIdx / 4u];
          let sh = (byteIdx % 4u) * 8u;
          let b0 = (w >> sh) & 255u;
          let b1 = (w >> (sh + 8u)) & 255u;
          let q0 = f32(select(b0 & 15u, b0 >> 4u, isHigh));
          let q1 = f32(select(b1 & 15u, b1 >> 4u, isHigh));
          let y0 = d * (sc * q0) - dmin * mv;
          let y1 = d * (sc * q1) - dmin * mv;
          let blkB = (k0 / 8u) * 4u + n / 8u;
          let baseB = blkB * 64u + (n % 8u);
          let kr = k0 % 8u;
          shared_B[baseB + kr * 8u] = f16(y0);
          shared_B[baseB + (kr + 1u) * 8u] = f16(y1);
        }
      }
      workgroupBarrier();
      for (var k8: u32 = 0u; k8 < 4u; k8 = k8 + 1u) {
        if (frag0) {
          Ax[0u] = subgroupMatrixLoad<subgroup_matrix_left<f16,8,8>>(&shared_A, ((sgRow * 2u) * 4u + k8) * 64u, false, 8u);
          Bx[0u] = subgroupMatrixLoad<subgroup_matrix_right<f16,8,8>>(&shared_B, (k8 * 4u + sgCol * 2u) * 64u, false, 8u);
          Bx[1u] = subgroupMatrixLoad<subgroup_matrix_right<f16,8,8>>(&shared_B, (k8 * 4u + sgCol * 2u + 1u) * 64u, false, 8u);
          Cx[0u] = subgroupMatrixMultiplyAccumulate(Ax[0u], Bx[0u], Cx[0u]);
          Cx[1u] = subgroupMatrixMultiplyAccumulate(Ax[0u], Bx[1u], Cx[1u]);
        }
        if (frag1) {
          Ax[1u] = subgroupMatrixLoad<subgroup_matrix_left<f16,8,8>>(&shared_A, ((sgRow * 2u + 1u) * 4u + k8) * 64u, false, 8u);
          Cx[2u] = subgroupMatrixMultiplyAccumulate(Ax[1u], Bx[0u], Cx[2u]);
          Cx[3u] = subgroupMatrixMultiplyAccumulate(Ax[1u], Bx[1u], Cx[3u]);
        }
      }
      workgroupBarrier();
    }
  }
  let mOff = sgRow * 16u;
  let nOff = sgCol * 16u;
  if (frag0) {
    subgroupMatrixStore(&c, ((rowBase + mOff) * 1408u) + (colBase + nOff), Cx[0u], false, 1408u);
    subgroupMatrixStore(&c, ((rowBase + mOff) * 1408u) + (colBase + nOff + 8u), Cx[1u], false, 1408u);
  }
  if (frag1) {
    subgroupMatrixStore(&c, ((rowBase + mOff + 8u) * 1408u) + (colBase + nOff), Cx[2u], false, 1408u);
    subgroupMatrixStore(&c, ((rowBase + mOff + 8u) * 1408u) + (colBase + nOff + 8u), Cx[3u], false, 1408u);
  }
}
