// Merged down dispatch: grid z==0 = dense down (IN2 elems, xD f16 → yD f32);
// z==1 = MoE slot-combined down (IN elems/slot, xS f16 [K][IN/4], all-K loop,
// topkW-weighted sum → yM f32). Both 2-row interleaved subgroup shape.
// Params: IN (expert, 704), IN2 (dense, 2112), OUT, K
enable f16;
enable subgroups;
@group(0) @binding(0) var<storage, read> xD: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> wD: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> wsD: array<u32>;
@group(0) @binding(3) var<storage, read> xS: array<vec4<f16>>;
@group(0) @binding(4) var<storage, read> wE: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read> wsE: array<u32>;
@group(0) @binding(6) var<storage, read> topk: array<u32>;
@group(0) @binding(7) var<storage, read> tkw: array<f32>;
@group(0) @binding(8) var<storage, read_write> yD: array<f32>;
@group(0) @binding(9) var<storage, read_write> yM: array<f32>;

struct XU {
  e0: vec4f, o0: vec4f, e1: vec4f, o1: vec4f,
  e2: vec4f, o2: vec4f, e3: vec4f, o3: vec4f,
};
fn unpxD(jb: u32) -> XU {
  let a0 = vec4<f32>(xD[jb * 8u]);      let a1 = vec4<f32>(xD[jb * 8u + 1u]);
  let a2 = vec4<f32>(xD[jb * 8u + 2u]); let a3 = vec4<f32>(xD[jb * 8u + 3u]);
  let a4 = vec4<f32>(xD[jb * 8u + 4u]); let a5 = vec4<f32>(xD[jb * 8u + 5u]);
  let a6 = vec4<f32>(xD[jb * 8u + 6u]); let a7 = vec4<f32>(xD[jb * 8u + 7u]);
  return XU(a0, a4, a1, a5, a2, a6, a3, a7);
}
fn unpxS(xoff: u32, jb: u32) -> XU {
  let a0 = vec4<f32>(xS[xoff + jb * 8u]);      let a1 = vec4<f32>(xS[xoff + jb * 8u + 1u]);
  let a2 = vec4<f32>(xS[xoff + jb * 8u + 2u]); let a3 = vec4<f32>(xS[xoff + jb * 8u + 3u]);
  let a4 = vec4<f32>(xS[xoff + jb * 8u + 4u]); let a5 = vec4<f32>(xS[xoff + jb * 8u + 5u]);
  let a6 = vec4<f32>(xS[xoff + jb * 8u + 6u]); let a7 = vec4<f32>(xS[xoff + jb * 8u + 7u]);
  return XU(a0, a4, a1, a5, a2, a6, a3, a7);
}
fn bdot(wv: vec4<u32>, u: XU) -> f32 {
  return dot(vec4f(vec4i(unpack4xU8(wv.x & 0x0F0F0F0Fu))) - vec4f(8.0), u.e0)
       + dot(vec4f(vec4i(unpack4xU8((wv.x >> 4u) & 0x0F0F0F0Fu))) - vec4f(8.0), u.o0)
       + dot(vec4f(vec4i(unpack4xU8(wv.y & 0x0F0F0F0Fu))) - vec4f(8.0), u.e1)
       + dot(vec4f(vec4i(unpack4xU8((wv.y >> 4u) & 0x0F0F0F0Fu))) - vec4f(8.0), u.o1)
       + dot(vec4f(vec4i(unpack4xU8(wv.z & 0x0F0F0F0Fu))) - vec4f(8.0), u.e2)
       + dot(vec4f(vec4i(unpack4xU8((wv.z >> 4u) & 0x0F0F0F0Fu))) - vec4f(8.0), u.o2)
       + dot(vec4f(vec4i(unpack4xU8(wv.w & 0x0F0F0F0Fu))) - vec4f(8.0), u.e3)
       + dot(vec4f(vec4i(unpack4xU8((wv.w >> 4u) & 0x0F0F0F0Fu))) - vec4f(8.0), u.o3);
}
fn scD(base: u32, b: u32) -> f32 {
  let i = base + b;
  let two = unpack2x16float(wsD[i / 2u]);
  return select(two.x, two.y, (i & 1u) == 1u);
}
fn scE(base: u32, b: u32) -> f32 {
  let i = base + b;
  let two = unpack2x16float(wsE[i / 2u]);
  return select(two.x, two.y, (i & 1u) == 1u);
}

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let o0 = (wid.y * 32768u + wid.x) * 2u;
  if (o0 >= ${OUT}u) { return; }
  var acc0: f32 = 0.0;
  var acc1: f32 = 0.0;
  if (wid.z == 0u) {                          // dense down
    let rowB = ${IN2}u / 32u;
    let b0 = o0 * rowB;
    let b1 = b0 + rowB;
    for (var jb = lid.x; jb < rowB; jb = jb + 32u) {
      let u = unpxD(jb);
      acc0 = acc0 + scD(b0, jb) * bdot(wD[b0 + jb], u);
      acc1 = acc1 + scD(b1, jb) * bdot(wD[b1 + jb], u);
    }
  } else {                                    // MoE slot-combined down
    let rowB = ${IN}u / 32u;
    for (var k: u32 = 0u; k < ${K}u; k = k + 1u) {
      let eb = topk[k] * (${OUT}u * rowB);
      let xoff = k * (${IN}u / 4u);
      let b0 = eb + o0 * rowB;
      let b1 = b0 + rowB;
      var s0: f32 = 0.0;
      var s1: f32 = 0.0;
      for (var jb = lid.x; jb < rowB; jb = jb + 32u) {
        let u = unpxS(xoff, jb);
        s0 = s0 + scE(b0, jb) * bdot(wE[b0 + jb], u);
        s1 = s1 + scE(b1, jb) * bdot(wE[b1 + jb], u);
      }
      let tw = tkw[k];
      acc0 = acc0 + tw * s0;
      acc1 = acc1 + tw * s1;
    }
  }
  let t0 = subgroupAdd(acc0);
  let t1 = subgroupAdd(acc1);
  if (wid.z == 0u) {
    if (lid.x == 0u) { yD[o0] = t0; }
    if (lid.x == 1u && o0 + 1u < ${OUT}u) { yD[o0 + 1u] = t1; }
  } else {
    if (lid.x == 0u) { yM[o0] = t0; }
    if (lid.x == 1u && o0 + 1u < ${OUT}u) { yM[o0 + 1u] = t1; }
  }
}
