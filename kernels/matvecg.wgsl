// Grouped subgroup matvec (concat of up to 3 linears sharing one norm+quant
// dispatch): rows [0,B0) = group 0, [B0,B1) = group 1, [B1,OUT) = group 2.
// Group g reads xq region g and srqs[g] = (inS, outS). 2 rows/WG interleaved.
// Params: BITS(4), IN, OUT, B0, B1
enable subgroups;
@group(0) @binding(0) var<storage, read> xq: array<vec4<u32>>;   // regions of IN/16
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> wscale: array<f32>;
@group(0) @binding(3) var<storage, read> srqs: array<vec2<f32>>; // [group] (inS,outS)
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

fn q4dot(wv: vec4<u32>, xoff: u32, jw: u32) -> f32 {
  var acc: f32 = 0.0;
  for (var h: u32 = 0u; h < 2u; h = h + 1u) {
    let xv = xq[xoff + jw * 2u + h];
    let a0 = vec4f(unpack4xI8(xv.x));
    let a1 = vec4f(unpack4xI8(xv.y));
    let a2 = vec4f(unpack4xI8(xv.z));
    let a3 = vec4f(unpack4xI8(xv.w));
    let w0 = wv[h * 2u];
    let w1 = wv[h * 2u + 1u];
    let lo0 = vec4f(vec4i(unpack4xU8(w0 & 0x0F0F0F0Fu))) - vec4f(8.0);
    let hi0 = vec4f(vec4i(unpack4xU8((w0 >> 4u) & 0x0F0F0F0Fu))) - vec4f(8.0);
    let lo1 = vec4f(vec4i(unpack4xU8(w1 & 0x0F0F0F0Fu))) - vec4f(8.0);
    let hi1 = vec4f(vec4i(unpack4xU8((w1 >> 4u) & 0x0F0F0F0Fu))) - vec4f(8.0);
    acc = acc + dot(lo0, vec4f(a0.x, a0.z, a1.x, a1.z))
              + dot(hi0, vec4f(a0.y, a0.w, a1.y, a1.w))
              + dot(lo1, vec4f(a2.x, a2.z, a3.x, a3.z))
              + dot(hi1, vec4f(a2.y, a2.w, a3.y, a3.w));
  }
  return acc;
}

fn grp(o: u32) -> u32 {
  if (o < ${B0}u) { return 0u; }
  if (o < ${B1}u) { return 1u; }
  return 2u;
}

fn finish(total: f32, o: u32) {
  let sq = srqs[grp(o)];
  var out = total * sq.x * wscale[o];
  if (sq.y != 0.0) { out = clamp(round(out / sq.y), -128.0, 127.0) * sq.y; }
  y[o] = out;
}

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let o0 = (wid.y * 32768u + wid.x) * 2u;
  if (o0 >= ${OUT}u) { return; }
  let rowW = ${IN}u / 32u;
  let xr = ${IN}u / 16u;                 // xq region stride in vec4s
  let b0 = o0 * rowW;
  // group boundaries are even → a row pair never straddles groups; ONE xq
  // stream for both rows (two streams would defeat load CSE — measured 25% loss)
  let xoff = grp(o0) * xr;
  var acc0: f32 = 0.0;
  var acc1: f32 = 0.0;
  for (var jw = lid.x; jw < rowW; jw = jw + 32u) {
    acc0 = acc0 + q4dot(w[b0 + jw], xoff, jw);
    acc1 = acc1 + q4dot(w[b0 + rowW + jw], xoff, jw);
  }
  let t0 = subgroupAdd(acc0);
  let t1 = subgroupAdd(acc1);
  if (lid.x == 0u) { finish(t0, o0); }
  if (lid.x == 1u && o0 + 1u < ${OUT}u) { finish(t1, o0 + 1u); }
}
