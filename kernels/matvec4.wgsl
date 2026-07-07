// Subgroup matvec, R rows interleaved per 32-thread subgroup: one xq load feeds
// R weight rows (cuts x traffic, multiplies loads in flight). No barriers.
// Params: BITS(4/8), IN, OUT, R, SOFTCAP
enable subgroups;
@group(0) @binding(0) var<storage, read> xq: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> wscale: array<f32>;
@group(0) @binding(3) var<uniform> srq: vec2f;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

fn q4dot(wv: vec4<u32>, jw: u32) -> f32 {
  var acc: f32 = 0.0;
  for (var h: u32 = 0u; h < 2u; h = h + 1u) {
    let xv = xq[jw * 2u + h];
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

fn q8dot(wv: vec4<u32>, jw: u32) -> f32 {
  let xv = xq[jw];
  return dot(vec4f(unpack4xI8(xv.x)), vec4f(unpack4xI8(wv.x)))
       + dot(vec4f(unpack4xI8(xv.y)), vec4f(unpack4xI8(wv.y)))
       + dot(vec4f(unpack4xI8(xv.z)), vec4f(unpack4xI8(wv.z)))
       + dot(vec4f(unpack4xI8(xv.w)), vec4f(unpack4xI8(wv.w)));
}

fn finish(total: f32, o: u32) {
  var out = total * srq.x * wscale[o];
  if (srq.y != 0.0) { out = clamp(round(out / srq.y), -128.0, 127.0) * srq.y; }
  if (${SOFTCAP} != 0.0) { out = ${SOFTCAP} * tanh(out / ${SOFTCAP}); }
  y[o] = out;
}

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  // SER serial passes of the proven 2-row interleave (2*SER rows per WG):
  // longer per-thread streams pipeline better than many short WGs.
  let og = (wid.y * 32768u + wid.x) * (2u * ${SER}u);
  if (og >= ${OUT}u) { return; }
  let rowW = select(${IN}u / 16u, ${IN}u / 32u, ${BITS}u == 4u);
  for (var rr: u32 = 0u; rr < ${SER}u; rr = rr + 1u) {
    let o0 = og + rr * 2u;
    if (o0 >= ${OUT}u) { continue; }
    let b0 = o0 * rowW;
    let b1 = b0 + rowW;
    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    for (var jw = lid.x; jw < rowW; jw = jw + 32u) {
      if (${BITS}u == 4u) {
        acc0 = acc0 + q4dot(w[b0 + jw], jw);
        acc1 = acc1 + q4dot(w[b1 + jw], jw);
      } else {
        acc0 = acc0 + q8dot(w[b0 + jw], jw);
        acc1 = acc1 + q8dot(w[b1 + jw], jw);
      }
    }
    let t0 = subgroupAdd(acc0);
    let t1 = subgroupAdd(acc1);
    if (lid.x == 0u) { finish(t0, o0); }
    if (lid.x == 1u && o0 + 1u < ${OUT}u) { finish(t1, o0 + 1u); }
  }
}
