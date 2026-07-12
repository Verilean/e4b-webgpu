// k08-pattern merge (webml OprojNorm class): the o-proj matvec AND the whole
// rmsacc3 epilogue in ONE dispatch. Matvec WGs atomicStore their rows into pp
// and take a ticket; the LAST workgroup re-reads d through the atomics and
// applies: h = hIn + rms(d)·wA;  y1 = rms(h)·w1 (f16), y2 = ·w2 (f32 router),
// y3 = ·w3 (f16), hOut = h.  WG=32 (one subgroup) ⇒ the epilogue reduces with
// subgroupAdd only — no barriers, no Tint-uniformity hazards.
// normCat = [wA | w1 | w2 | w3] (4×H). pp = [H rows (bitcast f32) | counter].
// Params: IN, OUT(=H), TOTAL_WGS, EPS
enable f16;
enable subgroups;
@group(0) @binding(0) var<storage, read> xh: array<vec4<f16>>;   // attnOut (f16)
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> ws: array<u32>;
@group(0) @binding(3) var<storage, read_write> pp: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> hIn: array<f32>;
@group(0) @binding(5) var<storage, read> normCat: array<f32>;    // [4*H]
@group(0) @binding(6) var<storage, read_write> y1h: array<f16>;  // normed (ffn)
@group(0) @binding(7) var<storage, read_write> y2: array<f32>;   // routerIn
@group(0) @binding(8) var<storage, read_write> y3h: array<f16>;  // moeIn
@group(0) @binding(9) var<storage, read_write> hOut: array<f32>;

struct XU {
  e0: vec4f, o0: vec4f, e1: vec4f, o1: vec4f,
  e2: vec4f, o2: vec4f, e3: vec4f, o3: vec4f,
};
fn unpx(jb: u32) -> XU {
  let a0 = vec4<f32>(xh[jb * 8u]);      let a1 = vec4<f32>(xh[jb * 8u + 1u]);
  let a2 = vec4<f32>(xh[jb * 8u + 2u]); let a3 = vec4<f32>(xh[jb * 8u + 3u]);
  let a4 = vec4<f32>(xh[jb * 8u + 4u]); let a5 = vec4<f32>(xh[jb * 8u + 5u]);
  let a6 = vec4<f32>(xh[jb * 8u + 6u]); let a7 = vec4<f32>(xh[jb * 8u + 7u]);
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
fn scaleOf(base: u32, b: u32) -> f32 {
  let i = base + b;
  let two = unpack2x16float(ws[i / 2u]);
  return select(two.x, two.y, (i & 1u) == 1u);
}

var<workgroup> wgLast: u32;

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let lane = lid.x;
  let o0 = (wid.y * 32768u + wid.x) * 2u;
  let rowB = ${IN}u / 32u;
  let b0 = o0 * rowB;
  let b1 = b0 + rowB;
  var acc0: f32 = 0.0;
  var acc1: f32 = 0.0;
  for (var jb = lane; jb < rowB; jb = jb + 32u) {
    let u = unpx(jb);
    acc0 = acc0 + scaleOf(b0, jb) * bdot(w[b0 + jb], u);
    acc1 = acc1 + scaleOf(b1, jb) * bdot(w[b1 + jb], u);
  }
  let t0 = subgroupAdd(acc0);
  let t1 = subgroupAdd(acc1);
  if (lane == 0u) {
    atomicStore(&pp[o0], bitcast<u32>(t0));
    atomicStore(&pp[o0 + 1u], bitcast<u32>(t1));
    var fl: u32 = 0u;
    let done = atomicAdd(&pp[${OUT}u], 1u) + 1u;
    if (done == ${TOTAL_WGS}u) {
      atomicStore(&pp[${OUT}u], 0u);           // reset for the next layer/token
      fl = 1u;
    }
    wgLast = fl;
  }
  let last = workgroupUniformLoad(&wgLast);    // Tint-provably uniform
  if (last == 0u) { return; }

  // ---- last-WG epilogue (single subgroup, barrier-free) ----
  var s1: f32 = 0.0;
  for (var i = lane; i < ${OUT}u; i = i + 32u) {
    let d = bitcast<f32>(atomicLoad(&pp[i]));
    s1 = s1 + d * d;
  }
  let inv1 = pow(subgroupAdd(s1) / f32(${OUT}u) + ${EPS}, -0.5);
  var s2: f32 = 0.0;
  for (var i = lane; i < ${OUT}u; i = i + 32u) {
    let d = bitcast<f32>(atomicLoad(&pp[i]));
    let h = hIn[i] + d * inv1 * normCat[i];
    hOut[i] = h;
    s2 = s2 + h * h;
  }
  let inv2 = pow(subgroupAdd(s2) / f32(${OUT}u) + ${EPS}, -0.5);
  for (var i = lane; i < ${OUT}u; i = i + 32u) {
    let v = hOut[i] * inv2;                    // own-lane re-read (same thread wrote it)
    y1h[i] = f16(v * normCat[${OUT}u + i]);
    y2[i]  = v * normCat[2u * ${OUT}u + i];
    y3h[i] = f16(v * normCat[3u * ${OUT}u + i]);
  }
}
