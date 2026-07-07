// RoPE in-place over HEADS×HEAD_DIM. Pairing (i, i+HEAD_DIM/2); angle i uses
// theta_i = pos * THETA^(-2i/HEAD_DIM) for i < ROPE_ANGLES, else identity
// (proportional RoPE's zero-frequency tail). pos from params[0].
// Params: HEADS, HEAD_DIM, ROPE_ANGLES, THETA, WG
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<u32>;   // [0]=pos

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let half = ${HEAD_DIM}u / 2u;
  let total = ${HEADS}u * half;
  let idx = gid.x;
  if (idx >= total) { return; }
  let h = idx / half;
  let p = idx % half;
  if (p >= ${ROPE_ANGLES}u) { return; }
  let pos = f32(params[0]);
  let theta = pos * pow(${THETA}, -2.0 * f32(p) / f32(${HEAD_DIM}u));
  let c = cos(theta);
  let s = sin(theta);
  let i0 = h * ${HEAD_DIM}u + p;
  let i1 = i0 + half;
  let x0 = x[i0];
  let x1 = x[i1];
  x[i0] = x0 * c - x1 * s;
  x[i1] = x0 * s + x1 * c;
}
