// Two-stage argmax, stage selected by STAGE. Stage 0: PARTS WGs each reduce a
// slice into part[]. Stage 1: one WG reduces part[]. Params: N, PARTS, STAGE, WG
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> part: array<vec2<u32>>; // (idx, bits(val))
@group(0) @binding(2) var<storage, read_write> outv: array<u32>;
var<workgroup> bestV: array<f32, ${WG}>;
var<workgroup> bestI: array<u32, ${WG}>;
@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  // keep all bindings live in both stages (layout:"auto" drops DCE'd bindings)
  _ = x[0];
  _ = outv[0];
  var bv: f32 = -3.0e38;
  var bi: u32 = 0u;
  if (${STAGE}u == 0u) {
    let per = (${N}u + ${PARTS}u - 1u) / ${PARTS}u;
    let lo = wid.x * per;
    let hi = min(lo + per, ${N}u);
    for (var i = lo + lid.x; i < hi; i = i + ${WG}u) {
      let v = x[i];
      if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
    }
  } else {
    for (var i = lid.x; i < ${PARTS}u; i = i + ${WG}u) {
      let p = part[i];
      let v = bitcast<f32>(p.y);
      if (v > bv || (v == bv && p.x < bi)) { bv = v; bi = p.x; }
    }
  }
  bestV[lid.x] = bv; bestI[lid.x] = bi;
  workgroupBarrier();
  var stride = ${WG}u / 2u;
  while (stride > 0u) {
    if (lid.x < stride) {
      let ov = bestV[lid.x + stride];
      let oi = bestI[lid.x + stride];
      if (ov > bestV[lid.x] || (ov == bestV[lid.x] && oi < bestI[lid.x])) {
        bestV[lid.x] = ov; bestI[lid.x] = oi;
      }
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (lid.x == 0u) {
    if (${STAGE}u == 0u) { part[wid.x] = vec2<u32>(bestI[0], bitcast<u32>(bestV[0])); }
    else { outv[0] = bestI[0]; outv[1] = bitcast<u32>(bestV[0]); }
  }
}
