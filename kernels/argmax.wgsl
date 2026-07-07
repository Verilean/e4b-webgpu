// Single-WG argmax over N logits → out[0]=index (as u32), out[1]=bitcast(max). Params: N, WG
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> outv: array<u32>;
var<workgroup> bestV: array<f32, ${WG}>;
var<workgroup> bestI: array<u32, ${WG}>;
@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  var bv: f32 = -3.0e38;
  var bi: u32 = 0u;
  for (var i = lid.x; i < ${N}u; i = i + ${WG}u) {
    let v = x[i];
    if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
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
  if (lid.x == 0u) { outv[0] = bestI[0]; outv[1] = bitcast<u32>(bestV[0]); }
}
