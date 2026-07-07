// Fused PLE input prep: one WG per layer block b.
// identity[i] = int2(pleTable[token], block b) * scale * sqrt(PD)
// pleInput[b*PD+i] = ( rms(proj_b)[i] * normW[b*PD+i] + identity[i] ) * sqrt(1/2)
// Replaces pleRow + rmsPle + addMulPle (3 dispatches + 2 fences).
// Params: PD (block size), LAYERS, MULT (sqrt(PD)), EPS, WG
@group(0) @binding(0) var<storage, read> table: array<u32>;    // packed int2
@group(0) @binding(1) var<storage, read> scales: array<f32>;   // [vocab][LAYERS]
@group(0) @binding(2) var<storage, read> params: array<u32>;   // [2]=token
@group(0) @binding(3) var<storage, read> proj: array<f32>;     // [LAYERS*PD]
@group(0) @binding(4) var<storage, read> normW: array<f32>;    // [LAYERS*PD]
@group(0) @binding(5) var<storage, read_write> pleInput: array<f32>;

var<workgroup> red: array<f32, ${WG}>;

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let b = wid.x;
  let base = b * ${PD}u;
  var s: f32 = 0.0;
  for (var i = lid.x; i < ${PD}u; i = i + ${WG}u) {
    let v = proj[base + i];
    s = s + v * v;
  }
  red[lid.x] = s;
  workgroupBarrier();
  var st = ${WG}u / 2u;
  while (st > 0u) {
    if (lid.x < st) { red[lid.x] = red[lid.x] + red[lid.x + st]; }
    workgroupBarrier();
    st = st / 2u;
  }
  let inv = pow(red[0] / f32(${PD}u) + ${EPS}, -0.5);

  let row = params[2];
  let n = ${LAYERS}u * ${PD}u;
  let rowBytes = n / 4u;
  let sc = scales[row * ${LAYERS}u + b] * ${MULT};
  for (var i = lid.x; i < ${PD}u; i = i + ${WG}u) {
    let gi = base + i;                       // index within the full PLE row
    let byteIdx = row * rowBytes + (gi / 4u);
    let word = table[byteIdx >> 2u];
    let byte = (word >> ((byteIdx & 3u) * 8u)) & 0xFFu;
    let idv = (f32(i32((byte >> ((gi & 3u) * 2u)) & 3u)) - 2.0) * sc;
    pleInput[gi] = (proj[gi] * inv * normW[gi] + idv) * 0.70710678118654752;
  }
}
