enable f16;
// RMSNorm over ROWS rows of DIM: y = x * (mean(x^2)+EPS)^-0.5 [* weight]
// One workgroup per row. SUMOUT=1: also writes sums[row] = Σ y (lm_head ZP fold).
// Params: DIM, ROWS, EPS, WITH_SCALE (0/1), SUMOUT (0/1), WG
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;   // [DIM] (dummy if WITH_SCALE=0)
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read_write> yh: array<f16>;
@group(0) @binding(3) var<storage, read_write> sums: array<f32>;

var<workgroup> partial: array<f32, ${WG}>;

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wid.x;
  let base = row * ${DIM}u;
  var s: f32 = 0.0;
  for (var i: u32 = lid.x; i < ${DIM}u; i = i + ${WG}u) {
    let v = x[base + i];
    s = s + v * v;
  }
  partial[lid.x] = s;
  workgroupBarrier();
  var stride: u32 = ${WG}u / 2u;
  while (stride > 0u) {
    if (lid.x < stride) { partial[lid.x] = partial[lid.x] + partial[lid.x + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let inv = pow(partial[0] / f32(${DIM}u) + ${EPS}, -0.5);
  _ = sums[0]; _ = y[0]; _ = yh[0];
  var s2: f32 = 0.0;
  for (var i: u32 = lid.x; i < ${DIM}u; i = i + ${WG}u) {
    var v = x[base + i] * inv;
    if (${WITH_SCALE}u == 1u) { v = v * weight[i]; }
    if (${F16OUT}u == 1u) { yh[base + i] = f16(v); } else { y[base + i] = v; }
    s2 = s2 + v;
  }
  if (${SUMOUT}u == 1u) {
    workgroupBarrier();
    partial[lid.x] = s2;
    workgroupBarrier();
    var st = ${WG}u / 2u;
    while (st > 0u) {
      if (lid.x < st) { partial[lid.x] = partial[lid.x] + partial[lid.x + st]; }
      workgroupBarrier();
      st = st / 2u;
    }
    if (lid.x == 0u) { sums[row] = partial[0]; }
  }
}
