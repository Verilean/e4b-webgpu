// PLE gate matvec (int8, IN=hidden → OUT=pleDim) with fused epilogue:
// xq[o] byte = quant( gelu(SRQ_out(dot)) * ple[OFF+o] , projInS )
// One WG per 4 rows (packs one u32 of the projection's input).
// Params: IN, OUT, OFF
enable subgroups;
@group(0) @binding(0) var<storage, read> xq: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> wscale: array<f32>;
@group(0) @binding(3) var<storage, read> srqs: array<f32>;   // [gateIn, gateOut, projIn]
@group(0) @binding(4) var<storage, read> ple: array<f32>;
@group(0) @binding(5) var<storage, read_write> xqOut: array<u32>;

var<workgroup> vals: array<f32, 4>;

fn q8dotx(wv: vec4<u32>, x0: vec4f, x1: vec4f, x2: vec4f, x3: vec4f) -> f32 {
  return dot(x0, unpack4x8snorm(wv.x)) + dot(x1, unpack4x8snorm(wv.y))
       + dot(x2, unpack4x8snorm(wv.z)) + dot(x3, unpack4x8snorm(wv.w));
}

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let o0 = wid.x * 4u;
  let rowW = ${IN}u / 16u;
  var a0: f32 = 0.0; var a1: f32 = 0.0; var a2: f32 = 0.0; var a3: f32 = 0.0;
  for (var jw = lid.x; jw < rowW; jw = jw + 32u) {
    let xv = xq[jw];
    let x0 = unpack4x8snorm(xv.x);
    let x1 = unpack4x8snorm(xv.y);
    let x2 = unpack4x8snorm(xv.z);
    let x3 = unpack4x8snorm(xv.w);
    a0 = a0 + q8dotx(w[(o0     ) * rowW + jw], x0, x1, x2, x3);
    a1 = a1 + q8dotx(w[(o0 + 1u) * rowW + jw], x0, x1, x2, x3);
    a2 = a2 + q8dotx(w[(o0 + 2u) * rowW + jw], x0, x1, x2, x3);
    a3 = a3 + q8dotx(w[(o0 + 3u) * rowW + jw], x0, x1, x2, x3);
  }
  let t0 = subgroupAdd(a0);
  let t1 = subgroupAdd(a1);
  let t2 = subgroupAdd(a2);
  let t3 = subgroupAdd(a3);
  if (lid.x < 4u) {
    var t: f32;
    if (lid.x == 0u) { t = t0; } else if (lid.x == 1u) { t = t1; }
    else if (lid.x == 2u) { t = t2; } else { t = t3; }
    let o = o0 + lid.x;
    var y = 16129.0 * t * srqs[0] * wscale[o];
    let outS = srqs[1];
    if (outS != 0.0) { y = clamp(round(y / outS), -128.0, 127.0) * outS; }
    let g = 0.5 * y * (1.0 + tanh(clamp(0.7978845608028654 * (y + 0.044715 * y*y*y), -20.0, 20.0)));
    vals[lid.x] = g * ple[${OFF}u + o];
  }
  workgroupBarrier();
  if (lid.x == 0u) {
    var packed: u32 = 0u;
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
      let qv = i32(clamp(round(vals[k] / srqs[2]), -127.0, 127.0));
      packed = packed | ((u32(qv) & 0xFFu) << (k * 8u));
    }
    xqOut[wid.x] = packed;
  }
}
