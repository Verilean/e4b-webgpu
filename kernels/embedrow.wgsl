// Dequantize ONE int2 embedding row: out[i] = unpack2(table[row]) * scale * MULT.
// Row index = params[PARAM_IDX]. Scales: BLOCKS blocks per row (embed: 1, PLE: 42),
// scale index = row*BLOCKS + i/(N/BLOCKS). Params: N, BLOCKS, MULT, PARAM_IDX, WG
@group(0) @binding(0) var<storage, read> table: array<u32>;    // packed int2 bytes
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@group(0) @binding(3) var<storage, read_write> outv: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${N}u) { return; }
  let row = params[${PARAM_IDX}u];
  let rowBytes = ${N}u / 4u;
  let byteIdx = row * rowBytes + (i / 4u);
  let word = table[byteIdx >> 2u];
  let b = (word >> ((byteIdx & 3u) * 8u)) & 0xFFu;
  let v = f32(i32((b >> ((i & 3u) * 2u)) & 3u)) - 2.0;
  let blockSize = ${N}u / ${BLOCKS}u;
  let s = scales[row * ${BLOCKS}u + i / blockSize];
  outv[i] = v * s * ${MULT};
}
