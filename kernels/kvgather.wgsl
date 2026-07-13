enable f16;
// KV-cache compaction gather: dst slot i <- src slot keep[i], for K and V.
// Grid x = keepCount (runtime). Params: KVH, HD, WG
@group(0) @binding(0) var<storage, read> keep: array<u32>;
@group(0) @binding(1) var<storage, read> srcK: array<f16>;
@group(0) @binding(2) var<storage, read> srcV: array<f16>;
@group(0) @binding(3) var<storage, read_write> dstK: array<f16>;
@group(0) @binding(4) var<storage, read_write> dstV: array<f16>;
@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let i = wid.x;
  let src = keep[i];
  let n = ${KVH}u * ${HD}u;
  for (var e = lid.x; e < n; e = e + ${WG}u) {
    dstK[i * n + e] = srcK[src * n + e];
    dstV[i * n + e] = srcV[src * n + e];
  }
}
