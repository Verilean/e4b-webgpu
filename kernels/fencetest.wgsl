// Boundary-tax probe: trivial 1-WG dispatch. DEP=1 chains (read+write the same
// counter → every dispatch hazard-fenced); DEP=0 writes disjoint slots (no
// hazards → free to overlap). Params: DEP
@group(0) @binding(0) var<storage, read_write> buf: array<u32>;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  if (lid.x == 0u) {
    if (${DEP}u == 1u) { buf[0] = buf[0] + 1u; }        // same-buffer RMW chain
    else if (${DEP}u == 2u) { buf[0] = src[0] + 1u; }   // ping-pong RAW chain
    else if (${DEP}u == 4u) { buf[0] = lid.x + 7u; }    // pure WAW chain
    else { buf[1u + src[0]] = 1u; }                     // independent
  }
}
