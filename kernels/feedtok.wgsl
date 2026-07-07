// GPU-side greedy feedback: params[2] (token id) = argmax result of the
// previous step. Lets decode run without any CPU readback in the loop.
@group(0) @binding(0) var<storage, read> amax: array<u32>;
@group(0) @binding(1) var<storage, read_write> params: array<u32>;
@compute @workgroup_size(1)
fn main() { params[2] = amax[0]; }
