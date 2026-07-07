// safetensors range-fetch loader for the QAT-mobile checkpoint.
// Format spec: NOTES/qat-format.md. Only model.language_model.* / lm_head.* are used.

export async function openSafetensors(url) {
  const head = await (await fetch(url, { headers: { Range: "bytes=0-7" } })).arrayBuffer();
  const n = Number(new DataView(head).getBigUint64(0, true));
  const hj = await (await fetch(url, { headers: { Range: `bytes=8-${8 + n - 1}` } })).arrayBuffer();
  const index = JSON.parse(new TextDecoder().decode(hj));
  delete index.__metadata__;
  const base = 8 + n;
  return {
    url,
    index,
    has: (name) => name in index,
    info: (name) => index[name],
    async fetch(name) {
      const t = index[name];
      if (!t) throw new Error(`tensor not found: ${name}`);
      const [a, b] = t.data_offsets;
      const r = await fetch(url, { headers: { Range: `bytes=${base + a}-${base + b - 1}` } });
      return { ...t, buf: await r.arrayBuffer() };
    },
  };
}

export function bf16ToF32(buf) {
  const u16 = new Uint16Array(buf);
  const out = new Float32Array(u16.length);
  const u32 = new Uint32Array(out.buffer);
  for (let i = 0; i < u16.length; i++) u32[i] = u16[i] << 16;
  return out;
}

export function f32Scalar(buf) {
  return new Float32Array(buf)[0] ?? 0;
}
