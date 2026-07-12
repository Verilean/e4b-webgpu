// GGUF v3 range-fetch reader (browser). Parses the header once, then fetches
// tensor data by byte range. Format verified against goldens/gguf_inspect.py.
export const GGML = { F32: 0, F16: 1, Q4_0: 2, Q8_0: 8, Q6_K: 14, BF16: 30 };

const TYPE_SIZES = {           // [blockBytes, blockElems]
  [GGML.F32]: [4, 1], [GGML.F16]: [2, 1], [GGML.BF16]: [2, 1],
  [GGML.Q4_0]: [18, 32], [GGML.Q8_0]: [34, 32], [GGML.Q6_K]: [210, 256],
};

export function tensorBytes(t) {
  const [bb, be] = TYPE_SIZES[t.type];
  const n = t.dims.reduce((a, b) => a * b, 1);
  return (n / be) * bb;
}

export async function openGGUF(url) {
  // header: read a generous prefix (tokenizer KV arrays are skipped by parse)
  const head = await (await fetch(url, { headers: { Range: "bytes=0-67108863" } })).arrayBuffer();
  const dv = new DataView(head);
  const dec = new TextDecoder();
  let off = 0;
  const u32 = () => { const v = dv.getUint32(off, true); off += 4; return v; };
  const u64 = () => { const v = dv.getBigUint64(off, true); off += 8; return Number(v); };
  const str = () => { const n = u64(); const s = dec.decode(new Uint8Array(head, off, n)); off += n; return s; };
  const SCALAR = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 };
  function val(t) {
    if (t === 8) return str();
    if (t === 9) {                        // array
      const et = u32(); const n = u64();
      if (et === 8) { const out = []; for (let i = 0; i < n; i++) out.push(str()); return out; }
      const out = [];
      for (let i = 0; i < n; i++) out.push(val(et));
      return out;
    }
    const sz = SCALAR[t];
    let v;
    if (t === 6) v = dv.getFloat32(off, true);
    else if (t === 12) v = dv.getFloat64(off, true);
    else if (t === 7) v = dv.getUint8(off) !== 0;
    else if (sz === 1) v = dv.getUint8(off);
    else if (sz === 2) v = dv.getUint16(off, true);
    else if (sz === 4) v = dv.getUint32(off, true);
    else v = Number(dv.getBigUint64(off, true));
    off += sz;
    return v;
  }
  if (dec.decode(new Uint8Array(head, 0, 4)) !== "GGUF") throw new Error("not GGUF");
  off = 4;
  const version = u32();
  const nTensors = u64();
  const nKV = u64();
  const kv = {};
  for (let i = 0; i < nKV; i++) {
    const k = str(); const t = u32();
    // skip the giant tokenizer arrays' VALUES but keep scalar metadata
    kv[k] = val(t);
    if (k.startsWith("tokenizer.")) delete kv[k];   // not needed; free memory
  }
  const align = kv["general.alignment"] ?? 32;
  const tensors = {};
  for (let i = 0; i < nTensors; i++) {
    const name = str();
    const nd = u32();
    const dims = [];
    for (let d = 0; d < nd; d++) dims.push(u64());
    const type = u32();
    const offset = u64();
    tensors[name] = { name, dims, type, offset };
  }
  const dataStart = Math.ceil(off / align) * align;
  return {
    kv, tensors, dataStart,
    async fetch(name) {
      const t = tensors[name];
      if (!t) throw new Error("no tensor " + name);
      const start = dataStart + t.offset;
      const end = start + tensorBytes(t) - 1;
      const buf = await (await fetch(url, { headers: { Range: `bytes=${start}-${end}` } })).arrayBuffer();
      return { ...t, buf };
    },
  };
}
