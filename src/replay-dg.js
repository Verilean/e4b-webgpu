// DG->JS port M1: thin replayer for a hesper JSTrace package (docs/JS_REPLAY_TRACE.md
// in the hesper repo). Loads weights (GGUF ranges + derived .bin dumps), replays the
// step-0 dispatch stream, and gates on BIT-EQUALITY of the recorded readbacks.
import { openGGUF, tensorBytes } from "./gguf.js";

const L = (m) => { console.log(m); return fetch("/log", { method: "POST", body: String(m) }).catch(() => {}); };

function hexToBytes(hex) {
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(hex.substr(i * 2, 2), 16);
  return out;
}

async function initDeviceBig(maxBuf) {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("no WebGPU adapter");
  const need = Math.max(maxBuf, 1 << 28);
  if (adapter.limits.maxBufferSize < need)
    L(`WARN adapter maxBufferSize ${adapter.limits.maxBufferSize} < needed ${need}`);
  const device = await adapter.requestDevice({
    requiredFeatures: ["subgroups", "shader-f16", "chromium-experimental-subgroup-matrix"]
      .filter((f) => adapter.features.has(f)),
    requiredLimits: {
      maxStorageBufferBindingSize: Math.min(need, adapter.limits.maxStorageBufferBindingSize),
      maxBufferSize: Math.min(need, adapter.limits.maxBufferSize),
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
      maxStorageBuffersPerShaderStage: Math.min(16, adapter.limits.maxStorageBuffersPerShaderStage),
    },
  });
  device.addEventListener("uncapturederror", (e) => L("WEBGPU ERROR: " + e.error.message));
  return device;
}

async function fetchInto(device, buf, url, offset = 0) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  const data = await r.arrayBuffer();
  device.queue.writeBuffer(buf, offset, data, 0, data.byteLength & ~3);
  if (data.byteLength & 3) {                       // non-4-aligned tail
    const tail = new Uint8Array(4);
    tail.set(new Uint8Array(data, data.byteLength & ~3));
    device.queue.writeBuffer(buf, offset + (data.byteLength & ~3), tail);
  }
  return data.byteLength;
}

async function readbackRange(device, buf, offset, size) {
  const sz = (size + 3) & ~3;
  const staging = device.createBuffer({ size: sz, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, offset, staging, 0, sz);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = new Uint8Array(staging.getMappedRange().slice(0, size));
  staging.destroy();
  return out;
}

export async function runReplay(dir = "dgtrace", ggufUrl = "model-dg.gguf") {
  const t00 = performance.now();
  const [bufSizes, tensors, opsText] = await Promise.all([
    fetch(`${dir}/buffers.json`).then((r) => r.json()),
    fetch(`${dir}/tensors.json`).then((r) => r.json()),
    fetch(`${dir}/ops.jsonl`).then((r) => r.text()),
  ]);
  // "k" is a 64-bit kernel hash — quote it before JSON.parse (doubles lose
  // integer precision past 2^53, mangling the k<hash>.wgsl filename)
  const ops = opsText.split("\n").filter(Boolean)
    .map((l) => JSON.parse(l.replace(/"k":(\d+)/, '"k":"$1"')));
  const ref = new Set();
  for (const o of ops) {
    if (o.t === "d") for (const [, u] of o.b) ref.add(String(u));
    else if (o.t === "w" || o.t === "r") ref.add(String(o.u));
  }
  let maxBuf = 0, total = 0;
  for (const u of ref) { const s = bufSizes[u] | 0; maxBuf = Math.max(maxBuf, s); total += s; }
  L(`replay: ${ops.length} events, ${ref.size} buffers (${(total / 1e9).toFixed(1)}GB, max ${(maxBuf / 1e6).toFixed(0)}MB)`);

  const device = await initDeviceBig(maxBuf);

  // 1. allocate every referenced buffer
  const bufs = new Map();
  for (const u of ref) {
    const size = Math.max(4, (bufSizes[u] + 3) & ~3);
    bufs.set(u, device.createBuffer({
      size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC }));
  }

  // 2. contents: GGUF-provenance tensors by range, everything else from .bin dumps
  const gguf = await openGGUF(ggufUrl);
  let loaded = 0;
  for (const [u, name] of Object.entries(tensors)) {
    if (!ref.has(u)) continue;
    const t = gguf.tensors[name];
    if (!t) { L(`WARN tensor ${name} not in GGUF`); continue; }
    const off = gguf.dataStart + t.offset;
    const r = await fetch(ggufUrl, { headers: { Range: `bytes=${off}-${off + tensorBytes(t) - 1}` } });
    const data = await r.arrayBuffer();
    device.queue.writeBuffer(bufs.get(u), 0, data, 0, data.byteLength & ~3);
    loaded += data.byteLength;
  }
  L(`gguf tensors loaded: ${(loaded / 1e9).toFixed(1)}GB`);
  loaded = 0;
  const tset = new Set(Object.keys(tensors));
  for (const u of ref) {
    if (tset.has(u)) continue;
    loaded += await fetchInto(device, bufs.get(u), `${dir}/b${u}.bin`).catch(() => 0);
  }
  L(`derived .bin loaded: ${(loaded / 1e9).toFixed(1)}GB (t+${((performance.now() - t00) / 1000).toFixed(0)}s)`);

  // 3. pipelines (async compile, all in parallel)
  const kids = [...new Set(ops.filter((o) => o.t === "d").map((o) => o.k))];
  const pipes = new Map();
  await Promise.all(kids.map(async (k) => {
    const code = await fetch(`${dir}/k${k}.wgsl`).then((r) => r.text());
    const entry = ops.find((o) => o.t === "d" && o.k === k).n || "main";
    const module = device.createShaderModule({ code });
    pipes.set(k, await device.createComputePipelineAsync({
      layout: "auto", compute: { module, entryPoint: entry } }));
  }));
  L(`pipelines compiled: ${pipes.size} (t+${((performance.now() - t00) / 1000).toFixed(0)}s)`);

  // 4. replay the stream; gate on recorded readbacks
  const bgCache = new Map();
  const bgFor = (o) => {
    const sig = o.k + "|" + o.b.map(([, u]) => u).join(",");
    let bg = bgCache.get(sig);
    if (!bg) {
      bg = device.createBindGroup({
        layout: pipes.get(o.k).getBindGroupLayout(0),
        entries: o.b.map(([, u], i) => ({ binding: i, resource: { buffer: bufs.get(String(u)) } })),
      });
      bgCache.set(sig, bg);
    }
    return bg;
  };
  let enc = device.createCommandEncoder();
  let pass = null;
  let nd = 0, nOK = 0, nBAD = 0;
  const t1 = performance.now();
  const flushSubmit = () => {
    if (pass) { pass.end(); pass = null; }
    device.queue.submit([enc.finish()]);
    enc = device.createCommandEncoder();
  };
  for (const o of ops) {
    if (o.t === "w" && o.hex) {
      if (pass) { pass.end(); pass = null; }           // keep write/dispatch order
      device.queue.submit([enc.finish()]); enc = device.createCommandEncoder();
      const data = hexToBytes(o.hex);
      device.queue.writeBuffer(bufs.get(String(o.u)), o.o, data, 0, data.byteLength & ~3);
    } else if (o.t === "d") {
      if (!pass) pass = enc.beginComputePass();
      pass.setPipeline(pipes.get(o.k));
      pass.setBindGroup(0, bgFor(o));
      pass.dispatchWorkgroups(o.g[0], o.g[1], o.g[2]);
      nd++;
    } else if (o.t === "f") {
      flushSubmit();
    } else if (o.t === "r") {
      flushSubmit();
      const got = await readbackRange(device, bufs.get(String(o.u)), o.o, o.s);
      if (o.hex) {
        const want = hexToBytes(o.hex);
        let diff = -1;
        for (let i = 0; i < want.length; i++) if (got[i] !== want[i]) { diff = i; break; }
        if (diff < 0) { nOK++; }
        else {
          nBAD++;
          L(`READBACK MISMATCH u=${o.u} size=${o.s}: first diff @${diff} got=${got[diff]} want=${want[diff]}`);
        }
      }
    } else if (o.t === "m") {
      L(`marker: ${o.tag}`);
    }
  }
  flushSubmit();
  await device.queue.onSubmittedWorkDone();
  const ms = performance.now() - t1;
  L(`replay done: ${nd} dispatches in ${ms.toFixed(0)}ms | readback gate: ${nOK} OK, ${nBAD} MISMATCH`);

  // localization: compare every non-tensor buffer against the post-state
  // dump; the FIRST diverging buffer in dispatch order = the first bad kernel
  if (nBAD > 0) {
    // rank by LAST binding dispatch (ping-pong buffers are overwritten many
    // times; the final content belongs to the last binder). The earliest
    // last-binder among diverged buffers ≈ the first bad kernel.
    const lastBind = new Map();
    let di = 0;
    for (const o of ops) if (o.t === "d") {
      for (const [, u] of o.b) lastBind.set(String(u), di);
      di++;
    }
    const dl = ops.filter((o) => o.t === "d");
    const diverged = [];
    let checked = 0;
    for (const u of ref) {
      if (tset.has(u)) continue;
      const r = await fetch(`${dir}/b${u}.post.bin`);
      if (!r.ok) continue;
      const want = new Uint8Array(await r.arrayBuffer());
      const got = await readbackRange(device, bufs.get(u), 0, want.length);
      checked++;
      let diff = -1, nd2 = 0;
      for (let i = 0; i < want.length; i++) {
        if (got[i] !== want[i]) { if (diff < 0) diff = i; nd2++; }
      }
      if (diff >= 0) {
        // interpret as f32 and measure relative error: rounding-level drift
        // (different Tint versions → different FMA contraction) vs garbage
        const gw = new Float32Array(got.buffer, 0, want.length >> 2);
        const ww = new Float32Array(new Uint8Array(want).buffer, 0, want.length >> 2);
        let maxRel = 0, n = 0;
        for (let i = 0; i < ww.length; i++) {
          const a = gw[i], b = ww[i];
          if (!Number.isFinite(a) || !Number.isFinite(b)) { maxRel = Infinity; continue; }
          const d = Math.abs(a - b) / Math.max(1e-6, Math.abs(b));
          if (d > maxRel) maxRel = d;
          if (d > 1e-3) n++;
        }
        diverged.push({ u, diff, nd2, maxRel, nBig: n, last: lastBind.get(u) ?? -1, size: want.length });
      }
    }
    diverged.sort((a, b) => a.last - b.last);
    L(`post-state check: ${checked} buffers, ${diverged.length} diverged`);
    for (const d of diverged) {
      const o = dl[d.last];
      L(`  last-bound@#${d.last}/${dl.length} uid=${d.u} size=${d.size} nbad=${d.nd2} ` +
        `maxRel=${d.maxRel.toExponential(1)} nRel>1e-3=${d.nBig} binds=${o ? JSON.stringify(o.b.map(([n]) => n)) : ""}`);
    }
    // >1GB load-truncation probe: compare the TAIL of the biggest buffers
    // against their .bin files (Chrome fetch/writeBuffer 1GB edges)
    const bigs = [...ref].filter((u) => !tset.has(u) && (bufSizes[u] | 0) > 800e6);
    for (const u of bigs) {
      const size = bufSizes[u] | 0;
      const r = await fetch(`${dir}/b${u}.bin`, { headers: { Range: `bytes=${size - 64}-${size - 1}` } });
      if (!r.ok) continue;
      const want = new Uint8Array(await r.arrayBuffer());
      const got = await readbackRange(device, bufs.get(u), size - 64, 64);
      const eq = want.every((v, i) => got[i] === v);
      L(`  big-buffer tail probe uid=${u} size=${(size / 1e6).toFixed(0)}MB tail64=${eq ? "MATCH" : "TRUNCATED/DIFF"}`);
    }
  }
  await fetch("/result", { method: "POST", body: JSON.stringify({ nd, ms, nOK, nBAD }) });
  return { nd, ms, nOK, nBAD };
}
