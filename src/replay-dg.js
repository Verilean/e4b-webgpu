// DG->JS port M1: thin replayer for a hesper JSTrace package (docs/JS_REPLAY_TRACE.md
// in the hesper repo). Loads weights (GGUF ranges + derived .bin dumps), replays the
// step-0 dispatch stream, and gates on BIT-EQUALITY of the recorded readbacks.
import { openGGUF, tensorBytes } from "./gguf.js";

const L = (m) => { console.log(m); return fetch("/log", { method: "POST", body: String(m) }).catch(() => {}); };

const fnv32 = (u8) => {
  let h = 0x811c9dc5;
  for (let i = 0; i < u8.length; i++) { h ^= u8[i]; h = Math.imul(h, 16777619); }
  return h >>> 0;
};

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

  // 2. contents: .bin dumps are AUTHORITATIVE (exact bytes of hesper's
  // buffers); GGUF ranges only as fallback for uids without a dump
  const gguf = await openGGUF(ggufUrl);
  const tset = new Set(Object.keys(tensors));
  let loaded = 0, viaGguf = 0;
  for (const u of ref) {
    const n = await fetchInto(device, bufs.get(u), `${dir}/b${u}.bin`).catch(() => 0);
    if (n > 0) { loaded += n; continue; }
    if (tset.has(u)) {
      const t = gguf.tensors[tensors[u]];
      if (!t) { L(`WARN tensor ${tensors[u]} not in GGUF`); continue; }
      const off = gguf.dataStart + t.offset;
      const r = await fetch(ggufUrl, { headers: { Range: `bytes=${off}-${off + tensorBytes(t) - 1}` } });
      const data = await r.arrayBuffer();
      device.queue.writeBuffer(bufs.get(u), 0, data, 0, data.byteLength & ~3);
      viaGguf += data.byteLength;
    }
  }
  L(`loaded: ${(loaded / 1e9).toFixed(1)}GB from dumps + ${(viaGguf / 1e9).toFixed(1)}GB via GGUF (t+${((performance.now() - t00) / 1000).toFixed(0)}s)`);

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
  let nd = 0, nOK = 0, nBAD = 0, ckOK = 0, ckBAD = 0;
  let ckULP = 0, ckMOD = 0, ckWorst = 0, ckWorstAt = -1, ckFirstFlip = -1;
  let lastD = null;
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
      // ONE PASS PER DISPATCH — matching hesper's bridge.cpp exactly. Dawn on
      // Metal is known (our own bug ledger) to drop inter-dispatch barriers
      // at scale inside a large encoder; packing hundreds of dispatches into
      // one pass reproduced a tail-only divergence here.
      pass = enc.beginComputePass();
      pass.setPipeline(pipes.get(o.k));
      pass.setBindGroup(0, bgFor(o));
      pass.dispatchWorkgroups(o.g[0], o.g[1], o.g[2]);
      pass.end(); pass = null;
      lastD = o;
      nd++;
    } else if (o.t === "c") {
      // per-dispatch checksum stream (DG_TRACE_JS_CKSUM): mirror hesper's
      // XOR of per-buffer FNV32 over each bound buffer's first 4KB
      flushSubmit();
      const got = [];
      for (const [, u] of lastD.b) {
        const size = Math.min(4096, (bufSizes[String(u)] | 0) || 4096);
        got.push(fnv32(await readbackRange(device, bufs.get(String(u)), 0, size)));
      }
      // tolerance gate (M1): classify each mismatch by snapshot maxRel —
      // ULP-level drift (compiler FMA differences) is EXPECTED across
      // Dawn/Tint versions; the interesting curve is where drift first
      // amplifies into a near-tie flip (>1e-2 or integer content changes)
      const bad = o.hs.map((w, i) => (got[i] !== w ? i : null)).filter((x) => x !== null);
      if (bad.length === 0) { ckOK++; }
      else {
        let worst = 0;
        for (const i of bad) {
          const r = await fetch(`${dir}/c${o.n}_${i}.bin`);
          if (!r.ok) { worst = Infinity; continue; }
          const want = new Uint8Array(await r.arrayBuffer());
          const [, u] = lastD.b[i];
          const size = Math.min(4096, (bufSizes[String(u)] | 0) || 4096);
          const g = await readbackRange(device, bufs.get(String(u)), 0, size);
          const gf = new Float32Array(g.buffer, 0, size >> 2);
          const wf = new Float32Array(new Uint8Array(want).buffer, 0, size >> 2);
          // RMS-normalized max deviation: plain per-element relative error
          // over-penalizes near-zero elements (ULP noise on 1e-6 reads as
          // 1e-2 "error"); normalizing by the buffer's own RMS classifies
          // by how much the SIGNAL moved
          let rms = 0, n2 = 0, maxAbs = 0;
          for (let j = 0; j < wf.length; j++) {
            if (!Number.isFinite(wf[j]) || !Number.isFinite(gf[j])) continue;
            rms += wf[j] * wf[j]; n2++;
            const d2 = Math.abs(gf[j] - wf[j]);
            if (d2 > maxAbs) maxAbs = d2;
          }
          rms = Math.sqrt(rms / Math.max(1, n2));
          const relRMS = maxAbs / Math.max(1e-9, rms);
          if (relRMS > worst) worst = relRMS;
        }
        if (worst <= 1e-5) ckULP++;
        else if (worst <= 1e-2) ckMOD++;
        else {
          ckBAD++;
          if (ckFirstFlip < 0) ckFirstFlip = o.n;
          if (ckBAD <= 3)
            L(`FLIP at dispatch #${o.n} kernel=${lastD.k} maxRel=${worst.toExponential(1)} ` +
              `binds=${JSON.stringify(lastD.b.map(([n2]) => n2))}`);
        }
        if (worst > ckWorst) { ckWorst = worst; ckWorstAt = o.n; }
      }
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
  L(`replay done: ${nd} dispatches in ${ms.toFixed(0)}ms | readback gate: ${nOK} OK, ${nBAD} MISMATCH` +
    (ckOK + ckULP + ckMOD + ckBAD > 0
      ? ` | cksum: ${ckOK} exact, ${ckULP} ULP(≤1e-5), ${ckMOD} mod(≤1e-2), ${ckBAD} FLIP; ` +
        `worst ${ckWorst.toExponential(1)}@#${ckWorstAt}, first flip @#${ckFirstFlip}`
      : ""));

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
