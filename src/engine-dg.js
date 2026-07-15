// M2a: DG decode ENGINE on the trace-replay substrate — runs the full
// renoise/eb scheduler (port of hesper Examples/DiffusionGemmaDecode.lean
// lines ~1330-1360 + ~2005-2125) over the traced per-step dispatch stream.
// Gate: decoded France text contains [Pp]aris (near-tie wording drift OK).
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

async function fetchInto(device, buf, url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  const data = await r.arrayBuffer();
  device.queue.writeBuffer(buf, 0, data, 0, data.byteLength & ~3);
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

// hesper's LCG (Lean UInt64 semantics), bit-exact via BigInt
const fnv32 = (u8) => {
  let h = 0x811c9dc5;
  for (let i = 0; i < u8.length; i++) { h ^= u8[i]; h = Math.imul(h, 16777619); }
  return h >>> 0;
};
const M64 = (1n << 64n) - 1n;
class Rng {
  constructor(seed) { this.s = BigInt(seed) & M64; }
  next() { this.s = (this.s * 6364136223846793005n + 1442695040888963407n) & M64; return this.s; }
  u() { const v = Number(this.next() >> 11n) / 9007199254740992; return Math.max(v, 1e-12); }
  tok(vocab) { return Number(this.next() >> 33n) % vocab; }
}

export async function runEngine(dir = "dgtrace", opts = {}) {
  const S = opts.steps ?? 48;            // decodeSteps (hesper CLI default? France ran 24-step schedule, 6 eff)
  const vocabSize = 262144;
  const tmax = 0.8, tmin = 0.4, ebBound = 0.1, ebConfTh = 0.02, ebStab = 1, ebStabTol = 2;
  const scK = 8;

  const [bufSizes, opsText, vocab] = await Promise.all([
    fetch(`${dir}/buffers.json`).then((r) => r.json()),
    fetch(`${dir}/ops.jsonl`).then((r) => r.text()),
    fetch(`${dir}/vocab.json`).then((r) => r.json()),
  ]);
  const ops = opsText.split("\n").filter(Boolean)
    .map((l) => JSON.parse(l.replace(/"k":(\d+)/, '"k":"$1"')));
  // marker-sliced streams: step 0's dispatch set differs structurally (the
  // SC path only runs for step > 0), so hesper records steps 0/1/2 and the
  // engine picks the right stream per step
  const streams = {};
  { let cur = null;
    for (const o of ops) {
      if (o.t === "m") { cur = o.tag; streams[cur] = []; }
      else if (cur) streams[cur].push(o);
    } }
  const stream0 = streams["step-0"] ?? ops;
  const streamN = streams["step-2"] ?? streams["step-1"] ?? ops;
  L(`streams: step0=${stream0.length} ops, stepN=${streamN.length} ops`);

  // ---- role discovery ----------------------------------------------------
  const uidOf = {};                      // bindingName -> uid (first occurrence)
  for (const o of ops) if (o.t === "d")
    for (const [n, u] of o.b) if (!(n in uidOf)) uidOf[n] = String(u);
  // step-head dynamic writes, identified by size order (tok 4*(P+C), scTok/scProb 4*C*K, scT 4)
  const wEvents = ops.filter((o) => o.t === "w" && o.hex);
  const tokU = uidOf["token_ids"];
  const tokW = wEvents.find((o) => String(o.u) === tokU);
  const nToks = tokW.s / 4;
  const C = 256, P = nToks - C;
  const promptToks = [];
  { const b = hexToBytes(tokW.hex); const dv = new DataView(b.buffer);
    for (let i = 0; i < P; i++) promptToks.push(dv.getUint32(i * 4, true)); }
  const scW = wEvents.filter((o) => o.s === C * scK * 4).map((o) => String(o.u)); // [scTok, scProb] in order
  const scTU = scW[0], scPU = scW[1];
  const sctW = wEvents.find((o) => o.s === 4);
  const scTempU = sctW ? String(sctW.u) : null;
  const ebUU = uidOf["uin"], ebPU = uidOf["params"];
  const dynamic = new Set([tokU, scTU, scPU, scTempU, ebUU, ebPU].filter(Boolean));
  L(`engine: P=${P} C=${C} prompt=[${promptToks.slice(0, 6)}…] roles tok=${tokU} scT=${scTU} scP=${scPU} t=${scTempU} u=${ebUU} p=${ebPU}`);

  // ---- allocate + load ----------------------------------------------------
  const ref = new Set();
  for (const o of ops) {
    if (o.t === "d") for (const [, u] of o.b) ref.add(String(u));
    else if (o.t === "w" || o.t === "r") ref.add(String(o.u));
  }
  let maxBuf = 0;
  for (const u of ref) maxBuf = Math.max(maxBuf, bufSizes[u] | 0);
  const device = await initDeviceBig(maxBuf);
  const bufs = new Map();
  for (const u of ref)
    bufs.set(u, device.createBuffer({ size: Math.max(4, (bufSizes[u] + 3) & ~3),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC }));
  let loaded = 0;
  for (const u of ref) loaded += await fetchInto(device, bufs.get(u), `${dir}/b${u}.bin`).catch(() => 0);
  // cross-step state buffers: the dumps hold PRE-STEP-2 state, but hesper's
  // step 0 sees ZERO-initialized SC buffers (they are only written for
  // step > 0) — reset them or step 0 reads another step's leftovers
  for (const u of [scTU, scPU, scTempU]) if (u)
    device.queue.writeBuffer(bufs.get(u), 0, new Uint8Array((bufSizes[u] + 3) & ~3));
  L(`engine: ${ref.size} buffers, ${(loaded / 1e9).toFixed(1)}GB loaded (SC state zeroed)`);

  const kids = [...new Set(ops.filter((o) => o.t === "d").map((o) => o.k))];
  const pipes = new Map();
  await Promise.all(kids.map(async (k) => {
    const code = await fetch(`${dir}/k${k}.wgsl`).then((r) => r.text());
    const module = device.createShaderModule({ code });
    pipes.set(k, await device.createComputePipelineAsync({
      layout: "auto", compute: { module, entryPoint: "main" } }));
  }));
  L(`engine: ${pipes.size} pipelines`);
  const bgCache = new Map();
  const bgFor = (o) => {
    const sig = o.k + "|" + o.b.map(([, u]) => u).join(",");
    let bg = bgCache.get(sig);
    if (!bg) {
      bg = device.createBindGroup({ layout: pipes.get(o.k).getBindGroupLayout(0),
        entries: o.b.map(([, u], i) => ({ binding: i, resource: { buffer: bufs.get(String(u)) } })) });
      bgCache.set(sig, bg);
    }
    return bg;
  };

  // ---- scheduler state (hesper L1330-1356) --------------------------------
  const rng = new Rng(12345);
  const toks = new Uint32Array(P + C);
  toks.set(promptToks);
  for (let i = 0; i < C; i++) toks[P + i] = rng.tok(vocabSize);   // schedEB canvas randomization
  let scTok = new Uint32Array(C * scK);
  const scProb = new Float32Array(C * scK);                        // zeros in renoise mode
  let prevT = 1.0;
  let prevArgmax = null, held = 0;

  // ---- step loop -----------------------------------------------------------
  const t0 = performance.now();
  let effSteps = 0, finished = false;
  for (let step = 0; step < S && !finished; step++) {
    const tCur = tmin + (tmax - tmin) * ((S - step) / S);
    const uArr = new Float32Array(C);
    // dynamic bytes for this step (uArr drawn at the ebU write position order:
    // hesper draws them right before the ebSample dispatch — same LCG order
    // as writing here since no other draws intervene during the forward)
    for (let i = 0; i < C; i++) uArr[i] = rng.u();
    const dyn = new Map();
    dyn.set(tokU, new Uint8Array(toks.buffer.slice(0)));
    if (step > 0) {
      dyn.set(scTU, new Uint8Array(scTok.buffer.slice(0)));
      dyn.set(scPU, new Uint8Array(scProb.buffer.slice(0)));
      if (scTempU) dyn.set(scTempU, new Uint8Array(new Float32Array([prevT]).buffer));
    }
    dyn.set(ebUU, new Uint8Array(uArr.buffer.slice(0)));
    dyn.set(ebPU, new Uint8Array(new Float32Array([tCur, 0, 0, 0]).buffer));

    let enc = device.createCommandEncoder();
    let lastD = null, s0diverged = 0;
    const flush = () => { device.queue.submit([enc.finish()]); enc = device.createCommandEncoder(); };
    for (const o of (step === 0 ? stream0 : streamN)) {
      if (o.t === "w" && o.hex) {
        flush();
        const u = String(o.u);
        if (dyn.has(u)) {
          if (dyn.get(u) !== null) { device.queue.writeBuffer(bufs.get(u), 0, dyn.get(u)); dyn.set(u, null); }
          // dynamic buffer already written this step (or intentionally skipped at step 0)
        } else {
          const data = hexToBytes(o.hex);
          device.queue.writeBuffer(bufs.get(u), o.o, data, 0, data.byteLength & ~3);
        }
      } else if (o.t === "d") {
        const pass = enc.beginComputePass();
        pass.setPipeline(pipes.get(o.k));
        pass.setBindGroup(0, bgFor(o));
        pass.dispatchWorkgroups(o.g[0], o.g[1], o.g[2]);
        pass.end();
        lastD = o;
      } else if (o.t === "f") flush();
      else if (o.t === "c" && step === 0 && lastD) {
        // per-dispatch golden (hesper snapshots): find the FIRST diverging
        // dispatch of step 0 exactly
        flush();
        const got = [];
        for (const [, u] of lastD.b) {
          const size = Math.min(4096, (bufSizes[String(u)] | 0) || 4096);
          got.push(fnv32(await readbackRange(device, bufs.get(String(u)), 0, size)));
        }
        const bad = o.hs.map((w, i) => (got[i] !== w ? i : null)).filter((x) => x !== null);
        if (bad.length && s0diverged < 3) {
          s0diverged++;
          L(`  [s0-cksum] FIRST DIVERGENCE at c#${o.n} kernel=${lastD.k} bufs=${JSON.stringify(bad.map((i) => lastD.b[i][0]))} grid=${JSON.stringify(lastD.g)}`);
        }
      }
      // 'r' events: skipped — the scheduler reads its role buffers below
    }
    flush();
    await device.queue.onSubmittedWorkDone();

    // readbacks (roles)
    const rd32 = async (name, n) =>
      new Uint32Array((await readbackRange(device, bufs.get(uidOf[name]), 0, n * 4)).buffer);
    const rdF = async (name, n) =>
      new Float32Array((await readbackRange(device, bufs.get(uidOf[name]), 0, n * 4)).buffer);
    const amax = await rd32("oamax", C);
    const samp = await rd32("osamp", C);
    const hArr = await rdF("oh", C);
    const ktok = await rd32("otok", C * scK);
    if (step === 0) {
      // step-0 golden: the trace records hesper's actual readbacks as hex —
      // compare ours to split GPU-state issues from CPU-scheduler issues
      for (const o of stream0.filter((o) => o.t === "r" && o.hex)) {
        const want = hexToBytes(o.hex);
        const got = await readbackRange(device, bufs.get(String(o.u)), o.o, o.s);
        let first = -1, nb = 0;
        for (let j = 0; j < want.length; j++) if (got[j] !== want[j]) { if (first < 0) first = j; nb++; }
        const name = Object.entries(uidOf).find(([, u]) => u === String(o.u))?.[0] ?? o.u;
        if (first < 0) L(`  [s0-golden] ${name}: MATCH (${o.s}B)`);
        else {
          const gf = new Float32Array(got.buffer, 0, o.s >> 2);
          const wf = new Float32Array(new Uint8Array(want).buffer, 0, o.s >> 2);
          L(`  [s0-golden] ${name}: ${nb}/${o.s} bytes differ, first@${first}; f32[${first >> 2}] got=${gf[first >> 2]} want=${wf[first >> 2]}`);
        }
      }
    }

    // accept lowest-entropy under the MI bound (hesper L2082-2098)
    const order = [...hArr.keys()].sort((a, b) => hArr[a] - hArr[b]);
    const accepted = new Uint8Array(C);
    let cumE = 0, nAcc = 0;
    for (const pos of order) {
      if (cumE <= ebBound) { accepted[pos] = 1; nAcc++; }
      cumE += hArr[pos];
    }
    let entSum = 0;
    for (let pos = 0; pos < C; pos++) {
      entSum += hArr[pos];
      toks[P + pos] = accepted[pos] ? samp[pos] : rng.tok(vocabSize);
    }
    scTok = new Uint32Array(ktok);        // SC top-K for the next step (probs stay zero)
    let nChanged = 0;
    if (prevArgmax) for (let i = 0; i < C; i++) if (amax[i] !== prevArgmax[i]) nChanged++;
    const stable = prevArgmax !== null && nChanged <= ebStabTol;
    held = stable ? held + 1 : 0;
    prevArgmax = amax;
    prevT = tCur;
    const meanH = entSum / C;
    effSteps++;
    finished = (held >= ebStab && meanH < ebConfTh) || step + 1 >= S;
    L(`step ${step}: acc=${nAcc} chg=${prevArgmax ? nChanged : "-"} meanH=${meanH.toFixed(4)} t=${tCur.toFixed(3)}${finished ? " | STOP" : ""}`);
    if (finished) for (let i = 0; i < C; i++) toks[P + i] = amax[i];
  }
  const ms = performance.now() - t0;

  // detok (piece concat; ▁ → space); log raw ids first, stop at the SECOND
  // eos-ish token (channel markers may open with one)
  L(`ids[0:48]: ${Array.from(toks.slice(P, P + 48)).join(",")}`);
  let text = "";
  let eosSeen = 0;
  for (let i = 0; i < C; i++) {
    const id = toks[P + i];
    if (id === 106 || id === 1) { if (++eosSeen >= 2) break; continue; }
    text += (vocab[id] ?? "<unk>").replace(/▁/g, " ");
  }
  const pass = /[Pp]aris/.test(text);
  L(`ENGINE ${pass ? "PASS" : "FAIL"}: ${effSteps} steps, ${(ms / effSteps).toFixed(0)}ms/step | text: ${text.slice(0, 200)}`);
  await fetch("/result", { method: "POST", body: JSON.stringify({ pass, effSteps, msPerStep: ms / effSteps, text: text.slice(0, 400) }) });
  return { pass, effSteps, ms, text };
}
