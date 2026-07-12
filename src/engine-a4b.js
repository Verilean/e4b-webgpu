// Gemma-4 26B-A4B (MoE) decoder on WebGPU — GGUF q4_0 direct load.
// Semantics verified against transformers 5.13 modeling_gemma4.py and the
// llama.cpp gemma4 graph (refs/llama.cpp-diffusiongemma/src/models/gemma4-common.h):
//   layer: inNorm→attn→postAttnNorm→+res;
//          dense: ffnNorm→gate/up(2112)→geglu→down→postFfw1  (parallel with)
//          moe:   router(raw residual)→top8; preFfw2→gate_up_exps→geglu→down_exps
//                 →Σ topkW[k](incl per-expert scale)→postFfw2;
//          combined=mlp+moe → postFfwNorm → +res → ×layer_output_scale
//   full-attn layers: k_eq_v (no v_proj; V = v_norm(k_proj out), no rope), 2 kv
//   heads × 512, proportional rope 64 angles θ=1M; sliding: 8 kv × 256, θ=10k.
//   lm_head = token_embd (Q6_K, tied), final softcap 30.
import { openGGUF } from "./gguf.js";
import { initDevice, upload, alloc, readback, Kernels } from "./gpu.js";

const L = (m) => fetch("/log", { method: "POST", body: String(m) }).catch(() => {});
const MAXSEQ = 640;

// ---- repackers (q4_0 / q6_K planes; 18B and 210B blocks are not word-aligned) ----
function repackQ40(buf) {
  const nb = buf.byteLength / 18;
  const src = new Uint8Array(buf);
  const nib = new Uint8Array(nb * 16);
  const sc = new Uint16Array(nb + (nb & 1));         // pad to word
  for (let b = 0; b < nb; b++) {
    sc[b] = src[b * 18] | (src[b * 18 + 1] << 8);
    nib.set(src.subarray(b * 18 + 2, b * 18 + 18), b * 16);
  }
  return { nib, sc };
}
// Tile-transposed Q6_K planes (T=128 rows/tile): unit u of row o lives at
// (tile*unitsPerRow + u)*128 + (o%128) — coalesced across a 128-thread WG.
// The embed row-gather uses the same indexing. d stored as f32.
function repackQ6K(buf, rowLen, T = 128) {
  const bpr = rowLen / 256;                       // blocks per row
  const rows = buf.byteLength / 210 / bpr;
  const src = new Uint8Array(buf);
  const dv = new DataView(buf);
  const qlU = bpr * 32, qhU = bpr * 16, scU = bpr * 4;   // u32 units per row
  const ql = new Uint32Array(rows * qlU);
  const qh = new Uint32Array(rows * qhU);
  const sc = new Uint32Array(rows * scU);
  const d = new Float32Array(rows * bpr);
  const f16 = (h) => {                             // f16 bits → f32
    const s2 = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e === 0) return s2 * m * 2 ** -24;
    if (e === 31) return m ? NaN : s2 * Infinity;
    return s2 * (1 + m / 1024) * 2 ** (e - 15);
  };
  for (let o = 0; o < rows; o++) {
    const tile = (o / T) | 0, t = o % T;
    for (let b = 0; b < bpr; b++) {
      const off = (o * bpr + b) * 210;
      for (let u = 0; u < 32; u++)
        ql[(tile * qlU + b * 32 + u) * T + t] = dv.getUint32(off + u * 4, true);
      for (let u = 0; u < 16; u++)
        qh[(tile * qhU + b * 16 + u) * T + t] = dv.getUint32(off + 128 + u * 4, true);
      for (let u = 0; u < 4; u++)
        sc[(tile * scU + b * 4 + u) * T + t] = dv.getUint32(off + 192 + u * 4, true);
      d[(tile * bpr + b) * T + t] = f16(dv.getUint16(off + 208, true));
    }
  }
  return { ql, qh, sc, d };
}

export async function loadEngineA4B(ggufUrl = "model-a4b/gemma-4-26B_q4_0-it.gguf") {
  const t0 = performance.now();
  const st = await openGGUF(ggufUrl);
  const device = await initDevice();
  const K = new Kernels(device);
  const kvOf = (k) => st.kv[k];

  const C = {
    hidden: kvOf("gemma4.embedding_length"),           // 2816
    layers: kvOf("gemma4.block_count"),                // 30
    qHeads: kvOf("gemma4.attention.head_count"),       // 16
    inter: kvOf("gemma4.feed_forward_length"),         // 2112
    expInter: kvOf("gemma4.expert_feed_forward_length"), // 704
    nExperts: kvOf("gemma4.expert_count"),             // 128
    topK: kvOf("gemma4.expert_used_count"),            // 8
    vocab: 262144,
    window: kvOf("gemma4.attention.sliding_window"),   // 1024
    eps: kvOf("gemma4.attention.layer_norm_rms_epsilon"),
    softcap: kvOf("gemma4.final_logit_softcapping"),   // 30
    hdFull: kvOf("gemma4.attention.key_length"),       // 512
    hdSwa: kvOf("gemma4.attention.key_length_swa"),    // 256
    kvHeadsPerLayer: kvOf("gemma4.attention.head_count_kv"),  // [30]
    swaPattern: kvOf("gemma4.attention.sliding_window_pattern"), // [30] 1=swa
  };

  // ---- upload helpers ----
  const f32buf = async (name) => upload(device, new Float32Array((await st.fetch(name)).buf));
  const scalarOf = async (name) => new Float32Array((await st.fetch(name)).buf)[0];
  async function q40(name) {
    const t = await st.fetch(name);
    const { nib, sc } = repackQ40(t.buf);
    return { dims: t.dims, nibBuf: upload(device, nib), scBuf: upload(device, sc) };
  }
  async function q40cat(names) {                 // concat rows of same-IN linears
    const parts = [];
    for (const n of names) parts.push(repackQ40((await st.fetch(n)).buf));
    const nib = new Uint8Array(parts.reduce((a, p2) => a + p2.nib.length, 0));
    const nBlocks = parts.reduce((a, p2) => a + p2.nib.length / 16, 0);
    const sc = new Uint16Array((nBlocks + 1) & ~1);      // even count → 4B multiple
    let no = 0, so = 0;
    for (const p2 of parts) {
      nib.set(p2.nib, no); no += p2.nib.length;
      sc.set(p2.sc.subarray(0, p2.nib.length / 16), so); so += p2.nib.length / 16;
    }
    return { nibBuf: upload(device, nib), scBuf: upload(device, sc) };
  }

  L("loading weights (GGUF q4_0 → repacked planes)…");
  const emb = repackQ6K((await st.fetch("token_embd.weight")).buf, C.hidden);
  const model = {
    embQl: upload(device, emb.ql), embQh: upload(device, emb.qh),
    embSc: upload(device, emb.sc), embD: upload(device, emb.d),
    outNorm: await f32buf("output_norm.weight"),
  };

  const layers = [];
  for (let i = 0; i < C.layers; i++) {
    const p = `blk.${i}.`;
    const isSliding = C.swaPattern[i] === true || C.swaPattern[i] === 1;
    const headDim = isSliding ? C.hdSwa : C.hdFull;
    const kvHeads = C.kvHeadsPerLayer[i];
    const l = {
      i, isSliding, headDim, kvHeads,
      keqv: !isSliding,                                  // full layers: V = K proj
      attnNorm: await f32buf(p + "attn_norm.weight"),
      postAttnNorm: await f32buf(p + "post_attention_norm.weight"),
      qNorm: await f32buf(p + "attn_q_norm.weight"),
      kNorm: await f32buf(p + "attn_k_norm.weight"),
      q: await q40(p + "attn_q.weight"),
      k: await q40(p + "attn_k.weight"),
      o: await q40(p + "attn_output.weight"),
      ffnNorm: await f32buf(p + "ffn_norm.weight"),
      gate: await q40(p + "ffn_gate.weight"),
      up: await q40(p + "ffn_up.weight"),
      down: await q40(p + "ffn_down.weight"),
      postFfw: await f32buf(p + "post_ffw_norm.weight"),
      postFfw1: await f32buf(p + "post_ffw_norm_1.weight"),
      postFfw2: await f32buf(p + "post_ffw_norm_2.weight"),
      preFfw2: await f32buf(p + "pre_ffw_norm_2.weight"),
      layerScalarVal: await scalarOf(p + "layer_output_scale.weight"),
      routerW: await f32buf(p + "ffn_gate_inp.weight"),
      routerS: await (async () => {
        const v = new Float32Array((await st.fetch(p + "ffn_gate_inp.scale")).buf);
        const m = 1 / Math.sqrt(C.hidden);            // fold h^-0.5 into the scale
        for (let j = 0; j < v.length; j++) v[j] *= m;
        return upload(device, v);
      })(),
      pes: await f32buf(p + "ffn_down_exps.scale"),
      guExps: await q40(p + "ffn_gate_up_exps.weight"),
      downExps: await q40(p + "ffn_down_exps.weight"),
      kCache: alloc(device, MAXSEQ * kvHeads * headDim * 2),   // f16
      vCache: alloc(device, MAXSEQ * kvHeads * headDim * 2),
    };
    if (!l.keqv) l.v = await q40(p + "attn_v.weight");
    l.qOut = l.q.dims[1];                                // rows
    l.kvOut = l.k.dims[1];
    l.qkvCat = await q40cat(l.keqv
      ? [p + "attn_q.weight", p + "attn_k.weight"]
      : [p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"]);
    l.qkvRows = l.qOut + l.kvOut * (l.keqv ? 1 : 2);
    l.guCat = await q40cat([p + "ffn_gate.weight", p + "ffn_up.weight"]);
    layers.push(l);
    if (i % 5 === 0) L(`  layer ${i}/${C.layers} (${((performance.now() - t0) / 1000).toFixed(0)}s)`);
  }
  L(`weights on GPU in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

  // ---- activation buffers ----
  const KEXP = C.topK;
  const bgCache = new Map();
  const A = {
    params: alloc(device, 16),
    paramsRing: Array.from({ length: 16 }, () => alloc(device, 16)),
    hidden: alloc(device, C.hidden * 4),
    normed: alloc(device, C.hidden * 2),        // f16
    moeIn: alloc(device, C.hidden * 2),          // f16 (pre-ffw-2 normed)
    tmp: alloc(device, C.hidden * 4),
    mlpOut: alloc(device, C.hidden * 4),
    moeOut: alloc(device, C.hidden * 4),
    qkv: alloc(device, (16 * C.hdFull + 2 * 2 * C.hdFull) * 4),   // worst case
    attnOut: alloc(device, 16 * C.hdFull * 2),   // f16
    gu: alloc(device, 2 * C.inter * 4),
    geglu: alloc(device, C.inter * 2),           // f16
    guSlots: alloc(device, KEXP * 2 * C.expInter * 4),
    gegluSlots: alloc(device, KEXP * C.expInter * 2),   // f16
    downSlots: alloc(device, KEXP * C.hidden * 4),
    routerIn: alloc(device, C.hidden * 4),
    routerScores: alloc(device, 128 * 4),
    routerCtr: alloc(device, 16),
    topkIdx: alloc(device, KEXP * 4),
    topkW: alloc(device, KEXP * 4),
    logits: alloc(device, C.vocab * 4),
    amax: alloc(device, 16),
    amaxPart: alloc(device, 256 * 8),
    tokRing: alloc(device, 1024 * 8),
    onesE: upload(device, new Float32Array(128).fill(1)),
    srqZero: upload(device, new Float32Array([0, 0]), GPUBufferUsage.UNIFORM),
    dummySums: alloc(device, 16),
    dummySumI: alloc(device, 16),
  };

  // ---- pipelines (re-callable: kernel hot-reload without reloading weights) ----
  const kern = {};
  async function rebuildPipelines(Knew) {
    const K = Knew ?? new Kernels(device);
    const mv = (IN, OUT, opts = {}) => K.pipeline("q40mv", {
      IN, OUT, EXPERT: opts.expert ? 1 : 0, XSLOT: opts.xslot ? 1 : 0, XF16: opts.xf16 ? 1 : 0, WG: opts.wg ?? 32 });
    Object.assign(kern, {
    embed: await K.pipeline("q6k", { N: C.hidden, OUT: C.hidden, MODE: 0, TILE: 128,
      MULT: Math.sqrt(C.hidden).toFixed(8), SOFTCAP: "0.0" }),
    lmHead: await K.pipeline("q6k", { N: C.hidden, OUT: C.vocab, MODE: 1, TILE: 128,
      MULT: "1.0", SOFTCAP: C.softcap.toFixed(1) }),
    rms: await K.pipeline("rmsnorm", { DIM: C.hidden, EPS: C.eps, WITH_SCALE: 1, SUMOUT: 0, F16OUT: 1, WG: 256 }),
    rmsF32: await K.pipeline("rmsnorm", { DIM: C.hidden, EPS: C.eps, WITH_SCALE: 1, SUMOUT: 0, F16OUT: 0, WG: 256 }),
    routerTop: await K.pipeline("routertop", { H: C.hidden, E: C.nExperts, K: KEXP, WG: 64 }),
    gegluDense: await K.pipeline("geglumul", { N: C.inter, WG: 256 }),
    gegluSlots: await K.pipeline("geglusl", { FF: C.expInter, K: KEXP, WG: 256 }),
    rms3: await K.pipeline("rms3", { DIM: C.hidden, EPS: C.eps, WG: 256 }),
    rmsacc3: await K.pipeline("rmsacc3", { DIM: C.hidden, EPS: C.eps, WG: 256 }),
    accH: await K.pipeline("acc", { N: C.hidden, WG: 256 }),
    argmax0: await K.pipeline("argmax2", { N: C.vocab, PARTS: 256, STAGE: 0, WG: 256 }),
    argmax1: await K.pipeline("argmax2", { N: C.vocab, PARTS: 256, STAGE: 1, WG: 256 }),
    feedTok: await K.pipeline("feedtok", {}),
    });
    for (const l of layers) {
    const ra = l.isSliding ? l.headDim / 2 : Math.floor(0.25 * l.headDim / 2);
    const theta = l.isSliding ? "10000.0" : "1000000.0";
    l.qkvMv = await K.pipeline("q40mv", { IN: C.hidden, OUT: l.qkvRows, EXPERT: 0, XSLOT: 0, XF16: 1, WG: 64 });
    l.oMv = await K.pipeline("q40mv", { IN: l.qOut, OUT: C.hidden, EXPERT: 0, XSLOT: 0, XF16: 1, WG: 32 });
    l.guMv = await K.pipeline("q40gu", { IN: C.hidden, FF: C.inter, E: C.nExperts, K: KEXP, EXPERT: 0 });
    l.downMv = await K.pipeline("q40mv", { IN: C.inter, OUT: C.hidden, EXPERT: 0, XSLOT: 0, XF16: 1, WG: 32 });
    l.guExpsMv = await K.pipeline("q40gu", { IN: C.hidden, FF: C.expInter, E: C.nExperts, K: KEXP, EXPERT: 1 });
    l.downExpsMv = await K.pipeline("q40moedown", { IN: C.expInter, OUT: C.hidden, K: KEXP });
    l.headprep = await K.pipeline("headprep", { QH: C.qHeads, KVH: l.kvHeads,
      HEAD_DIM: l.headDim, ROPE_ANGLES: ra, THETA: theta, EPS: C.eps,
      KEQV: l.keqv ? 1 : 0, WG: 128 });
    l.attn = await K.pipeline("attnf32", { Q_HEADS: C.qHeads, KV_HEADS: l.kvHeads,
      HEAD_DIM: l.headDim, MAXSEQ, WINDOW: l.isSliding ? C.window : 0, DT: 1, WG: 256 });
    l.tail = await K.pipeline("a4btail", { H: C.hidden, K: KEXP, EPS: C.eps,
      MUL: l.layerScalarVal.toPrecision(9), NEXT: l.i + 1 < C.layers ? 1 : 0, WG: 256 });
      l.rmsaccOne = await K.pipeline("rmsacc", { DIM: C.hidden, EPS: C.eps, MUL: "1.0", WG: 256 });
    }
    bgCache.clear();
  }
  await rebuildPipelines(K);
  L("pipelines built");

  function bind(kernEntry, buffers) {
    const key = kernEntry.pipeline.label + "|" + buffers.map((b) => {
      const buf = b.buffer ?? b;
      return (buf.__id ?? (buf.__id = Math.random())) + ":" + (b.offset ?? 0);
    }).join(",");
    let bg = bgCache.get(key);
    if (!bg) {
      bg = device.createBindGroup({
        layout: kernEntry.pipeline.getBindGroupLayout(0),
        entries: buffers.map((b, i) => ({ binding: i,
          resource: b.buffer ? { buffer: b.buffer, offset: b.offset, size: b.size } : { buffer: b } })),
      });
      bgCache.set(key, bg);
    }
    return bg;
  }
  const mkRun = (pass) => (k, bufs, groups) => {
    pass.setPipeline(k.pipeline); pass.setBindGroup(0, bind(k, bufs));
    pass.dispatchWorkgroups(...(Array.isArray(groups) ? groups : [groups]));
  };
  const wg = (n, w) => Math.ceil(n / w);
  const wg2 = (rows) => rows <= 32768 ? [rows] : [32768, Math.ceil(rows / 32768)];

  function encodeLayer(run, l, P) {
    // attention (layer 0 norms here; later layers get A.normed from the prev tail)
    if (l.i === 0) run(kern.rms, [A.hidden, l.attnNorm, A.tmp /*unused f32 y*/, A.dummySums, A.normed], 1);
    run(l.qkvMv, [A.tmp /*unused f32 x*/, l.qkvCat.nibBuf, l.qkvCat.scBuf, A.topkIdx, A.qkv, A.normed], wg(l.qkvRows, 4));
    run(l.headprep, [A.qkv, l.qNorm, l.kNorm, P, l.kCache, l.vCache, A.dummySumI],
        C.qHeads + 2 * l.kvHeads);
    run(l.attn, [A.qkv, l.kCache, l.vCache, P, A.attnOut], [C.qHeads, 1]);
    run(l.oMv, [A.mlpOut /*unused f32 x*/, l.o.nibBuf, l.o.scBuf, A.topkIdx, A.tmp, A.attnOut], wg(C.hidden, 2));
    // fused: postAttn norm + residual + the ffn/router/pre-ffw-2 triple norm
    run(kern.rmsacc3, [A.tmp, l.postAttnNorm, l.ffnNorm, l.routerS, l.preFfw2,
        A.hidden, A.normed, A.routerIn, A.moeIn], 1);
    run(l.guMv, [A.normed, l.guCat.nibBuf, l.guCat.scBuf, A.topkIdx,
        A.mlpOut /*dead f32 slot*/, A.geglu], wg(C.inter, 4));
    run(l.downMv, [A.mlpOut /*unused f32 x*/, l.down.nibBuf, l.down.scBuf, A.topkIdx, A.tmp, A.geglu], wg(C.hidden, 2));
    run(kern.routerTop, [A.routerIn, l.routerW, l.pes, A.routerScores, A.routerCtr,
        A.topkIdx, A.topkW], C.nExperts);
    run(l.guExpsMv, [A.moeIn, l.guExps.nibBuf, l.guExps.scBuf, A.topkIdx,
        A.mlpOut /*dead f32 slot*/, A.gegluSlots], [wg(C.expInter, 4), 1, KEXP]);
    run(l.downExpsMv, [A.gegluSlots, l.downExps.nibBuf, l.downExps.scBuf, A.topkIdx, A.topkW,
        A.moeOut], wg(C.hidden, 4));
    // fused tail: postFfw1/2 + add + post norm + residual + scalar (+ next norm)
    const nx = layers[l.i + 1];
    run(l.tail, [A.tmp, l.postFfw1, A.moeOut, l.postFfw2, l.postFfw, A.hidden,
        nx ? nx.attnNorm : l.attnNorm, A.normed], 1);
  }

  function encodeForward(run, P = A.params) {
    run(kern.embed, [model.embQl, model.embQh, model.embSc, model.embD, P,
        A.normed /*unused x*/, A.hidden], 1);
    for (const l of layers) encodeLayer(run, l, P);
    run(kern.rmsF32, [A.hidden, model.outNorm, A.tmp, A.dummySums, A.normed], 1);
    run(kern.lmHead, [model.embQl, model.embQh, model.embSc, model.embD, P,
        A.tmp, A.logits], wg2(C.vocab / 128));
    run(kern.argmax0, [A.logits, A.amaxPart, A.amax], 256);
    run(kern.argmax1, [A.logits, A.amaxPart, A.amax], 1);
  }

  async function step(token, pos) {
    device.queue.writeBuffer(A.params, 0, new Uint32Array([pos, pos + 1, token, 0]));
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    encodeForward(mkRun(pass));
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  // per-dispatch GPU budget (pass-per-dispatch timestamps; RANKING only)
  async function profileStep(token, pos) {
    if (!device.features.has("timestamp-query")) return null;
    device.queue.writeBuffer(A.params, 0, new Uint32Array([pos, pos + 1, token, 0]));
    const qs = device.createQuerySet({ type: "timestamp", count: 4096 });
    const enc = device.createCommandEncoder();
    const labels = [];
    const run = (k, bufs, groups) => {
      const i = labels.length;
      const pass = enc.beginComputePass({ timestampWrites: {
        querySet: qs, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } });
      pass.setPipeline(k.pipeline);
      pass.setBindGroup(0, bind(k, bufs));
      pass.dispatchWorkgroups(...(Array.isArray(groups) ? groups : [groups]));
      pass.end();
      labels.push(k.pipeline.label);
    };
    encodeForward(run);
    const qbuf = device.createBuffer({ size: labels.length * 16,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    enc.resolveQuerySet(qs, 0, labels.length * 2, qbuf, 0);
    device.queue.submit([enc.finish()]);
    const t = new BigUint64Array(await readback(device, qbuf, labels.length * 16));
    const agg = new Map();
    labels.forEach((lb, i) => {
      const us = Number(t[2 * i + 1] - t[2 * i]) / 1000;
      const e = agg.get(lb) ?? { us: 0, n: 0 };
      e.us += us; e.n += 1;
      agg.set(lb, e);
    });
    qs.destroy(); qbuf.destroy();
    return agg;
  }

  async function argmaxFast() {
    const u = new Uint32Array(await readback(device, A.amax, 8));
    return u[0];
  }

  function decodeChunk(startPos, count, ringBase) {
    // one encoder/submit per up to 8 tokens; per-token params come from a ring
    // (positions are known up front; the token id flows GPU-side via feedTok)
    for (let c = 0; c < count; c += 16) {
      const n = Math.min(16, count - c);
      for (let i = 0; i < n; i++) {
        const pos = startPos + c + i;
        device.queue.writeBuffer(A.paramsRing[i], 0, new Uint32Array([pos, pos + 1]));
      }
      const enc = device.createCommandEncoder();
      for (let i = 0; i < n; i++) {
        const P = A.paramsRing[i];
        const pass = enc.beginComputePass();
        const run = mkRun(pass);
        run(kern.feedTok, [A.amax, P], 1);
        encodeForward(run, P);
        pass.end();
        enc.copyBufferToBuffer(A.amax, 0, A.tokRing, (ringBase + c + i) * 8, 8);
      }
      device.queue.submit([enc.finish()]);
    }
  }

  async function generateFast(inputIds, maxNew, eosIds = new Set([106, 1])) {
    let pos = 0;
    for (const t of inputIds) await step(t, pos++);
    const g0 = await argmaxFast();
    const out = [g0];
    if (eosIds.has(g0)) return out;
    let done = 1;
    while (done < maxNew) {
      const n = Math.min(8, maxNew - done);
      decodeChunk(pos, n, done - 1);
      pos += n;
      const ring = new Uint32Array(await readback(device, A.tokRing, (done - 1 + n) * 8));
      let stop = false;
      for (let k = done - 1; k < done - 1 + n; k++) {
        const t = ring[k * 2];
        out.push(t);
        if (eosIds.has(t)) { stop = true; break; }
      }
      done += n;
      if (stop) break;
    }
    return out.slice(0, maxNew);
  }

  async function readHidden() {
    return new Float32Array(await readback(device, A.hidden, C.hidden * 4));
  }

  return { device, C, layers, model, A, kern, step, decodeChunk, generateFast,
           argmaxFast, readHidden, encodeForward, encodeLayerPub: encodeLayer, mkRun, bindPub: bind, rebuildPipelines, profileStep };
}
