// Gemma-4 E4B QAT decoder on WebGPU — bring-up build (f32, naive kernels, correctness first).
// Arch: NOTES/arch-spec.md · weight format: NOTES/qat-format.md.
import { openSafetensors, bf16ToF32 } from "./loader.js";
import { initDevice, upload, alloc, readback, Kernels } from "./gpu.js";

const L = (m) => fetch("/log", { method: "POST", body: String(m) }).catch(() => {});
const MAXSEQ = 640;
const AB_NOFUSE = new URLSearchParams(location.search).get("nofuse") === "1";

export async function loadEngine(modelUrl = "model/model.safetensors", configUrl = "model/config.json") {
  const t0 = performance.now();
  const cfg = (await (await fetch(configUrl)).json()).text_config;
  const C = {
    hidden: cfg.hidden_size, layers: cfg.num_hidden_layers,
    qHeads: cfg.num_attention_heads, kvHeads: cfg.num_key_value_heads,
    headDim: cfg.head_dim, globalHeadDim: cfg.global_head_dim ?? 512,
    vocab: cfg.vocab_size, inter: cfg.intermediate_size,
    window: cfg.sliding_window, kvShared: cfg.num_kv_shared_layers,
    pleDim: cfg.hidden_size_per_layer_input, eps: cfg.rms_norm_eps ?? 1e-6,
    layerTypes: cfg.layer_types, softcap: cfg.final_logit_softcapping ?? 30.0,
  };
  const PLE_TOTAL = C.layers * C.pleDim;
  const st = await openSafetensors(modelUrl);
  const device = await initDevice();
  const K = new Kernels(device);
  const P = "model.language_model.";

  const f32buf = (t) => upload(device, t.dtype === "BF16" ? bf16ToF32(t.buf) : new Float32Array(t.buf));
  const scalar = (t) => (t.dtype === "BF16" ? bf16ToF32(t.buf) : new Float32Array(t.buf))[0];

  async function linearConcat(prefixes) {
    // concat rows of several int4 linears (same IN) into one grouped matvec
    const parts = [];
    for (const pf of prefixes) {
      parts.push({
        w: await st.fetch(pf + ".weight"),
        ws: new Float32Array((await st.fetch(pf + ".weight_scale")).buf),
        inS: scalar(await st.fetch(pf + ".input_activation_scale")),
        outS: scalar(await st.fetch(pf + ".output_activation_scale")),
      });
    }
    const wBytes = parts.reduce((a, p) => a + p.w.buf.byteLength, 0);
    const wAll = new Uint8Array(wBytes);
    const wsAll = new Float32Array(parts.reduce((a, p) => a + p.ws.length, 0));
    const srqs = new Float32Array(parts.length * 2);
    const bounds = [];
    let wo = 0, so = 0, rows = 0;
    parts.forEach((p, i) => {
      wAll.set(new Uint8Array(p.w.buf), wo); wo += p.w.buf.byteLength;
      wsAll.set(p.ws, so); so += p.ws.length;
      rows += p.w.shape[0]; bounds.push(rows);
      srqs[i * 2] = p.inS; srqs[i * 2 + 1] = p.outS;
    });
    return {
      out: rows, bounds, inScales: parts.map((p) => p.inS), outScales: parts.map((p) => p.outS),
      wBuf: upload(device, wAll), wsBuf: upload(device, wsAll),
      srqsBuf: upload(device, srqs),
    };
  }

  async function linear(prefix) {
    const w = await st.fetch(prefix + ".weight");
    const inS = scalar(await st.fetch(prefix + ".input_activation_scale"));
    const outS = scalar(await st.fetch(prefix + ".output_activation_scale"));
    return {
      out: w.shape[0], inS, outS,
      wBuf: upload(device, new Uint8Array(w.buf)),
      wsBuf: upload(device, new Float32Array((await st.fetch(prefix + ".weight_scale")).buf)),
      srqBuf: upload(device, new Float32Array([inS, outS]), GPUBufferUsage.UNIFORM),
    };
  }

  L("loading weights…");
  const layers = [];
  const firstShared = C.layers - C.kvShared;
  for (let i = 0; i < C.layers; i++) {
    const p = `${P}layers.${i}.`;
    const isSliding = C.layerTypes[i] === "sliding_attention";
    const headDim = isSliding ? C.headDim : C.globalHeadDim;
    const isShared = i >= firstShared;
    const l = {
      i, isSliding, headDim, isShared,
      inNorm: f32buf(await st.fetch(p + "input_layernorm.weight")),
      postAttnNorm: f32buf(await st.fetch(p + "post_attention_layernorm.weight")),
      preFfnNorm: f32buf(await st.fetch(p + "pre_feedforward_layernorm.weight")),
      postFfnNorm: f32buf(await st.fetch(p + "post_feedforward_layernorm.weight")),
      postPleNorm: f32buf(await st.fetch(p + "post_per_layer_input_norm.weight")),
      layerScalar: f32buf(await st.fetch(p + "layer_scalar")),
      layerScalarVal: scalar(await st.fetch(p + "layer_scalar")),
      qNorm: f32buf(await st.fetch(p + "self_attn.q_norm.weight")),
      qkv: await linearConcat(isShared
        ? [p + "self_attn.q_proj"]
        : [p + "self_attn.q_proj", p + "self_attn.k_proj", p + "self_attn.v_proj"]),
      qOut: C.qHeads * headDim, kvOut: C.kvHeads * headDim,
      o: await linear(p + "self_attn.o_proj"),
      gu: await linearConcat([p + "mlp.gate_proj", p + "mlp.up_proj"]),
      down: await linear(p + "mlp.down_proj"),
      pleGate: await linear(p + "per_layer_input_gate"),
      pleProj: await linear(p + "per_layer_projection"),
    };
    if (!isShared) {
      l.kNorm = f32buf(await st.fetch(p + "self_attn.k_norm.weight"));
      l.kCache = alloc(device, MAXSEQ * C.kvHeads * headDim * 4);
      l.vCache = alloc(device, MAXSEQ * C.kvHeads * headDim * 4);
    }
    l.qkvScales = upload(device, new Float32Array(l.qkv.inScales));
    l.gateUpScales = upload(device, new Float32Array(l.gu.inScales));
    l.pleGateScales = upload(device, new Float32Array([l.pleGate.inS]));
    l.pleSrqs = upload(device, new Float32Array([l.pleGate.inS, l.pleGate.outS, l.pleProj.inS]));
    l.guSrqs = upload(device, new Float32Array([...l.gu.inScales.flatMap((x, i) => [x, l.gu.outScales[i]]), l.down.inS]));
    layers.push(l);
    if (i % 7 === 0) L(`  layer ${i}/${C.layers}`);
  }
  const shareSrc = {};
  for (let j = firstShared - 1; j >= 0; j--) {
    const t = C.layerTypes[j];
    if (!(t in shareSrc)) shareSrc[t] = layers[j];
  }
  for (let i = firstShared; i < C.layers; i++) layers[i].cacheSrc = shareSrc[C.layerTypes[i]];

  const model = {
    embQ: upload(device, new Uint8Array((await st.fetch(P + "embed_tokens.embedding_quantized")).buf)),
    embS: upload(device, new Float32Array((await st.fetch(P + "embed_tokens.embedding_scale")).buf)),
    pleQ: upload(device, new Uint8Array((await st.fetch(P + "embed_tokens_per_layer.embedding_quantized")).buf)),
    pleS: upload(device, new Float32Array((await st.fetch(P + "embed_tokens_per_layer.embedding_scale")).buf)),
    pleProjW: upload(device, bf16ToF32((await st.fetch(P + "per_layer_model_projection.weight")).buf)),
    pleProjScale: upload(device, new Float32Array(PLE_TOTAL).fill(1 / Math.sqrt(C.hidden))),
    pleProjNorm: f32buf(await st.fetch(P + "per_layer_projection_norm.weight")),
    finalNorm: f32buf(await st.fetch(P + "norm.weight")),
    lmHeadQ: upload(device, await (async () => {
      // repack row-major int2 rows (IN/4 bytes each) into 32-row tiles laid out
      // [vec4word j][row t] so subgroup reads are coalesced (see lmhead.wgsl)
      const raw = new Uint32Array((await st.fetch("lm_head.weight")).buf.slice(0));
      const rows = C.vocab, vwords = C.hidden / 64, T = 128;   // vec4<u32> per row
      const out = new Uint32Array(rows * vwords * 4);
      for (let r = 0; r < rows; r++) {
        const tile = Math.floor(r / T), t = r % T;
        for (let j = 0; j < vwords; j++)
          for (let h = 0; h < 4; h++)
            out[(tile * vwords * T + j * T + t) * 4 + h] = raw[r * vwords * 4 + j * 4 + h];
      }
      return out;
    })()),
    lmHeadS: upload(device, new Float32Array((await st.fetch("lm_head.weight_scale")).buf)),
    lmHeadSrq: upload(device, new Float32Array([
      scalar(await st.fetch("lm_head.input_activation_scale")),
      scalar(await st.fetch("lm_head.output_activation_scale"))]), GPUBufferUsage.UNIFORM),
  };
  L(`weights on GPU in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

  // ---- activation buffers ----
  const A = {
    params: alloc(device, 16),                       // [pos, cacheLen, token]
    hidden: alloc(device, C.hidden * 4),
    normed: alloc(device, C.hidden * 4),
    tmp: alloc(device, C.hidden * 4),
    tmp2: alloc(device, C.hidden * 4),
    qkv: alloc(device, (C.qHeads + 2 * C.kvHeads) * C.globalHeadDim * 4),
    attnOut: alloc(device, C.qHeads * C.globalHeadDim * 4),
    attPartO: alloc(device, C.qHeads * (MAXSEQ / 64) * C.globalHeadDim * 4),
    attPartME: alloc(device, C.qHeads * (MAXSEQ / 64) * 8),
    gu: alloc(device, 2 * C.inter * 4),
    pleIdentity: alloc(device, PLE_TOTAL * 4),
    pleProj: alloc(device, PLE_TOTAL * 4),
    pleNormed: alloc(device, PLE_TOTAL * 4),
    pleInput: alloc(device, PLE_TOTAL * 4),
    pleGateOut: alloc(device, C.pleDim * 4),
    pleMul: alloc(device, C.pleDim * 4),
    logits: alloc(device, C.vocab * 4),
    xq: alloc(device, Math.ceil(C.inter / 4) * 4),         // packed int8 activation (max IN)
    xq3: alloc(device, 3 * C.hidden),                       // 3 offset regions (qkv / gate+up)
    amax: alloc(device, 16),
    tokRing: alloc(device, 1024 * 8),
    xqSums: alloc(device, 16),
    xqSumI: alloc(device, 16),
    amaxPart: alloc(device, 256 * 8),
    srqOff: upload(device, new Float32Array([0, 0]), GPUBufferUsage.UNIFORM),
    combineScale: upload(device, new Float32Array([Math.SQRT1_2])),
  };

  // ---- pipelines ----
  const MV_R = 2;
  const mv = (bits, IN, OUT, zp = 0) => K.pipeline("matvec4", { BITS: bits, IN, OUT, SER: 1, ZP: zp, SOFTCAP: "0.0" });
  const srqCache = new Map();
  const srq8 = async (N) => {
    if (!srqCache.has(N)) srqCache.set(N, await K.pipeline("srq8", { N, WG: 64 }));
    return srqCache.get(N);
  };
  const kern = {
    embed: await K.pipeline("embedrow", { N: C.hidden, BLOCKS: 1, MULT: Math.sqrt(C.hidden).toFixed(8), PARAM_IDX: 2, WG: 256 }),
    plePrep: await K.pipeline("pleprep", { PD: C.pleDim, LAYERS: C.layers, MULT: Math.sqrt(C.pleDim).toFixed(8), EPS: C.eps, WG: 64 }),
    rmsHidden: await K.pipeline("rmsnorm", { DIM: C.hidden, EPS: C.eps, WITH_SCALE: 1, SUMOUT: 1, WG: 256 }),
    accH: await K.pipeline("acc", { N: C.hidden, WG: 256 }),
    accMulH: await K.pipeline("accmul", { N: C.hidden, WG: 256 }),
    addMulPle: await K.pipeline("addmul", { N: PLE_TOTAL, WG: 256 }),
    gegluMul: await K.pipeline("geglumul", { N: C.inter, WG: 256 }),
    pleProjMv: await K.pipeline("matvec2f", { BITS: 32, IN: C.hidden, OUT: PLE_TOTAL, WG: 64, SOFTCAP: "0.0" }),
    lmHead: await K.pipeline("lmhead", { IN: C.hidden, OUT: C.vocab, TILE: 128, SOFTCAP: C.softcap.toFixed(1) }),
    srqH: await (async () => K.pipeline("srq8", { N: C.hidden, WG: 64 }))().then(x=>x),
    argmax0: await K.pipeline("argmax2", { N: C.vocab, PARTS: 256, STAGE: 0, WG: 256 }),
    argmax1: await K.pipeline("argmax2", { N: C.vocab, PARTS: 256, STAGE: 1, WG: 256 }),
    rmssrq1: await K.pipeline("rmssrq", { DIM: C.hidden, NS: 1, EPS: C.eps, WG: 256 }),
    rmssrq2: await K.pipeline("rmssrq", { DIM: C.hidden, NS: 2, EPS: C.eps, WG: 256 }),
    rmssrq3: await K.pipeline("rmssrq", { DIM: C.hidden, NS: 3, EPS: C.eps, WG: 256 }),
    rmsacc: await K.pipeline("rmsacc", { DIM: C.hidden, EPS: C.eps, MUL: "1.0", WG: 256 }),
    gegluSrq: await K.pipeline("geglusrq", { N: C.inter, WG: 256 }),
    rmsaccFfn: await K.pipeline("rmsaccsrq", { DIM: C.hidden, NS: 2, EPS: C.eps, MUL: "1.0", NORM2: 1, WG: 256 }),
    rmsaccPle: await K.pipeline("rmsaccsrq", { DIM: C.hidden, NS: 1, EPS: C.eps, MUL: "1.0", NORM2: 0, WG: 256 }),
    feedTok: await K.pipeline("feedtok", {}),
  };
  const attKernCache = new Map();
  async function attKerns(headDim, isSliding) {
    const key = headDim + ":" + isSliding;
    if (!attKernCache.has(key)) {
      const angles = isSliding ? headDim / 2 : Math.floor(0.25 * headDim / 2);
      const theta = isSliding ? "10000.0" : "1000000.0";
      attKernCache.set(key, {
        qNorm: await K.pipeline("headnorm", { HEAD_DIM: headDim, WITH_SCALE: 1, EPS: C.eps, WG: 128 }),
        vNorm: await K.pipeline("headnorm", { HEAD_DIM: headDim, WITH_SCALE: 0, EPS: C.eps, WG: 128 }),
        ropeQ: await K.pipeline("rope", { HEADS: C.qHeads, HEAD_DIM: headDim, ROPE_ANGLES: angles, THETA: theta, WG: 128 }),
        ropeK: await K.pipeline("rope", { HEADS: C.kvHeads, HEAD_DIM: headDim, ROPE_ANGLES: angles, THETA: theta, WG: 128 }),
        kvW: await K.pipeline("kvwrite", { N: C.kvHeads * headDim, WG: 128 }),
        headprep: await K.pipeline("headprep", { QH: C.qHeads, KVH: C.kvHeads, HEAD_DIM: headDim, ROPE_ANGLES: isSliding ? headDim / 2 : Math.floor(0.25 * headDim / 2), THETA: isSliding ? "10000.0" : "1000000.0", EPS: C.eps, WG: 128 }),
        att1: await K.pipeline("attention1", { Q_HEADS: C.qHeads, KV_HEADS: C.kvHeads, HEAD_DIM: headDim, MAXSEQ, WINDOW: isSliding ? C.window : 0, DT: 4, WG: 64 }),
      });
    }
    return attKernCache.get(key);
  }
  for (const l of layers) {
    l.srqAttnOut = await srq8(C.qHeads * l.headDim);
    l.qkvMv = await K.pipeline("matvecg", { BITS: 4, IN: C.hidden, OUT: l.qkv.out,
      B0: l.qkv.bounds[0], B1: l.qkv.bounds[1] ?? l.qkv.bounds[0] });
    l.guMv = await K.pipeline("matvecgu", { IN: C.hidden, OUT: C.inter });
    l.pleGateMvF = await K.pipeline("plegatemv", { IN: C.hidden, OUT: C.pleDim, OFF: l.i * C.pleDim });
    l.oMv = await mv(4, C.qHeads * l.headDim, C.hidden, 1);   // Σq from att1 atomics
    l.downMv = await mv(4, C.inter, C.hidden, 1);    // Σq from the gu epilogue atomics
    l.downMvZ = l.downMv;                            // downtest alias
    l.pleGateMv = await mv(8, C.hidden, C.pleDim);
    l.pleProjMv = await mv(8, C.pleDim, C.hidden);
    l.pleMulSrq = await K.pipeline("plemulsrq", { N: C.pleDim, OFF: l.i * C.pleDim, WG: 64 });
    l.rmsaccMul = await K.pipeline("rmsacc", { DIM: C.hidden, EPS: C.eps, MUL: l.layerScalarVal.toPrecision(9), WG: 256 });
    l.kern = await attKerns(l.headDim, l.isSliding);
  }
  for (let i = 0; i + 1 < layers.length; i++) {
    const nx = layers[i + 1];
    layers[i].rmsaccNext = await K.pipeline("rmsaccsrq", {
      DIM: C.hidden, NS: nx.isShared ? 1 : 3, EPS: C.eps,
      MUL: layers[i].layerScalarVal.toPrecision(9), NORM2: 1, WG: 256 });
    layers[i].next = nx;
  }
  L("pipelines built");

  // ---- bind-group cache (engine-level) ----
  const bgCache = new Map();
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

  // ---- forward: one token, one compute pass ----
  const wg = (n, w) => Math.ceil(n / w);
  const mkRun = (pass) => (k, bufs, groups) => { pass.setPipeline(k.pipeline); pass.setBindGroup(0, bind(k, bufs)); pass.dispatchWorkgroups(...(Array.isArray(groups) ? groups : [groups])); };
  const wg2 = (rows) => rows <= 32768 ? [rows] : [32768, Math.ceil(rows / 32768)];

  function encodePre(ctx) {
    const run = ctx.run;
    run(kern.embed, [model.embQ, model.embS, A.params, A.hidden], wg(C.hidden, 256));
    // PLE inputs: identity + context projection

    run(kern.pleProjMv, [A.hidden, model.pleProjW, model.pleProjScale, A.srqOff, A.pleProj], PLE_TOTAL);
    // norm per 256-block (42 rows), then (proj + identity) * 2^-0.5
    run(kern.plePrep, [model.pleQ, model.pleS, A.params, A.pleProj, model.pleProjNorm, A.pleInput], C.layers);
  }

  function encodeLayer(ctx, l) {
      const run = ctx.run;
      const kk = l.kern;
      const cache = l.isShared ? l.cacheSrc : l;
      const xv = (r) => ({ buffer: A.xq3, offset: r * C.hidden, size: C.hidden });
      const qview = { buffer: A.qkv, offset: 0, size: l.qOut * 4 };
      const kview = { buffer: A.qkv, offset: l.qOut * 4, size: l.kvOut * 4 };
      const vview = { buffer: A.qkv, offset: (l.qOut + l.kvOut) * 4, size: l.kvOut * 4 };
      // attention: layer 0's norm+quant runs here; later layers get it fused
      // into the previous layer's boundary op (rmsaccNext)
      if (l.i === 0 || AB_NOFUSE) run(l.isShared ? kern.rmssrq1 : kern.rmssrq3, [A.hidden, l.inNorm, l.qkvScales, A.xq3, A.xqSums], 1);
      run(l.qkvMv, [A.xq3, l.qkv.wBuf, l.qkv.wsBuf, l.qkv.srqsBuf, A.qkv, A.xqSums], wg2(Math.ceil(l.qkv.out / 2)));
      run(kk.headprep, [A.qkv, l.qNorm, l.isShared ? l.qNorm : l.kNorm, A.params, cache.kCache, cache.vCache, A.xqSumI],
          l.isShared ? C.qHeads : C.qHeads + 2 * C.kvHeads);
      run(kk.att1, [qview, cache.kCache, cache.vCache, A.params, l.o.srqBuf, A.xq, A.xqSumI], [C.qHeads, 4]);
      run(l.oMv, [A.xq, l.o.wBuf, l.o.wsBuf, l.o.srqBuf, A.tmp, A.xqSumI], Math.ceil(C.hidden / MV_R));
      // fused: residual add + pre-ffn norm + gate/up quant regions
      run(kern.rmsaccFfn, [A.tmp, l.postAttnNorm, A.hidden, l.preFfnNorm, l.gateUpScales, A.xq3, A.xqSums, A.xqSumI], 1);
      run(l.guMv, [A.xq3, l.gu.wBuf, l.gu.wsBuf, l.guSrqs, A.xq, A.xqSums, A.xqSumI], wg2(Math.ceil(C.inter / 4)));
      run(l.downMv, [A.xq, l.down.wBuf, l.down.wsBuf, l.down.srqBuf, A.tmp, A.xqSumI], Math.ceil(C.hidden / MV_R));
      // fused: residual add + pleGate quant (raw hidden, no norm)
      run(kern.rmsaccPle, [A.tmp, l.postFfnNorm, A.hidden, l.postFfnNorm, l.pleGateScales, A.xq, A.xqSums, A.xqSumI], 1);
      run(l.pleGateMvF, [A.xq, l.pleGate.wBuf, l.pleGate.wsBuf, l.pleSrqs, A.pleInput,
                          { buffer: A.xq3, offset: 0, size: 256 }], C.pleDim / 4);
      run(l.pleProjMv, [{ buffer: A.xq3, offset: 0, size: 256 }, l.pleProj.wBuf, l.pleProj.wsBuf, l.pleProj.srqBuf, A.tmp, A.xqSumI], Math.ceil(C.hidden / MV_R));
      // layer boundary: residual+layer_scalar + NEXT layer's input norm+quant, fused
      if (l.next && !AB_NOFUSE) {
        run(l.rmsaccNext, [A.tmp, l.postPleNorm, A.hidden, l.next.inNorm, l.next.qkvScales, A.xq3, A.xqSums, A.xqSumI], 1);
      } else {
        run(l.rmsaccMul, [A.tmp, l.postPleNorm, A.hidden], 1);
      }
  }

  function encodeFinal(ctx) {
    const run = ctx.run;
    run(kern.rmsHidden, [A.hidden, model.finalNorm, A.normed, A.xqSums], 1);
    run(kern.lmHead, [A.normed, model.lmHeadQ, model.lmHeadS, A.logits, A.xqSums], wg2(C.vocab / 128));
    run(kern.argmax0, [A.logits, A.amaxPart, A.amax], 256);
    run(kern.argmax1, [A.logits, A.amaxPart, A.amax], 1);
  }

  function encodeForward(ctx) {
    encodePre(ctx);
    for (const l of layers) encodeLayer(ctx, l);
    encodeFinal(ctx);
  }

  async function step(token, pos) {
    device.queue.writeBuffer(A.params, 0, new Uint32Array([pos, pos + 1, token, 0]));
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    encodeForward({ run: mkRun(pass) });
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  // GPU per-dispatch budget: pass-per-dispatch with timestamp queries.
  // NOTE (hesper lesson): pass-per-dispatch serializes — per-class times are upper
  // bounds and their sum exceeds the real single-pass wall. Use for RANKING only.
  async function profileStep(token, pos) {
    if (!device.features.has("timestamp-query")) return null;
    device.queue.writeBuffer(A.params, 0, new Uint32Array([pos, pos + 1, token, 0]));
    const qs = device.createQuerySet({ type: "timestamp", count: 4096 });
    const enc = device.createCommandEncoder();
    const labels = [];
    const ctx = { run: (k, bufs, groups) => {
      const i = labels.length;
      const pass = enc.beginComputePass({ timestampWrites: {
        querySet: qs, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } });
      pass.setPipeline(k.pipeline);
      pass.setBindGroup(0, bind(k, bufs));
      pass.dispatchWorkgroups(...(Array.isArray(groups) ? groups : [groups]));
      pass.end();
      labels.push(k.pipeline.label);
    } };
    encodeForward(ctx);
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

  async function argmaxLogits() {
    const buf = new Float32Array(await readback(device, A.logits, C.vocab * 4));
    let best = 0;
    for (let i = 1; i < C.vocab; i++) if (buf[i] > buf[best]) best = i;
    return { id: best, logits: buf };
  }

  async function argmaxFast() {
    const u = new Uint32Array(await readback(device, A.amax, 8));
    return u[0];
  }

  // decode WITHOUT per-token CPU sync: feedTok copies the previous argmax into
  // params.token on the GPU; each step's argmax is copied into a ring buffer
  // and read back in chunks. CPU encodes ahead while the GPU executes.
  function decodeChunk(startPos, count, ringBase) {
    for (let i = 0; i < count; i++) {
      const pos = startPos + i;
      device.queue.writeBuffer(A.params, 0, new Uint32Array([pos, pos + 1]));
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      const run = mkRun(pass);
      run(kern.feedTok, [A.amax, A.params], 1);
      encodeForward({ run });
      pass.end();
      enc.copyBufferToBuffer(A.amax, 0, A.tokRing, (ringBase + i) * 8, 8);
      device.queue.submit([enc.finish()]);
    }
  }

  // greedy decode, GPU-side feedback. Returns generated ids (incl. EOS if hit).
  async function generateFast(inputIds, maxNew, eosIds = new Set([106, 1])) {
    let pos = 0;
    for (const t of inputIds) await step(t, pos++);
    const g0 = await argmaxFast();                 // first generated token
    const out = [g0];
    if (eosIds.has(g0)) return out;
    const CHUNK = 8;
    let done = 1;                                  // generated tokens so far
    while (done < maxNew) {
      const n = Math.min(CHUNK, maxNew - done);
      decodeChunk(pos, n, done - 1);               // ring[k] = token g_{k+1}
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

  async function generate(inputIds, maxNew, onToken = null) {
    let pos = 0;
    for (const t of inputIds) await step(t, pos++);
    const out = [];
    let id = await argmaxFast();
    out.push(id);
    onToken?.(id);
    while (out.length < maxNew) {
      await step(id, pos++);
      id = await argmaxFast();
      out.push(id);
      onToken?.(id);
    }
    return out;
  }

  // Debug: run one token layer-by-layer, calling cb(stage, hiddenF32) after embed and each layer.
  async function stepBisect(token, pos, cb) {
    device.queue.writeBuffer(A.params, 0, new Uint32Array([pos, pos + 1, token, 0]));
    let enc = device.createCommandEncoder();
    let pass = enc.beginComputePass();
    encodePre({ run: mkRun(pass) });
    pass.end();
    device.queue.submit([enc.finish()]);
    await cb("embed", new Float32Array(await readback(device, A.hidden, C.hidden * 4)));
    for (const l of layers) {
      enc = device.createCommandEncoder();
      pass = enc.beginComputePass();
      encodeLayer({ run: mkRun(pass) }, l);
      pass.end();
      device.queue.submit([enc.finish()]);
      await cb("layer" + l.i, new Float32Array(await readback(device, A.hidden, C.hidden * 4)));
    }
    enc = device.createCommandEncoder();
    pass = enc.beginComputePass();
    encodeFinal({ run: mkRun(pass) });
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  async function readHidden() {
    return new Float32Array(await readback(device, A.hidden, C.hidden * 4));
  }

  return { device, C, layers, model, A, kern, step, generate, generateFast, decodeChunk, argmaxLogits, argmaxFast, readHidden, encodeForward, stepBisect, profileStep, encodePre, encodeLayerPub: encodeLayer, bindPub: bind };
}
