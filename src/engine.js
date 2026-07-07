// Gemma-4 E4B QAT decoder on WebGPU — bring-up build (f32, naive kernels, correctness first).
// Arch: NOTES/arch-spec.md · weight format: NOTES/qat-format.md.
import { openSafetensors, bf16ToF32 } from "./loader.js";
import { initDevice, upload, alloc, readback, Kernels } from "./gpu.js";

const L = (m) => fetch("/log", { method: "POST", body: String(m) }).catch(() => {});
const MAXSEQ = 640;

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

  async function linear(prefix) {
    const w = await st.fetch(prefix + ".weight");
    const inS = scalar(await st.fetch(prefix + ".input_activation_scale"));
    const outS = scalar(await st.fetch(prefix + ".output_activation_scale"));
    return {
      out: w.shape[0],
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
      qNorm: f32buf(await st.fetch(p + "self_attn.q_norm.weight")),
      q: await linear(p + "self_attn.q_proj"),
      o: await linear(p + "self_attn.o_proj"),
      gate: await linear(p + "mlp.gate_proj"),
      up: await linear(p + "mlp.up_proj"),
      down: await linear(p + "mlp.down_proj"),
      pleGate: await linear(p + "per_layer_input_gate"),
      pleProj: await linear(p + "per_layer_projection"),
    };
    if (!isShared) {
      l.kNorm = f32buf(await st.fetch(p + "self_attn.k_norm.weight"));
      l.k = await linear(p + "self_attn.k_proj");
      l.v = await linear(p + "self_attn.v_proj");
      l.kCache = alloc(device, MAXSEQ * C.kvHeads * headDim * 4);
      l.vCache = alloc(device, MAXSEQ * C.kvHeads * headDim * 4);
    }
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
    lmHeadQ: upload(device, new Uint8Array((await st.fetch("lm_head.weight")).buf)),
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
    q: alloc(device, C.qHeads * C.globalHeadDim * 4),
    kv: alloc(device, C.kvHeads * C.globalHeadDim * 4),
    kv2: alloc(device, C.kvHeads * C.globalHeadDim * 4),
    attnOut: alloc(device, C.qHeads * C.globalHeadDim * 4),
    gate: alloc(device, C.inter * 4),
    up: alloc(device, C.inter * 4),
    geglu: alloc(device, C.inter * 4),
    pleIdentity: alloc(device, PLE_TOTAL * 4),
    pleProj: alloc(device, PLE_TOTAL * 4),
    pleNormed: alloc(device, PLE_TOTAL * 4),
    pleInput: alloc(device, PLE_TOTAL * 4),
    pleGateOut: alloc(device, C.pleDim * 4),
    pleMul: alloc(device, C.pleDim * 4),
    logits: alloc(device, C.vocab * 4),
    srqOff: upload(device, new Float32Array([0, 0]), GPUBufferUsage.UNIFORM),
    combineScale: upload(device, new Float32Array([Math.SQRT1_2])),
  };

  // ---- pipelines ----
  const mv = (bits, IN, OUT) => K.pipeline("matvec", { BITS: bits, IN, OUT, WG: 64, SOFTCAP: "0.0" });
  const kern = {
    embed: await K.pipeline("embedrow", { N: C.hidden, BLOCKS: 1, MULT: Math.sqrt(C.hidden).toFixed(8), PARAM_IDX: 2, WG: 256 }),
    pleRow: await K.pipeline("embedrow", { N: PLE_TOTAL, BLOCKS: C.layers, MULT: Math.sqrt(C.pleDim).toFixed(8), PARAM_IDX: 2, WG: 256 }),
    rmsHidden: await K.pipeline("rmsnorm", { DIM: C.hidden, EPS: C.eps, WITH_SCALE: 1, WG: 256 }),
    rmsPle: await K.pipeline("rmsnorm", { DIM: C.pleDim, EPS: C.eps, WITH_SCALE: 1, WG: 64 }),
    accH: await K.pipeline("acc", { N: C.hidden, WG: 256 }),
    accMulH: await K.pipeline("accmul", { N: C.hidden, WG: 256 }),
    addMulPle: await K.pipeline("addmul", { N: PLE_TOTAL, WG: 256 }),
    gegluMul: await K.pipeline("geglumul", { N: C.inter, WG: 256 }),
    pleProjMv: await K.pipeline("matvec", { BITS: 32, IN: C.hidden, OUT: PLE_TOTAL, WG: 64, SOFTCAP: "0.0" }),
    lmHead: await K.pipeline("matvec", { BITS: 2, IN: C.hidden, OUT: C.vocab, WG: 64, SOFTCAP: C.softcap.toFixed(1) }),
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
        attn: await K.pipeline("attention", { Q_HEADS: C.qHeads, KV_HEADS: C.kvHeads, HEAD_DIM: headDim, MAXSEQ, WINDOW: isSliding ? C.window : 0, WG: 128 }),
      });
    }
    return attKernCache.get(key);
  }
  for (const l of layers) {
    l.qMv = await mv(4, C.hidden, l.q.out);
    l.oMv = await mv(4, C.qHeads * l.headDim, C.hidden);
    l.gateMv = await mv(4, C.hidden, C.inter);
    l.upMv = await mv(4, C.hidden, C.inter);
    l.downMv = await mv(4, C.inter, C.hidden);
    l.pleGateMv = await mv(8, C.hidden, C.pleDim);
    l.pleProjMv = await mv(8, C.pleDim, C.hidden);
    l.pleMulK = await K.pipeline("plemul", { N: C.pleDim, OFF: l.i * C.pleDim, WG: 64 });
    if (!l.isShared) { l.kMv = await mv(4, C.hidden, l.k.out); l.vMv = await mv(4, C.hidden, l.v.out); }
    l.kern = await attKerns(l.headDim, l.isSliding);
  }
  L("pipelines built");

  // ---- bind-group cache (engine-level) ----
  const bgCache = new Map();
  function bind(kernEntry, buffers) {
    const key = kernEntry.pipeline.label + "|" + buffers.map((b) => b.__id ?? (b.__id = Math.random())).join(",");
    let bg = bgCache.get(key);
    if (!bg) {
      bg = device.createBindGroup({
        layout: kernEntry.pipeline.getBindGroupLayout(0),
        entries: buffers.map((b, i) => ({ binding: i, resource: { buffer: b } })),
      });
      bgCache.set(key, bg);
    }
    return bg;
  }

  // ---- forward: one token, one compute pass ----
  const wg = (n, w) => Math.ceil(n / w);
  const mkRun = (pass) => (k, bufs, groups) => { pass.setPipeline(k.pipeline); pass.setBindGroup(0, bind(k, bufs)); pass.dispatchWorkgroups(groups); };

  function encodePre(pass) {
    const run = mkRun(pass);
    run(kern.embed, [model.embQ, model.embS, A.params, A.hidden], wg(C.hidden, 256));
    // PLE inputs: identity + context projection
    run(kern.pleRow, [model.pleQ, model.pleS, A.params, A.pleIdentity], wg(PLE_TOTAL, 256));
    run(kern.pleProjMv, [A.hidden, model.pleProjW, model.pleProjScale, A.srqOff, A.pleProj], wg(PLE_TOTAL, 64));
    // norm per 256-block (42 rows), then (proj + identity) * 2^-0.5
    run(kern.rmsPle, [A.pleProj, model.pleProjNorm, A.pleNormed], C.layers);
    run(kern.addMulPle, [A.pleNormed, A.pleIdentity, A.combineScale, A.pleInput], wg(PLE_TOTAL, 256));
  }

  function encodeLayer(pass, l) {
      const run = mkRun(pass);
      const kk = l.kern;
      const cache = l.isShared ? l.cacheSrc : l;
      // attention
      run(kern.rmsHidden, [A.hidden, l.inNorm, A.normed], 1);
      run(l.qMv, [A.normed, l.q.wBuf, l.q.wsBuf, l.q.srqBuf, A.q], wg(l.q.out, 64));
      run(kk.qNorm, [l.qNorm, A.q], C.qHeads);
      run(kk.ropeQ, [A.q, A.params], wg(C.qHeads * l.headDim / 2, 128));
      if (!l.isShared) {
        run(l.kMv, [A.normed, l.k.wBuf, l.k.wsBuf, l.k.srqBuf, A.kv], wg(l.k.out, 64));
        run(kk.qNorm, [l.kNorm, A.kv], C.kvHeads);               // k_norm (same kernel shape)
        run(kk.ropeK, [A.kv, A.params], wg(C.kvHeads * l.headDim / 2, 128));
        run(kk.kvW, [A.kv, A.params, l.kCache], wg(C.kvHeads * l.headDim, 128));
        run(l.vMv, [A.normed, l.v.wBuf, l.v.wsBuf, l.v.srqBuf, A.kv2], wg(l.v.out, 64));
        run(kk.vNorm, [l.qNorm /*unused dummy*/, A.kv2], C.kvHeads);
        run(kk.kvW, [A.kv2, A.params, l.vCache], wg(C.kvHeads * l.headDim, 128));
      }
      run(kk.attn, [A.q, cache.kCache, cache.vCache, A.params, A.attnOut], C.qHeads);
      run(l.oMv, [A.attnOut, l.o.wBuf, l.o.wsBuf, l.o.srqBuf, A.tmp], wg(C.hidden, 64));
      run(kern.rmsHidden, [A.tmp, l.postAttnNorm, A.tmp2], 1);
      run(kern.accH, [A.tmp2, A.hidden], wg(C.hidden, 256));
      // mlp
      run(kern.rmsHidden, [A.hidden, l.preFfnNorm, A.normed], 1);
      run(l.gateMv, [A.normed, l.gate.wBuf, l.gate.wsBuf, l.gate.srqBuf, A.gate], wg(C.inter, 64));
      run(l.upMv, [A.normed, l.up.wBuf, l.up.wsBuf, l.up.srqBuf, A.up], wg(C.inter, 64));
      run(kern.gegluMul, [A.gate, A.up, A.geglu], wg(C.inter, 256));
      run(l.downMv, [A.geglu, l.down.wBuf, l.down.wsBuf, l.down.srqBuf, A.tmp], wg(C.hidden, 64));
      run(kern.rmsHidden, [A.tmp, l.postFfnNorm, A.tmp2], 1);
      run(kern.accH, [A.tmp2, A.hidden], wg(C.hidden, 256));
      // PLE block (+ layer_scalar folded into the final addmul)
      run(l.pleGateMv, [A.hidden, l.pleGate.wBuf, l.pleGate.wsBuf, l.pleGate.srqBuf, A.pleGateOut], wg(C.pleDim, 64));
      run(l.pleMulK, [A.pleGateOut, A.pleInput, A.pleMul], wg(C.pleDim, 64));
      run(l.pleProjMv, [A.pleMul, l.pleProj.wBuf, l.pleProj.wsBuf, l.pleProj.srqBuf, A.tmp], wg(C.hidden, 64));
      run(kern.rmsHidden, [A.tmp, l.postPleNorm, A.tmp2], 1);
      run(kern.accMulH, [A.tmp2, l.layerScalar, A.hidden], wg(C.hidden, 256));
  }

  function encodeFinal(pass) {
    const run = mkRun(pass);
    run(kern.rmsHidden, [A.hidden, model.finalNorm, A.normed], 1);
    run(kern.lmHead, [A.normed, model.lmHeadQ, model.lmHeadS, model.lmHeadSrq, A.logits], wg(C.vocab, 64));
  }

  function encodeForward(pass) {
    encodePre(pass);
    for (const l of layers) encodeLayer(pass, l);
    encodeFinal(pass);
  }

  async function step(token, pos) {
    device.queue.writeBuffer(A.params, 0, new Uint32Array([pos, pos + 1, token, 0]));
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    encodeForward(pass);
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  async function argmaxLogits() {
    const buf = new Float32Array(await readback(device, A.logits, C.vocab * 4));
    let best = 0;
    for (let i = 1; i < C.vocab; i++) if (buf[i] > buf[best]) best = i;
    return { id: best, logits: buf };
  }

  async function generate(inputIds, maxNew, onToken = null) {
    let pos = 0;
    for (const t of inputIds) await step(t, pos++);
    const out = [];
    let { id } = await argmaxLogits();
    out.push(id);
    onToken?.(id);
    while (out.length < maxNew) {
      await step(id, pos++);
      ({ id } = await argmaxLogits());
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
    encodePre(pass);
    pass.end();
    device.queue.submit([enc.finish()]);
    await cb("embed", new Float32Array(await readback(device, A.hidden, C.hidden * 4)));
    for (const l of layers) {
      enc = device.createCommandEncoder();
      pass = enc.beginComputePass();
      encodeLayer(pass, l);
      pass.end();
      device.queue.submit([enc.finish()]);
      await cb("layer" + l.i, new Float32Array(await readback(device, A.hidden, C.hidden * 4)));
    }
    enc = device.createCommandEncoder();
    pass = enc.beginComputePass();
    encodeFinal(pass);
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  async function readHidden() {
    return new Float32Array(await readback(device, A.hidden, C.hidden * 4));
  }

  return { device, C, layers, model, A, kern, step, generate, argmaxLogits, readHidden, encodeForward, stepBisect };
}
