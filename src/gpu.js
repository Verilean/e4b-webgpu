// Thin WebGPU helpers: device init, buffer upload, kernel registry with ${} substitution
// (kernels are DATA — plain .wgsl files fetched at runtime, hot-reloadable).

export async function initDevice() {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("no WebGPU adapter");
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: Math.min(1 << 30, adapter.limits.maxStorageBufferBindingSize),
      maxBufferSize: Math.min(1 << 30, adapter.limits.maxBufferSize),
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
    },
  });
  device.addEventListener("uncapturederror", (e) => {
    // surface validation errors loudly — hesper lesson: silent drops cost a day
    console.error("WebGPU error:", e.error.message);
    fetch("/log", { method: "POST", body: "WEBGPU ERROR: " + e.error.message }).catch(() => {});
  });
  return device;
}

export function upload(device, data, usage = GPUBufferUsage.STORAGE) {
  const src = ArrayBuffer.isView(data) ? data : new Uint8Array(data);
  const size = Math.max(16, (src.byteLength + 3) & ~3);
  const buf = device.createBuffer({ size, usage: usage | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buf, 0, src.buffer ?? src, src.byteOffset ?? 0, src.byteLength);
  return buf;
}

export function alloc(device, bytes) {
  return device.createBuffer({
    size: Math.max(16, (bytes + 3) & ~3),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
}

export async function readback(device, buf, bytes) {
  const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, staging, 0, bytes);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = staging.getMappedRange().slice(0);
  staging.destroy();
  return out;
}

// ---- kernel registry: fetch kernels/<name>.wgsl, substitute ${K} params, cache pipelines ----
export class Kernels {
  constructor(device) {
    this.device = device;
    this.sources = new Map();   // file -> text
    this.pipelines = new Map(); // key -> {pipeline, layout}
  }
  async source(file) {
    if (!this.sources.has(file)) {
      const r = await fetch(`kernels/${file}.wgsl?t=${Date.now()}`); // no-cache: hot reload
      if (!r.ok) throw new Error(`kernel fetch failed: ${file}`);
      this.sources.set(file, await r.text());
    }
    return this.sources.get(file);
  }
  async pipeline(file, params = {}) {
    const key = file + JSON.stringify(params);
    if (this.pipelines.has(key)) return this.pipelines.get(key);
    let src = await this.source(file);
    src = src.replace(/\$\{(\w+)\}/g, (_, k) => {
      if (!(k in params)) throw new Error(`kernel ${file}: missing param ${k}`);
      return String(params[k]);
    });
    const module = this.device.createShaderModule({ code: src, label: key });
    const pipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
      label: key,
    });
    const entry = { pipeline };
    this.pipelines.set(key, entry);
    return entry;
  }
}

// One token's dispatches recorded into a single encoder, submitted once.
export class Pass {
  constructor(device) {
    this.device = device;
    this.enc = device.createCommandEncoder();
    this.pass = this.enc.beginComputePass();
    this.bgCache = new Map();
  }
  run(kern, buffers, groups) {
    const key = kern.pipeline.label + buffers.map((b) => b.label ?? b.size).join(",");
    let bg = this.bgCache.get(key);
    if (!bg) {
      bg = this.device.createBindGroup({
        layout: kern.pipeline.getBindGroupLayout(0),
        entries: buffers.map((b, i) => ({ binding: i, resource: { buffer: b } })),
      });
      this.bgCache.set(key, bg);
    }
    this.pass.setPipeline(kern.pipeline);
    this.pass.setBindGroup(0, bg);
    this.pass.dispatchWorkgroups(...(Array.isArray(groups) ? groups : [groups]));
  }
  submit() {
    this.pass.end();
    this.device.queue.submit([this.enc.finish()]);
  }
}
