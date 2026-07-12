// Campaign 3 Metal runner: executes the browser engine's traced dispatch plans
// (metal/out/manifest.json from convert.py; kernels = tint-MSL compiled at
// RUNTIME → kernels stay data) with the real dumped weights (metal/bins/).
// One serial compute encoder per command buffer = the boundary-tax experiment.
//
//   clang++ -fobjc-arc -std=c++17 -O2 metal/runner.mm \
//     -framework Metal -framework Foundation -o metal/runner
//   ./metal/runner gate [n] [fastmath]     token gate vs goldens-a4b.json
//   ./metal/runner bench [n] [fastmath]    decode wall (feed plan, GPU feedback)
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <vector>
#include <string>
#include <set>
#include <unordered_map>
#include <mach/mach_time.h>

static double nowMs() {
  static mach_timebase_info_data_t tb; if (!tb.denom) mach_timebase_info(&tb);
  return mach_absolute_time() * (double)tb.numer / tb.denom / 1e6;
}

struct Op { int p; MTLSize grid; std::vector<std::array<int,3>> binds; /* k, buf, binding */
            uint32_t sizes[16]; };

int main(int argc, char** argv) { @autoreleasepool {
  std::string mode = argc > 1 ? argv[1] : "gate";
  int nTok = argc > 2 ? atoi(argv[2]) : 16;
  bool fastMath = argc > 3 && atoi(argv[3]) == 1;
  NSString* root = [[NSString stringWithUTF8String:__FILE__]
                    stringByDeletingLastPathComponent];       // .../metal
  NSString* outDir = [root stringByAppendingPathComponent:@"out"];
  NSString* binDir = [root stringByAppendingPathComponent:@"bins"];

  NSData* mfd = [NSData dataWithContentsOfFile:[outDir stringByAppendingPathComponent:@"manifest.json"]];
  if (!mfd) { fprintf(stderr, "no manifest\n"); return 1; }
  NSDictionary* mf = [NSJSONSerialization JSONObjectWithData:mfd options:0 error:nil];

  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> queue = [dev newCommandQueue];
  printf("[runner] device=%s fastMath=%d\n", dev.name.UTF8String, fastMath);

  // ---- pipelines (runtime MSL compile: kernels as data) ----
  double t0 = nowMs();
  NSArray* pipes = mf[@"pipes"];
  std::vector<id<MTLComputePipelineState>> psos;
  std::vector<MTLSize> wgs;
  std::vector<NSUInteger> tgBytes;
  for (NSDictionary* p in pipes) {
    NSString* msl = [NSString stringWithContentsOfFile:
      [outDir stringByAppendingPathComponent:p[@"msl"]] encoding:NSUTF8StringEncoding error:nil];
    MTLCompileOptions* opt = [MTLCompileOptions new];
    opt.fastMathEnabled = fastMath;
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:msl options:opt error:&err];
    if (!lib) { fprintf(stderr, "compile FAIL %s: %s\n",
      [p[@"msl"] UTF8String], err.localizedDescription.UTF8String); return 1; }
    id<MTLFunction> fn = [lib newFunctionWithName:p[@"entry"]];
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) { fprintf(stderr, "pso FAIL %s: %s\n",
      [p[@"msl"] UTF8String], err.localizedDescription.UTF8String); return 1; }
    psos.push_back(pso);
    NSArray* w = p[@"wg"];
    wgs.push_back(MTLSizeMake([w[0] intValue], [w[1] intValue], [w[2] intValue]));
    tgBytes.push_back([p[@"tgBytes"] unsignedIntegerValue]);
  }
  printf("[runner] %lu PSOs in %.1fs\n", psos.size(), (nowMs() - t0) / 1000);

  // ---- buffers: bins/<id>.bin > manifest contents (base64) > zeros ----
  t0 = nowMs();
  std::unordered_map<int, id<MTLBuffer>> bufs;
  NSDictionary* contents = mf[@"contents"];
  double loaded = 0;
  for (NSDictionary* b in mf[@"buffers"]) {
    int bid = [b[@"id"] intValue];
    NSUInteger size = [b[@"size"] unsignedIntegerValue];
    id<MTLBuffer> mb = [dev newBufferWithLength:MAX(size, (NSUInteger)16)
                        options:MTLResourceStorageModeShared];
    NSString* binPath = [binDir stringByAppendingPathComponent:
                         [NSString stringWithFormat:@"%d.bin", bid]];
    NSData* bin = [NSData dataWithContentsOfFile:binPath
                   options:NSDataReadingMappedIfSafe error:nil];
    if (bin) {
      memcpy(mb.contents, bin.bytes, MIN((NSUInteger)bin.length, size));
      loaded += bin.length;
    } else {
      NSString* key = [NSString stringWithFormat:@"%d", bid];
      NSData* c = contents[key] ? [[NSData alloc] initWithBase64EncodedString:contents[key] options:0] : nil;
      memset(mb.contents, 0, size);
      if (c) memcpy(mb.contents, c.bytes, MIN((NSUInteger)c.length, size));
    }
    bufs[bid] = mb;
  }
  printf("[runner] %lu buffers, %.2f GB weights in %.1fs\n",
         bufs.size(), loaded / 1e9, (nowMs() - t0) / 1000);

  // ---- ops per plan; sizes UBO slots indexed by WGSL binding ----
  std::vector<Op> stepOps, feedOps, preOps, pre512Ops;
  std::unordered_map<int, NSUInteger> bufSize;
  for (NSDictionary* b in mf[@"buffers"]) bufSize[[b[@"id"] intValue]] = [b[@"size"] unsignedIntegerValue];
  for (NSDictionary* o in mf[@"ops"]) {
    Op op; op.p = [o[@"p"] intValue];
    NSArray* g = o[@"grid"];
    op.grid = MTLSizeMake([g[0] intValue], [g[1] intValue], [g[2] intValue]);
    memset(op.sizes, 0, sizeof(op.sizes));
    for (NSArray* e in o[@"binds"]) {
      int k = [e[0] intValue], bid = [e[1] intValue], bnd = [e[3] intValue];
      op.binds.push_back({k, bid, bnd});
      // tint's sizes UBO slots are indexed by the MSL buffer index (k), NOT
      // the WGSL binding (field tint_array_length_0_K reads sizes[K/4][K%4])
      if (k < 16) op.sizes[k] = (uint32_t)bufSize[bid];
    }
    NSString* pl = o[@"plan"];
    ([pl isEqualToString:@"step"] ? stepOps :
     [pl isEqualToString:@"feed"] ? feedOps :
     [pl isEqualToString:@"pre"] ? preOps : pre512Ops).push_back(op);
  }
  printf("[runner] step=%lu feed=%lu pre=%lu pre512=%lu ops\n",
         stepOps.size(), feedOps.size(), preOps.size(), pre512Ops.size());

  // semantic buffers from plan structure (engine layout, verified in DEVPLAN):
  // op0 = embed: MSL k4 = params; last op = argmax1: k2 = amax, k0 = logits
  auto findK = [](Op& op, int k) { for (auto& b : op.binds) if (b[0] == k) return b[1]; return -1; };
  int stepParams = findK(stepOps.front(), 4);
  int feedParams = findK(feedOps.front(), 4);
  int amaxId = findK(stepOps.back(), 2);
  // prefill plan op0 = embedB (q6k BATCH=1): k4 = paramsPre, k5 = tokPre
  int preParams = preOps.empty() ? -1 : findK(preOps.front(), 4);
  int preTok = preOps.empty() ? -1 : findK(preOps.front(), 5);
  printf("[runner] params(step)=%d params(feed)=%d amax=%d\n", stepParams, feedParams, amaxId);

  auto encodePlan = [&](id<MTLComputeCommandEncoder> enc, std::vector<Op>& ops,
                        int substBuf, id<MTLBuffer> subst) {
    for (auto& op : ops) {
      [enc setComputePipelineState:psos[op.p]];
      for (auto& b : op.binds)
        [enc setBuffer:(b[1] == substBuf && subst ? subst : bufs[b[1]]) offset:0 atIndex:b[0]];
      [enc setBytes:op.sizes length:sizeof(op.sizes) atIndex:30];
      if (tgBytes[op.p]) [enc setThreadgroupMemoryLength:tgBytes[op.p] atIndex:0];
      [enc dispatchThreadgroups:op.grid threadsPerThreadgroup:wgs[op.p]];
    }
  };
  auto runToken = [&](std::vector<Op>& ops, uint32_t pos, uint32_t tok, int pbuf) {
    uint32_t pv[4] = {pos, pos + 1, tok, 0};
    memcpy(bufs[pbuf].contents, pv, 16);
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc =
      [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    encodePlan(enc, ops, -1, nil);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) { fprintf(stderr, "CB ERROR: %s\n", cb.error.description.UTF8String); exit(3); }
    if (cb.status != MTLCommandBufferStatusCompleted)
      { fprintf(stderr, "CB status %lu\n", (unsigned long)cb.status); exit(3); }
    static bool once = false;
    if (!once) { once = true;
      printf("[dbg] first token GPU time %.3f ms\n", (cb.GPUEndTime - cb.GPUStartTime) * 1000); }
  };
  auto amaxTok = [&]() { return ((uint32_t*)bufs[amaxId].contents)[0]; };

  NSString* gp = [[root stringByDeletingLastPathComponent]
                  stringByAppendingPathComponent:@"goldens/goldens-a4b.json"];
  NSDictionary* goldens = [NSJSONSerialization JSONObjectWithData:
    [NSData dataWithContentsOfFile:gp] options:0 error:nil];

  int logitsId = findK(stepOps.back(), 0);
  if (mode == "dbg") {
    // embed sensitivity: run ONLY op0 (embed) with two different tokens
    int hiddenId = findK(stepOps.front(), 7);   // q6k y = k7 (A.hidden)
    {
      // chain: op1 rms → normed (k3); op2 qkvMv → qkv (k5)
      int normedId = findK(stepOps[1], 3);
      int qkvId = findK(stepOps[2], 5);
      memset(bufs[qkvId].contents, 0xFF, bufs[qkvId].length);
      std::vector<Op> l0 = { stepOps[0], stepOps[1], stepOps[2] };
      runToken(l0, 0, 1000u, stepParams);
      uint16_t* nm = (uint16_t*)bufs[normedId].contents;
      float* qk = (float*)bufs[qkvId].contents;
      printf("[dbg] normed[0..3] %04x %04x %04x %04x  qkv[0..3] %g %g %g %g\n",
             nm[0], nm[1], nm[2], nm[3], qk[0], qk[1], qk[2], qk[3]);
      printf("[dbg] qkv k-rows [4096..4099] %g %g %g %g  [6000] %g  [8000] %g\n",
             qk[4096], qk[4097], qk[4098], qk[4099], qk[6000], qk[8000]);
    }
    // layer-0 headprep = step op 3; kcache = MSL k5 (decl order)
    int kcacheId = findK(stepOps[3], 5);
    uint16_t* kc = (uint16_t*)bufs[kcacheId].contents;
    uint16_t before[4] = {kc[0], kc[1], kc[2], kc[3]};
    for (uint32_t tok : {1000u, 2000u}) {
      runToken(stepOps, 0, tok, stepParams);
      printf("[dbg] tok=%u kcache[0..3] %04x %04x %04x %04x (was %04x %04x %04x %04x)\n",
             tok, kc[0], kc[1], kc[2], kc[3], before[0], before[1], before[2], before[3]);
    }
    return 0;
  }
  if (mode == "gate") {
    memset(bufs[logitsId].contents, 0, 64);        // poke: did lm_head write?
    bool allPass = true;
    for (NSDictionary* g in goldens[@"prompts"]) {
      NSArray* in = g[@"input_ids"];
      NSArray* want = g[@"generated_ids"];
      uint32_t pos = 0;
      for (NSNumber* t in in) runToken(stepOps, pos++, t.unsignedIntValue, stepParams);
      { float* lg = (float*)bufs[logitsId].contents;
        int best = 0; for (int i = 1; i < 262144; i++) if (lg[i] > lg[best]) best = i;
        uint32_t* am = (uint32_t*)bufs[amaxId].contents;
        printf("[dbg] cpuArgmax=%d (%g)  amaxBuf=[%u,%u]\n", best, lg[best], am[0], am[1]); }
      std::vector<uint32_t> out;
      out.push_back(amaxTok());
      while ((int)out.size() < nTok && out.back() != 106 && out.back() != 1) {
        runToken(feedOps, pos++, 0, feedParams);
        out.push_back(amaxTok());
      }
      bool match = true;
      for (int i = 0; i < (int)out.size() && i < nTok; i++)
        if (i >= (int)want.count || out[i] != [want[i] unsignedIntValue]) { match = false; break; }
      allPass &= match;
      printf("[gate] %s got=[", match ? "MATCH" : "MISMATCH");
      for (auto t : out) printf("%u,", t);
      printf("]\n");
    }
    printf(allPass ? "GATE PASS\n" : "GATE FAIL\n");
    return allPass ? 0 : 2;
  }

  if (mode == "bench") {
    NSArray* in = ((NSDictionary*)goldens[@"prompts"][0])[@"input_ids"];
    uint32_t pos = 0;
    for (NSNumber* t in in) runToken(stepOps, pos++, t.unsignedIntValue, stepParams);
    // n decode tokens, GPU-side feedback (amax→embedFeed), ONE command buffer,
    // ONE serial encoder, per-token params from a pre-written ring
    std::vector<id<MTLBuffer>> ring;
    for (int i = 0; i < nTok; i++) {
      id<MTLBuffer> pb = [dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
      uint32_t pv[4] = {pos + i, pos + i + 1, 0, 0};
      memcpy(pb.contents, pv, 16);
      ring.push_back(pb);
    }
    for (int warm = 0; warm < 2; warm++) {         // warmup then measure
      double t1 = nowMs();
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc =
        [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
      for (int i = 0; i < nTok; i++) encodePlan(enc, feedOps, feedParams, ring[i]);
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      double wall = nowMs() - t1;
      double gpu = (cb.GPUEndTime - cb.GPUStartTime) * 1000;
      printf("[bench]%s %d tokens: wall %.2f ms/tok (%.1f tok/s), GPU %.2f ms/tok\n",
             warm ? "" : " (warmup)", nTok, wall / nTok, 1000 / (wall / nTok), gpu / nTok);
    }
    return 0;
  }
  // prefill plan executor: the browser's hP→hidden blit between the layer
  // pass and the final (rmsF32/lmHead/argmax) pass is NOT a dispatch and so
  // is absent from the trace — re-insert it (last 4 ops = the final pass).
  auto runPre = [&](std::vector<Op>& ops, uint32_t M) {
    size_t n = ops.size();
    int hpId = findK(ops[n - 5], 6);          // last a4btail hOut = A.hP (k6)
    int hiddenId2 = findK(ops[n - 4], 0);     // rmsF32 x = A.hidden (k0)
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc =
      [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    std::vector<Op> body(ops.begin(), ops.end() - 4), tail(ops.end() - 4, ops.end());
    encodePlan(enc, body, -1, nil);
    [enc endEncoding];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:bufs[hpId] sourceOffset:(M - 1) * 2816 * 4
          toBuffer:bufs[hiddenId2] destinationOffset:0 size:2816 * 4];
    [blit endEncoding];
    enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    encodePlan(enc, tail, -1, nil);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) { fprintf(stderr, "CB ERROR: %s\n", cb.error.description.UTF8String); exit(3); }
    return (cb.GPUEndTime - cb.GPUStartTime) * 1000;
  };
  if (mode == "gate3") {
    // prefill(prompt0) via the traced pre plan, then feed-plan generation
    NSDictionary* g = goldens[@"prompts"][0];
    NSArray* in = g[@"input_ids"];
    uint32_t M = (uint32_t)in.count;
    uint32_t pv[4] = {0, M, 0, 0};
    memcpy(bufs[preParams].contents, pv, 16);
    uint32_t* tp = (uint32_t*)bufs[preTok].contents;
    for (uint32_t i = 0; i < M; i++) tp[i] = [in[i] unsignedIntValue];
    runPre(preOps, M);
    std::vector<uint32_t> out;
    out.push_back(amaxTok());
    uint32_t pos = M;
    while ((int)out.size() < nTok && out.back() != 106 && out.back() != 1) {
      runToken(feedOps, pos++, 0, feedParams);
      out.push_back(amaxTok());
    }
    NSArray* want = g[@"generated_ids"];
    bool match = true;
    for (int i = 0; i < (int)out.size() && i < nTok; i++)
      if (i >= (int)want.count || out[i] != [want[i] unsignedIntValue]) { match = false; break; }
    printf("[gate3] %s got=[", match ? "MATCH" : "MISMATCH");
    for (auto t : out) printf("%u,", t);
    printf("]\nGATE3 %s\n", match ? "PASS" : "FAIL");
    return match ? 0 : 2;
  }
  if (mode == "prebench") {
    // 512-token batched prefill (traced grids are M=512-shaped)
    NSArray* in = ((NSDictionary*)goldens[@"prompts"][0])[@"input_ids"];
    uint32_t pv[4] = {0, 512, 0, 0};
    memcpy(bufs[preParams].contents, pv, 16);
    uint32_t* tp = (uint32_t*)bufs[preTok].contents;
    for (uint32_t i = 0; i < 512; i++) tp[i] = [in[i % in.count] unsignedIntValue];
    for (int r = 0; r < 4; r++) {
      double t1 = nowMs();
      double gpu = runPre(pre512Ops, 512);
      double wall = nowMs() - t1;
      printf("[prebench]%s M=512: %.1f ms = %.2f ms/tok (%.0f tok/s), GPU %.2f ms/tok\n",
             r ? "" : " (warmup)", wall, wall / 512, 512000 / wall, gpu / 512);
    }
    return 0;
  }
  fprintf(stderr, "unknown mode\n");
  return 1;
} }
