# Draft: Replication section for hesper docs/ANATOMY_OF_DECODE_REPORT.md (★review)

## Replication: Gemma-4 E4B from scratch in one morning

To test whether the prescriptions generalize, we replicated the webml experiment
on a model the reference engine was not built for: **Gemma-4 E4B QAT-mobile**
(3.36 GB safetensors, SRQ static activation quantization, int4/int8/int2 mixed,
per-layer-input (PLE) blocks, KV-shared layers). New minimal repo, engine written
from scratch (webml read as reference only), kernels as data, transformers-CPU
goldens, headless-Chrome harness. Author time was logged as part of the experiment.

### Results (M4 Max 48 GB, 546 GB/s peak; decode, single stream)

| engine | ms/token | tok/s | eff. BW (2.20 GB/token) |
|---|---|---|---|
| llama.cpp (E4B QAT q4_0, llama-bench tg64) | 9.76 | **102.4** | 225 GB/s |
| webml engine, unmodified + E4B repo id | 8.10 | **123.5** | 272 GB/s |
| **ours (scratch, WGSL, 11 dispatches/layer)** | 11.11 | **90.0** | 198 GB/s |

Wall clock: **M0→M3 (repo → token-exact bring-up) = 36 min; M4 (15→90 tok/s, 6×)
= 70 min; total 1 h 46 min.** The bring-up passed its golden gate on the FIRST
execution — the seconds-TAT loop (hot-reload kernels, one-command gate) carried
the same leverage it did in the original experiment.

### What held, what didn't

- **P1 (webml swap runs E4B): held.** The unmodified engine ran E4B at 123.5 tok/s
  — config-driven kernels generalize; the swap became our stretch target/oracle.
- **P2 (llama.cpp 60-85 tok/s): missed low** — 102.4. Recorded, not rationalized.
- **P3 (beat llama.cpp): NOT reached** — 90.0 = 88%. The stretch (123.5) not reached.
- **P4 (author time ≤ 2-3 days): beaten by an order of magnitude** (1 h 46 min).

### New findings the original experiment did not surface

1. **Token-exactness vs a CPU oracle is unattainable for SRQ checkpoints, by
   mechanism**: static per-layer activation grids (steps 0.3-0.98) turn 1-ulp
   implementation differences into full quantization steps at grid boundaries.
   Drift stays bounded (SRQ re-snaps each layer) but flips near-tie tokens. The
   gate must be amended: (a) ≥1 prompt token-exact, (b) cross-engine text
   agreement, (c) layer-path ratios ≈ 1.000, (d) coherent/semantically-equal text.
2. **A serialized-dispatch cost model** explains browser decode wall time:
   `fenced dispatch ≈ bytes / streaming-rate + 9-33 µs fixed`, and greedy decode
   is a fully linear dependency chain — so DISPATCH COUNT, not kernel quality
   alone, dominates once kernels stream near peak (ours: 484 GB/s pure, 89% of
   peak). This is the quantitative form of the report's fat-kernel prescription;
   webml's 316-op graph sits exactly on the same model.
3. **The token gate alone can mask real bugs** (a 10× matvec under-dispatch
   passed the p1 token gate). Condition (c) — layer-fingerprint ratios — caught
   it; a bit-exact differential mode (fused-vs-explicit xq comparison) caught a
   lost-edit that silently skipped a fusion. Golden gates need a numeric
   condition, not just token equality.
4. **GPU-side greedy feedback** (argmax fed to the next token's embedding on-GPU,
   tokens read back in chunks) removed ~1.6 ms/token of CPU sync — the single
   largest M4 step. The CPU should not be in the decode loop at all.

### Honest residual gap

llama.cpp's native-Metal int4 kernels stream at ~400+ GB/s where our WGSL int4
matvec saturates at ~335 (the int8-activation second load stream + unpack ALU);
five shape/staging variants were tried and rejected with measurements. Closing
the last 12% needs either fewer than ~11 fenced dispatches per layer or an int4
inner loop the WGSL→Metal path currently does not produce.
