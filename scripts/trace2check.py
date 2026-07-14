#!/usr/bin/env python3
"""Convert a raw browser trace (posted by the resident harness's `trace`
command to metal/out/trace.json) into a wgsl-check manifest:

  checkout/
    k<shaderId>.wgsl     template-expanded WGSL as the GPU saw it
    manifest.json        one dispatch per unique (pipe, grid, binding-shape)

Buffer contents mirrored by the trace shim (small writeBuffer targets, i.e.
params UBOs) are attached as `values` so the checker can evaluate runtime
guards like `if (i >= mprm[1] * N)`.

Usage: python3 scripts/trace2check.py [trace.json] [outdir]
Then:  hesper$ ./.lake/build/bin/wgsl-check <outdir>/manifest.json
"""
import base64
import json
import os
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
trace_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "metal", "out", "trace.json")
outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "metal", "out", "checkout")
os.makedirs(outdir, exist_ok=True)

t = json.load(open(trace_path))
pipes = {p["id"]: p for p in t["pipes"]}
shaders = {s["id"]: s for s in t["shaders"]}
contents = t.get("contents", {})

for sid, s in shaders.items():
    with open(os.path.join(outdir, f"k{sid}.wgsl"), "w") as f:
        f.write(s["code"])

def u32s(b64, cap=64):
    raw = base64.b64decode(b64)
    n = min(len(raw) // 4, cap)
    return list(struct.unpack(f"<{n}I", raw[: n * 4]))

seen = {}
dispatches = []
for op in t["ops"]:
    pipe = pipes.get(op["pipe"])
    if pipe is None or pipe["shaderId"] < 0:
        continue
    bindings = []
    for bg in op.get("bgs", []):
        for e in bg["entries"]:
            b = {"group": bg["group"], "binding": e["binding"],
                 "bytes": e["bufSize"] - e.get("offset", 0)}
            key = str(e["buf"])
            if key in contents and e["bufSize"] <= 256:
                b["values"] = u32s(contents[key])
            bindings.append(b)
    sig = json.dumps([op["pipe"], op["grid"],
                      [(b["group"], b["binding"], b["bytes"],
                        tuple(b.get("values", []))) for b in bindings]],
                     default=str)
    if sig in seen:
        continue
    seen[sig] = True
    dispatches.append({
        "kernel": f"k{pipe['shaderId']}.wgsl",
        "entry": pipe.get("entryPoint") or "main",
        "plan": op.get("plan", ""),
        "grid": op["grid"],
        "bindings": bindings,
    })

json.dump({"dispatches": dispatches},
          open(os.path.join(outdir, "manifest.json"), "w"), indent=1)
print(f"{len(t['ops'])} ops -> {len(dispatches)} unique dispatches, "
      f"{len(shaders)} kernels -> {outdir}")
