#!/usr/bin/env python3
"""Convert a hesper JSTrace package (DG_TRACE_JS) into a wgsl-check manifest.

Differences from trace2check.py (raw browser shim): kernels are already
files (k<hash>.wgsl), dispatches carry (bindingName, uid) with sizes in
buffers.json, and params contents come from 'w' hex events (last write wins,
as the checker models the state at dispatch time conservatively).

Usage: python3 scripts/dgtrace2check.py <tracedir> [outfile]
Then:  wgsl-check <outfile>
"""
import json
import os
import re
import struct
import sys

d = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(d, "check-manifest.json")

bufs = json.load(open(os.path.join(d, "buffers.json")))
ops = []
for l in open(os.path.join(d, "ops.jsonl")):
    if not l.strip():
        continue
    ops.append(json.loads(re.sub(r'"k":(\d+)', r'"k":"\1"', l)))

# last-known contents per uid (from hex writes), as u32 lists (cap 64)
values = {}
for o in ops:
    if o["t"] == "w" and "hex" in o and o["o"] == 0:
        raw = bytes.fromhex(o["hex"])[:256]
        values[str(o["u"])] = list(struct.unpack(f"<{len(raw)//4}I", raw[: len(raw) // 4 * 4]))

seen = {}
dispatches = []
for o in ops:
    if o["t"] != "d":
        continue
    bindings = []
    for i, (name, uid) in enumerate(o["b"]):
        b = {"name": name, "bytes": int(bufs.get(str(uid), 0))}
        if str(uid) in values:
            b["values"] = values[str(uid)]
        bindings.append(b)
    sig = json.dumps([o["k"], o["g"], [(b["name"], b["bytes"], tuple(b.get("values", []))) for b in bindings]])
    if sig in seen:
        continue
    seen[sig] = True
    dispatches.append({"kernel": f"k{o['k']}.wgsl", "entry": o.get("n", "main"),
                       "grid": o["g"], "bindings": bindings})

json.dump({"dispatches": dispatches}, open(out, "w"))
print(f"{len(ops)} events -> {len(dispatches)} unique dispatches -> {out}")
