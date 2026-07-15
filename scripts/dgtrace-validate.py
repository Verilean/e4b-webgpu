#!/usr/bin/env python3
"""Trace completeness validator: flag buffers that are READ by dispatches but never
WRITTEN by any event in the whole ops stream and are not static weights.

Catches invisible-writer holes (e.g. hand-MSL kernels bypassing the Dawn-level
trace): a replayer/engine would consume stale snapshot data for such buffers.

Heuristics for "written":
  - bound to a dispatch as a read_write storage buffer -> we can't see access mode
    from ops.jsonl alone, so we parse each kernel's WGSL for var<storage, read_write>
    binding positions and use the dispatch's bind ORDER to classify each bind.
  - 'w' events (writeBuffer uploads).
"Static" allowlist: uids appearing in tensors.json (weights) or written by 'w'
before the first mark.
Usage: dgtrace-validate.py <tracedir>
"""
import json, re, sys, os

d = sys.argv[1]
ops = [json.loads(l) for l in open(os.path.join(d, "ops.jsonl"))]

# kernel hash -> list of (binding_index, is_read_write) from WGSL
kmode = {}
def kernel_modes(khash):
    if khash in kmode: return kmode[khash]
    path = os.path.join(d, f"k{khash}.wgsl")
    modes = {}
    if os.path.exists(path):
        src = open(path).read()
        for m in re.finditer(r"@binding\((\d+)\)\s*\nvar<storage,\s*(read_write|read)>", src):
            modes[int(m.group(1))] = m.group(2) == "read_write"
    kmode[khash] = modes
    return modes

weights = set()
tj = os.path.join(d, "tensors.json")
if os.path.exists(tj):
    t = json.load(open(tj))
    for v in (t.values() if isinstance(t, dict) else t):
        if isinstance(v, dict) and "uid" in v: weights.add(v["uid"])
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, int): weights.add(x)

written, read_only_reads = set(), {}
for o in ops:
    t = o.get("t")
    if t == "w":
        written.add(o.get("uid"))
    elif t == "d":
        modes = kernel_modes(str(o.get("k")))
        for i, (name, uid) in enumerate(o.get("b", [])):
            if modes.get(i, False):
                written.add(uid)
            else:
                read_only_reads.setdefault(uid, []).append((name, str(o.get("k"))[:10]))

bad = []
for uid, uses in read_only_reads.items():
    if uid in written or uid in weights: continue
    bad.append((uid, len(uses), uses[0]))

bad.sort(key=lambda x: -x[1])
print(f"buffers read-but-never-written (non-weight): {len(bad)}")
for uid, n, (name, k) in bad:
    print(f"  uid={uid} reads={n} first as '{name}' in k{k}...")
sys.exit(1 if bad else 0)
