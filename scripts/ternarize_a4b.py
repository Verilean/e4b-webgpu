#!/usr/bin/env python3
"""Campaign 6 Phase B: offline ternarizer for the A4B GGUF's MoE expert tensors.

Reads q4_0 blocks directly (18 B = f16 scale + 16 nibble bytes, elems 0-15 in
low nibbles, 16-31 in high), dequantizes per expert tensor, ternarizes per
OUTPUT ROW (rtn: alpha = mean|w|; aa: alpha weighted by h = E[x^2] if a dump
exists), and writes a sidecar:

  model-a4b/ternary.bin      concatenated per-tensor [t2 plane | row scales f16]
  model-a4b/ternary.json     manifest {name: {off, t2Bytes, scBytes, rows, cols, relmse}}

t2 packing (matches kernels/t2 group order): each u32 covers 16 elems; shift
group sh in {0,2,4,6} holds elems [4*(sh/2) .. +3] in byte lanes -> the WGSL
unpack `(w >> sh) & 0x03030303` yields a CONTIGUOUS vec4. Values stored 0..2
(= t+1).

Usage: python3 scripts/ternarize_a4b.py [--quant rtn|aa] [--layers 0,1,...]
"""
import argparse
import json
import os
import struct

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GGUF = os.path.join(ROOT, "model-a4b", "gemma-4-26B_q4_0-it.gguf")

GGUF_MAGIC = 0x46554747
T_Q4_0 = 2


def read_gguf_index(path):
    f = open(path, "rb")
    magic, version = struct.unpack("<II", f.read(8))
    assert magic == GGUF_MAGIC
    n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

    def rstr():
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode()

    def rval(t):
        if t == 4: return struct.unpack("<I", f.read(4))[0]
        if t == 5: return struct.unpack("<i", f.read(4))[0]
        if t == 6: return struct.unpack("<f", f.read(4))[0]
        if t == 7: return struct.unpack("<B", f.read(1))[0]
        if t == 8: return rstr()
        if t == 9:
            et = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            return [rval(et) for _ in range(n)]
        if t == 10: return struct.unpack("<Q", f.read(8))[0]
        if t == 11: return struct.unpack("<q", f.read(8))[0]
        if t == 12: return struct.unpack("<d", f.read(8))[0]
        if t in (0, 1): return struct.unpack("<Bb"[t] if t == 0 else "<b", f.read(1))[0]
        if t in (2, 3): return struct.unpack("<H" if t == 2 else "<h", f.read(2))[0]
        raise ValueError(t)

    align = 32
    for _ in range(n_kv):
        k = rstr()
        t = struct.unpack("<I", f.read(4))[0]
        v = rval(t)
        if k == "general.alignment":
            align = v
    tensors = {}
    for _ in range(n_tensors):
        name = rstr()
        nd = struct.unpack("<I", f.read(4))[0]
        dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
        ttype = struct.unpack("<I", f.read(4))[0]
        off = struct.unpack("<Q", f.read(8))[0]
        tensors[name] = {"dims": dims, "type": ttype, "off": off}
    data_start = (f.tell() + align - 1) // align * align
    return f, tensors, data_start


def dequant_q4_0(raw, n_elems):
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 18)
    scale = blocks[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
    nib = blocks[:, 2:]
    lo = (nib & 0x0F).astype(np.int8) - 8
    hi = (nib >> 4).astype(np.int8) - 8
    vals = np.concatenate([lo, hi], axis=1).astype(np.float32) * scale
    return vals.reshape(-1)[:n_elems]


def pack_t2(t):
    """t: [rows, cols] in {-1,0,1}; cols % 16 == 0. Returns uint32 [rows, cols/16]
    with shift-group layout (see module docstring)."""
    r, c = t.shape
    u = (t + 1).astype(np.uint32)              # 0..2
    u = u.reshape(r, c // 16, 4, 4)            # [.., group sh/2, elem-in-group]
    # byte lane = elem-in-group j; shift = 2*group
    out = np.zeros((r, c // 16), dtype=np.uint32)
    for g in range(4):
        for j in range(4):
            out |= (u[:, :, g, j] << (2 * g + 8 * j)).astype(np.uint32)
    return out


def ternarize_rows(W, h=None, iters=5):
    """W [rows, cols] fp32 -> alpha [rows], t [rows, cols]."""
    alpha = np.abs(W).mean(axis=1, keepdims=True).clip(1e-8, None)
    t = np.clip(np.round(W / alpha), -1, 1)
    if h is not None:
        hw = h.clip(1e-12, None)[None, :]
        for _ in range(iters):
            num = (hw * W * t).sum(1, keepdims=True)
            den = (hw * t * t).sum(1, keepdims=True).clip(1e-12, None)
            alpha = (num / den).clip(1e-8, None)
            t = np.clip(np.round(W / alpha), -1, 1)
    return alpha[:, 0], t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", default="rtn", choices=["rtn", "aa"])
    ap.add_argument("--layers", default="")
    args = ap.parse_args()

    hdump = None
    hp = os.path.join(ROOT, "model-a4b", "hdiag_a4b.json")
    if args.quant == "aa":
        hdump = json.load(open(hp))
        print(f"h-diag loaded: {len(hdump)} entries")

    f, tensors, data_start = read_gguf_index(GGUF)
    sel = set(int(x) for x in args.layers.split(",")) if args.layers else None
    out = open(os.path.join(ROOT, "model-a4b", "ternary.bin"), "wb")
    manifest = {}
    off = 0
    for li in range(30):
        if sel is not None and li not in sel:
            continue
        for suffix, per_exp_rows, cols in [("ffn_gate_up_exps.weight", 1408, 2816),
                                           ("ffn_down_exps.weight", 2816, 704)]:
            name = f"blk.{li}.{suffix}"
            t = tensors[name]
            n_elems = int(np.prod(t["dims"]))
            nbytes = n_elems // 32 * 18
            f.seek(data_start + t["off"])
            W = dequant_q4_0(f.read(nbytes), n_elems)
            E = 128
            W = W.reshape(E * per_exp_rows, cols)
            h = None
            if hdump is not None:
                key = "moeIn" if "gate_up" in suffix else "geglu"
                h = np.array(hdump[f"{li}.{key}"], dtype=np.float32)
            alpha, tw = ternarize_rows(W, h)
            rel = float(((alpha[:, None] * tw - W) ** 2).sum() / (W ** 2).sum())
            t2 = pack_t2(tw)
            sc = alpha.astype(np.float16)
            out.write(t2.tobytes())
            out.write(sc.tobytes())
            manifest[name] = {"off": off, "t2Bytes": t2.nbytes, "scBytes": sc.nbytes,
                              "rows": E * per_exp_rows, "cols": cols, "relmse": rel}
            off += t2.nbytes + sc.nbytes
            print(f"{name}: relMSE {rel:.3f}  t2 {t2.nbytes/1e6:.0f}MB", flush=True)
    out.close()
    json.dump({"quant": args.quant, "tensors": manifest},
              open(os.path.join(ROOT, "model-a4b", "ternary.json"), "w"))
    print(f"sidecar: {off/1e9:.2f} GB")


if __name__ == "__main__":
    main()
