#!/usr/bin/env python3
"""Minimal GGUF v3 inspector (no deps): tensor table + kv metadata summary."""
import struct, sys, collections

GGML_TYPES = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",9:"Q8_1",
              10:"Q2_K",11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K",15:"Q8_K",30:"BF16"}
KV_FMT = {4:"<I",5:"<i",6:"<f",10:"<Q",11:"<q",12:"<d",7:"<?"}

def rd_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode()

def rd_val(f, t):
    if t == 8: return rd_str(f)
    if t == 9:  # array
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        return [rd_val(f, et) for _ in range(n)] if n < 4096 else (f.seek_skip(et, n) if False else [rd_val(f, et) for _ in range(n)])
    if t == 0 or t == 1: return struct.unpack("<B" if t==0 else "<b", f.read(1))[0]
    if t in (2,3): return struct.unpack("<H" if t==2 else "<h", f.read(2))[0]
    fmt = KV_FMT[t]
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

f = open(sys.argv[1], "rb")
magic, ver = f.read(4), struct.unpack("<I", f.read(4))[0]
nt, nkv = struct.unpack("<QQ", f.read(16))
print(f"magic={magic} v{ver} tensors={nt} kv={nkv}")
align = 32
for _ in range(nkv):
    k = rd_str(f)
    t = struct.unpack("<I", f.read(4))[0]
    v = rd_val(f, t)
    if k == "general.alignment": align = v
    if isinstance(v, list):
        if len(v) < 8: print(f"  {k} = {v}")
        else: print(f"  {k} = [{len(v)} items]")
    elif not isinstance(v, str) or len(str(v)) < 80:
        print(f"  {k} = {v}")
infos = []
for _ in range(nt):
    name = rd_str(f)
    nd = struct.unpack("<I", f.read(4))[0]
    dims = struct.unpack(f"<{nd}Q", f.read(8*nd))
    ty = struct.unpack("<I", f.read(4))[0]
    off = struct.unpack("<Q", f.read(8))[0]
    infos.append((name, dims, GGML_TYPES.get(ty, ty), off))
data_start = (f.tell() + align - 1) // align * align
print(f"data_start={data_start} align={align}")
bytypes = collections.Counter(i[2] for i in infos)
print("type counts:", dict(bytypes))
pat = collections.OrderedDict()
for name, dims, ty, off in infos:
    import re
    key = re.sub(r"\d+", "N", name)
    if key not in pat: pat[key] = (name, dims, ty)
print(f"--- {len(pat)} unique tensor patterns:")
for key, (name, dims, ty) in pat.items():
    print(f"  {key:48s} {str(dims):24s} {ty}")
