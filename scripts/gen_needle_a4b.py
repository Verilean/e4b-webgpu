#!/usr/bin/env python3
"""Needle goldens for the A4B long-context gate (ctx 2048, 3 depths x 2
phrasings — the second phrasing is the mechanism-robustness check)."""
import glob, json, os
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "model-a4b"))

text = ""
for g in [os.path.expanduser("~/git/verilean/hesper/docs/*.md"),
          os.path.expanduser("~/git/verilean/hesper/*.md")]:
    for f in sorted(glob.glob(g)):
        text += open(f, encoding="utf-8", errors="ignore").read() + "\n\n"
filler = tok(text, add_special_tokens=False).input_ids

QS = ["\n\nQuestion: What is the secret access code? Answer: The secret access code is",
      "\n\nBefore we continue: recall the access code mentioned earlier. The code is"]
CTX = int(os.environ.get("CTX", 2048))
cases = []
for qi, q in enumerate(QS):
    for d in (0.1, 0.5, 0.9):
        code = f"ZX{int(d*100):02d}Q{qi}"
        fact = f"\n\nIMPORTANT: The secret access code is {code}. Remember it.\n\n"
        fids = tok(fact, add_special_tokens=False).input_ids
        qids = tok(q, add_special_tokens=False).input_ids
        body = CTX - len(fids) - len(qids) - 1
        pos = int(body * d)
        ids = [tok.bos_token_id] + filler[:pos] + fids + filler[pos:body] + qids
        cases.append({"depth": d, "phrasing": qi, "code": code,
                      "code_ids": tok(f" {code}", add_special_tokens=False).input_ids,
                      "input_ids": ids})
json.dump({"cases": cases}, open(os.path.join(ROOT, "goldens", os.environ.get("OUT", "needle-a4b.json")), "w"))
print(f"{len(cases)} cases, ctx={CTX}")
