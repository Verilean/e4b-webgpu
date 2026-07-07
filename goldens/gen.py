#!/usr/bin/env python3
"""Golden references for the E4B WebGPU engine (M2).

Runs the SAME QAT checkpoint (./model) with transformers on CPU (deterministic) and
saves, per fixed prompt: input token ids, greedy generated ids, and for prompt 0 the
first-step logits (full f32) plus per-layer hidden-state fingerprints for layer-bisect
debugging (the hesper method).
"""
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "..", "model")

PROMPTS = [
    "Write a short poem about the sea",
    "The capital of France is",
    "Explain why the sky is blue in one sentence.",
]
MAX_NEW = 24

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, device_map="cpu"
    )
    model.eval()
    print("loaded:", type(model).__name__, file=sys.stderr)

    out = {"prompts": []}
    for pi, p in enumerate(PROMPTS):
        msgs = [{"role": "user", "content": p}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_dict=True, return_tensors="pt")
        ids = enc["input_ids"]
        with torch.no_grad():
            gen = model.generate(
                ids, max_new_tokens=MAX_NEW, do_sample=False,
                output_hidden_states=(pi == 0), return_dict_in_generate=True,
            )
        gen_ids = gen.sequences[0, ids.shape[1]:].tolist()
        rec = {
            "prompt": p,
            "input_ids": ids[0].tolist(),
            "generated_ids": gen_ids,
            "generated_text": tok.decode(gen_ids, skip_special_tokens=False),
        }
        if pi == 0:
            # first decode step: full logits + per-layer hidden fingerprints
            with torch.no_grad():
                fwd = model(ids, output_hidden_states=True)
            logits = fwd.logits[0, -1].float().numpy()
            logits.astype("<f4").tofile(os.path.join(HERE, "p0_step0_logits.bin"))
            rec["step0_argmax"] = int(logits.argmax())
            rec["step0_logits_file"] = "p0_step0_logits.bin"
            hs = fwd.hidden_states  # tuple(len = layers+1) of [1, seq, hidden]
            rec["layer_fingerprints"] = [
                {"layer": i, "meanAbs": float(h[0, -1].abs().mean()),
                 "first8": [float(x) for x in h[0, -1, :8]]}
                for i, h in enumerate(hs)
            ]
            # last-position hidden of every layer, full, for bisect
            np.stack([h[0, -1].float().numpy() for h in hs]).astype("<f4").tofile(
                os.path.join(HERE, "p0_step0_hidden_all_layers.bin"))
            rec["hidden_file"] = "p0_step0_hidden_all_layers.bin"
            rec["hidden_shape"] = [len(hs), int(hs[0].shape[-1])]
        out["prompts"].append(rec)
        print(f"[{pi}] {p!r} -> {rec['generated_text']!r}", file=sys.stderr)

    with open(os.path.join(HERE, "goldens.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("goldens.json written", file=sys.stderr)

if __name__ == "__main__":
    main()
