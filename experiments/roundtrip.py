#!/usr/bin/env python
"""
Embed -> unembed round-trip experiment, v2: with layerwise logit-lens probing.

For each word in words.json (plus each model's BOS/EOS/PAD/UNK special tokens —
no holdout, everything is analysis):

  1. tokenize the word fresh, with no context (add_special_tokens=False)
  2. one full forward pass; hooks capture the embedding-layer output AND every
     decoder layer's output at every token position
  3. at each probe point (embedding, a sample of intermediate layers up to and
     including the penultimate and final layers), push the residual vector
     through the unembedding head two ways: raw, and with the model's final
     norm applied first (logit-lens convention)
  4. record top-k tokens + probs per probe, plus (a) the rank of the input
     token itself ("self" — did it round-trip?) and (b) the rank of the word's
     actual next token where one exists ("next" — is it becoming a next-token
     predictor?). The model's true output logits are recorded as probe "final_logits".

No sampling anywhere, so everything is deterministic.

Outputs in --outdir:
  meta.json        model facts (tied? norm module, layer count, probe layers, ...)
  results.json     full records incl. top-10 per probe per variant
  results.csv      flat one-row-per-(token, probe) summary
  activations.npz  captured residual vectors (fp32), key = "r{row}_{probe}"
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_final_norm(model):
    base = model.base_model
    for attr in ("norm", "ln_f", "final_layer_norm", "final_layernorm"):
        mod = getattr(base, attr, None)
        if mod is not None:
            return attr, mod
    return None, None


def find_layer_list(model):
    base = model.base_model
    for attr in ("layers", "h", "blocks"):
        mods = getattr(base, attr, None)
        if mods is not None and isinstance(mods, torch.nn.ModuleList):
            return attr, mods
    raise RuntimeError("could not find decoder layer list")


def pick_probe_layers(n_layers, max_probes=12):
    """1-indexed decoder layers to unembed at. Always 1, 2, penultimate, final;
    for deep models also every other layer over the last ten — the rank
    trajectory moves fastest near the end (user request 2026-07-05)."""
    if n_layers <= max_probes:
        return list(range(1, n_layers + 1))
    idxs = {1, 2, n_layers - 1, n_layers}
    for i in range(1, max_probes - 3):
        idxs.add(max(1, round(i * n_layers / (max_probes - 3))))
    idxs.update(range(max(1, n_layers - 9), n_layers + 1, 2))
    return sorted(idxs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--words", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "words.json"))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    words = json.load(open(args.words))["words"]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.eval()

    emb_module = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    norm_name, final_norm = find_final_norm(model)
    layers_attr, layers = find_layer_list(model)
    n_layers = len(layers)
    probe_layers = pick_probe_layers(n_layers)

    tied_cfg = getattr(model.config, "tie_word_embeddings", None)
    tied_actual = emb_module.weight.data_ptr() == lm_head.weight.data_ptr()

    # cache fp32 copies once; per-call casting of a 152k x 8k matrix is slow
    W32 = lm_head.weight.detach().float()
    b32 = lm_head.bias.detach().float() if lm_head.bias is not None else None

    meta = {
        "model": args.model,
        "script_version": 3,
        "dtype": args.dtype,
        "vocab_size_config": getattr(model.config, "vocab_size", None),
        "vocab_size_tokenizer": len(tokenizer),
        "hidden_size": model.config.hidden_size,
        "n_layers": n_layers,
        "probe_layers": probe_layers,
        "layers_attr": layers_attr,
        "tie_word_embeddings_config": tied_cfg,
        "tied_weights_actual_same_tensor": tied_actual,
        "final_norm_attr": norm_name,
        "final_norm_class": type(final_norm).__name__ if final_norm is not None else None,
        "lm_head_has_bias": lm_head.bias is not None,
        "special_tokens": {
            "bos": [tokenizer.bos_token, tokenizer.bos_token_id],
            "eos": [tokenizer.eos_token, tokenizer.eos_token_id],
            "pad": [tokenizer.pad_token, tokenizer.pad_token_id],
            "unk": [tokenizer.unk_token, tokenizer.unk_token_id],
        },
        "torch_version": torch.__version__,
    }

    captured = {}

    def make_hook(key):
        def hook(module, inp, out):
            captured[key] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    emb_module.register_forward_hook(make_hook("emb"))
    for i, layer in enumerate(layers):
        layer.register_forward_hook(make_hook(f"layer_{i + 1}"))

    def unembed(x):
        """x: [d] in model dtype -> (raw_logits, normed_logits) in fp32."""
        raw = F.linear(x.float(), W32, b32)
        if final_norm is not None:
            with torch.no_grad():
                xn = final_norm(x.unsqueeze(0)).squeeze(0)
            normed = F.linear(xn.float(), W32, b32)
        else:
            normed = None
        return raw, normed

    def topk_and_ranks(logits, self_id, next_id, k, final1_id=None):
        probs = F.softmax(logits, dim=-1)
        top = torch.topk(probs, k)
        out = {
            "top": [
                {
                    "token_id": int(i),
                    "token": tokenizer.convert_ids_to_tokens(int(i)),
                    "decoded": tokenizer.decode([int(i)]),
                    "prob": float(p),
                }
                for p, i in zip(top.values, top.indices)
            ],
            "self_rank": int((logits > logits[self_id]).sum()) + 1,
        }
        if next_id is not None:
            out["next_rank"] = int((logits > logits[next_id]).sum()) + 1
        if final1_id is not None:
            out["final1_rank"] = int((logits > logits[final1_id]).sum()) + 1
        return out

    # build the run list: words from the list, then per-model special tokens
    entries = []
    for w in words:
        ids = tokenizer(w["text"], add_special_tokens=False)["input_ids"]
        entries.append({**w, "ids": ids})
    for label in ("bos", "eos", "pad", "unk"):
        tid = getattr(tokenizer, f"{label}_token_id")
        tok = getattr(tokenizer, f"{label}_token")
        if tid is not None:
            entries.append({"text": tok, "category": f"special_{label}", "ids": [tid]})

    rows = []
    act_store = {}
    for entry in entries:
        ids = entry["ids"]
        if len(ids) == 0:
            rows.append({
                "word": entry["text"], "category": entry["category"],
                "n_tokens": 0, "note": "tokenizes to zero tokens",
            })
            continue
        captured.clear()
        with torch.no_grad():
            out = model(torch.tensor([ids], dtype=torch.long))
        final_logits = out.logits[0].float()  # [seq, vocab] — the model's true output
        final_top1 = final_logits.argmax(-1)  # [seq] — what the model actually predicts next

        for pos, tid in enumerate(ids):
            next_id = int(ids[pos + 1]) if pos + 1 < len(ids) else None
            f1_id = int(final_top1[pos])
            row = {
                "word": entry["text"],
                "category": entry["category"],
                "n_tokens": len(ids),
                "pos": pos,
                "token_id": int(tid),
                "token": tokenizer.convert_ids_to_tokens(int(tid)),
                "token_decoded": tokenizer.decode([int(tid)]),
                "next_token_id": next_id,
                "final_top1_id": f1_id,
                "final_top1_token": tokenizer.convert_ids_to_tokens(f1_id),
                "probes": {},
            }
            row_idx = len(rows)
            for probe in ["emb"] + [f"layer_{i}" for i in probe_layers]:
                x = captured[probe][0][pos]
                raw, normed = unembed(x)
                rec = {"raw": topk_and_ranks(raw, int(tid), next_id, args.topk, f1_id)}
                if normed is not None:
                    rec["normed"] = topk_and_ranks(normed, int(tid), next_id, args.topk, f1_id)
                if probe == "emb":
                    rec["l2_norm"] = float(x.float().norm())
                row["probes"][probe] = rec
                act_store[f"r{row_idx}_{probe}"] = x.float().numpy()
            row["probes"]["final_logits"] = {
                "raw": topk_and_ranks(final_logits[pos], int(tid), next_id, args.topk, f1_id)
            }
            rows.append(row)
        print(f"done: {entry['category']:22s} {entry['text']!r} -> {len(ids)} token(s)", flush=True)

    json.dump(meta, open(os.path.join(args.outdir, "meta.json"), "w"), indent=2)
    json.dump(rows, open(os.path.join(args.outdir, "results.json"), "w"), indent=2, ensure_ascii=False)
    np.savez_compressed(os.path.join(args.outdir, "activations.npz"), **act_store)

    csv_fields = [
        "word", "category", "n_tokens", "pos", "token_id", "token", "next_token_id",
        "final_top1_token", "probe",
        "raw_top1", "raw_top1_prob", "raw_self_rank", "raw_next_rank", "raw_final1_rank",
        "normed_top1", "normed_top1_prob", "normed_self_rank", "normed_next_rank", "normed_final1_rank",
        "raw_top5", "normed_top5",
    ]
    with open(os.path.join(args.outdir, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        wr.writeheader()
        for row in rows:
            for probe, rec in row.get("probes", {}).items():
                flat = {k: row.get(k) for k in csv_fields if k in row}
                flat["probe"] = probe
                for name in ("raw", "normed"):
                    r = rec.get(name)
                    if r:
                        flat[f"{name}_top1"] = r["top"][0]["token"]
                        flat[f"{name}_top1_prob"] = round(r["top"][0]["prob"], 6)
                        flat[f"{name}_self_rank"] = r["self_rank"]
                        flat[f"{name}_next_rank"] = r.get("next_rank")
                        flat[f"{name}_final1_rank"] = r.get("final1_rank")
                        flat[f"{name}_top5"] = ";".join(f"{t['token']}:{t['prob']:.4f}" for t in r["top"][:5])
                wr.writerow(flat)

    n_ok = sum(1 for r in rows if r.get("n_tokens", 0) > 0)
    print(f"\nwrote {len(rows)} rows ({n_ok} token rows) x {len(probe_layers) + 2} probes to {args.outdir}")
    print(f"tied(config)={tied_cfg} tied(actual)={tied_actual} norm={norm_name} n_layers={n_layers} probes={probe_layers}")


if __name__ == "__main__":
    main()
