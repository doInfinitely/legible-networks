#!/usr/bin/env python3
"""Exact unit-net transplant of Gemma 3 (gate G1 of the pilot pre-registration).

Applies the transplant law from explainable_ai/qwen to Gemma 3's architecture:

- v -> o, up -> down: per-row scales fold exactly into consumer columns.
- q, k: a per-matrix scalar is absorbed by that matrix's RMSNorm (uniform scale
  plus eps -> eps/c^2).
- o, down: each writes through its own RMSNorm, so a per-matrix scalar is
  absorbed the same way. Unlike Qwen, no global stream scale is needed.
- gate: GELU is not homogeneous, so scale stops here as a per-unit temperature.
- embedding: a scalar is absorbed by the embedding's constant multiplier.
- head: untied from the embedding, with a per-class temperature.

No reader model is involved. Usage:
    python gemma_fold.py --split <splits/pair100.json> --out results/g1_base.json
"""
import argparse
import copy
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "google/gemma-3-270m-it"
TEMPLATE = "{question} I think the answer is {guess} but I'm really not sure."


def row_scales(W):
    """Per-row max of positive sum, largest negative magnitude and 1 (no inflation)."""
    pos = W.clamp(min=0).sum(1)
    neg = (-W.clamp(max=0)).amax(1)
    return torch.maximum(torch.maximum(pos, neg), torch.ones_like(pos))


class TemperedActivation(torch.nn.Module):
    """act(z) with a recorded per-unit temperature: act(beta * z)."""

    def __init__(self, act, beta):
        super().__init__()
        self.act = act
        self.register_buffer("beta", beta)

    def forward(self, z):
        return self.act(z * self.beta)


class TemperedHead(torch.nn.Module):
    """Output head with legal rows and a per-class temperature."""

    def __init__(self, weight, gain):
        super().__init__()
        self.weight = torch.nn.Parameter(weight)
        self.register_buffer("gain", gain)

    def forward(self, h):
        return (h @ self.weight.T) * self.gain


def scale_through_norm_(W, norm):
    """Divide W by one scalar and make the RMSNorm reading it exactly invariant."""
    c = float(row_scales(W).max())
    W.div_(c)
    norm.eps = norm.eps / (c * c)
    return c


@torch.no_grad()
def transplant(model):
    cfg = model.config
    n_rep = cfg.num_attention_heads // cfg.num_key_value_heads
    head_dim = cfg.head_dim
    info = {"attention_scalars": [], "writer_scalars": [], "beta": []}

    for layer in model.model.layers:
        sa, mlp = layer.self_attn, layer.mlp

        # v -> o columns; one key/value head feeds n_rep query heads
        s = row_scales(sa.v_proj.weight)
        sa.v_proj.weight.div_(s[:, None])
        s_exp = s.view(cfg.num_key_value_heads, head_dim).repeat_interleave(n_rep, dim=0).flatten()
        sa.o_proj.weight.mul_(s_exp[None, :])

        # up -> down columns: enters the gated product linearly
        s = row_scales(mlp.up_proj.weight)
        mlp.up_proj.weight.div_(s[:, None])
        mlp.down_proj.weight.mul_(s[None, :])

        info["attention_scalars"] += [scale_through_norm_(sa.q_proj.weight, sa.q_norm),
                                      scale_through_norm_(sa.k_proj.weight, sa.k_norm)]
        info["writer_scalars"] += [scale_through_norm_(sa.o_proj.weight, layer.post_attention_layernorm),
                                   scale_through_norm_(mlp.down_proj.weight, layer.post_feedforward_layernorm)]

        # gate: scale stops at the non-homogeneous activation
        beta = row_scales(mlp.gate_proj.weight)
        mlp.gate_proj.weight.div_(beta[:, None])
        mlp.act_fn = TemperedActivation(mlp.act_fn, beta.clone())
        info["beta"].append(beta)

    # untie the head first: it reads after the final norm and keeps its own scale
    emb = model.model.embed_tokens
    head_w = model.lm_head.weight.data.clone()
    gain = row_scales(head_w)
    model.lm_head = TemperedHead(head_w / gain[:, None], gain.clone())

    c = float(row_scales(emb.weight).max())
    emb.weight.data = emb.weight.data / c
    emb.embed_scale = emb.embed_scale * c
    info["embed_scalar"] = c
    info["head_gain"] = gain
    return info


def census(model):
    """Fraction of 2-D weight entries in legal rows, plus the worst row seen."""
    total = legal = 0
    worst_pos = worst_neg = 0.0
    by_family = {}
    for name, p in model.named_parameters():
        if p.dim() != 2:
            continue
        pos = p.clamp(min=0).sum(1)
        neg = (-p.clamp(max=0)).amax(1)
        ok = (pos <= 1 + 1e-4) & (neg <= 1 + 1e-4)
        n_legal = int(ok.sum()) * p.shape[1]
        total += p.numel()
        legal += n_legal
        worst_pos, worst_neg = max(worst_pos, float(pos.max())), max(worst_neg, float(neg.max()))
        fam = name.split(".")[-2]
        a, b = by_family.get(fam, (0, 0))
        by_family[fam] = (a + n_legal, b + p.numel())
    return {"legal_fraction": legal / total, "weights": total, "worst_pos_sum": worst_pos,
            "worst_neg": worst_neg, "by_family": {k: a / b for k, (a, b) in by_family.items()}}


def prompts(tok, split):
    facts = json.load(open(split))["heldout"]
    users = []
    for r in facts:
        users += [r["question"], TEMPLATE.format(question=r["question"], guess=r["correct"]),
                  TEMPLATE.format(question=r["question"], guess=r["swap"])]
    out = []
    for u in users:
        text = tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
        out.append(tok(text + "The answer is", add_special_tokens=False, return_tensors="pt").input_ids)
    return out


@torch.no_grad()
def compare(model, ref, inputs):
    """Argmax agreement and KL at every position, plus at the final position."""
    agree = n = agree_last = 0
    kl_sum, kl_max, diff_max = 0.0, 0.0, 0.0
    for ids, r in zip(inputs, ref):
        t = model(input_ids=ids).logits[0].double()
        r = r.double()
        agree += int((t.argmax(-1) == r.argmax(-1)).sum())
        agree_last += int(t[-1].argmax() == r[-1].argmax())
        n += t.shape[0]
        lr, lt = r.log_softmax(-1), t.log_softmax(-1)
        kl = (lr.exp() * (lr - lt)).sum(-1)
        kl_sum += float(kl.sum())
        kl_max = max(kl_max, float(kl.max()))
        diff_max = max(diff_max, float((t - r).abs().max()))
    return {"prompts": len(inputs), "positions": n, "argmax_agreement": agree / n,
            "final_position_agreement": agree_last / len(inputs), "mean_kl": kl_sum / n,
            "max_kl": kl_max, "max_logit_diff": diff_max}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="eager").eval()
    inputs = prompts(tok, args.split)
    with torch.no_grad():
        ref = [model(input_ids=ids).logits[0].clone() for ids in inputs]
    before = census(model)

    folded = copy.deepcopy(model)
    # deepcopy keeps the tie between head and embedding, so transplant unties it
    info = transplant(folded)
    after = census(folded)
    result = compare(folded, ref, inputs)

    betas = torch.cat(info["beta"])
    report = {
        "gate": "G1",
        "model": args.model,
        "thresholds": {"legal_fraction": 1.0, "argmax_agreement": 0.9999, "mean_kl": 1e-6},
        "census_before": before, "census_after": after, "function": result,
        "temperatures": {
            "gate_beta_count": int(betas.numel()),
            "gate_beta_min_median_max": [float(betas.min()), float(betas.median()), float(betas.max())],
            "head_gain_count": int(info["head_gain"].numel()),
            "head_gain_median_max": [float(info["head_gain"].median()), float(info["head_gain"].max())],
        },
        "scalars": {
            "embed": info["embed_scalar"],
            "attention_max": max(info["attention_scalars"]),
            "writer_max": max(info["writer_scalars"]),
            "count": len(info["attention_scalars"]) + len(info["writer_scalars"]) + 1,
        },
    }
    report["pass"] = (after["legal_fraction"] == 1.0 and result["argmax_agreement"] >= 0.9999
                      and result["mean_kl"] <= 1e-6)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
