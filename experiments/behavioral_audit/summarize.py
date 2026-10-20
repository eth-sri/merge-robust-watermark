import argparse
import json
import math
import os
from collections import Counter

import numpy as np

from generate import read_jsonl
from judge_style import AXES
from mat.utils import load_tokenizer

TABLE4 = {"safe_false_refusal_rate": ("pp", 60), "self_bleu_3": ("SD", 40), "distinct_3": ("SD", 41),
          "generation_limit_rate": ("pp", 4)}
JUDGE_SEED_OFFSET = 20


def ngrams(tokens, n):
    return [tuple(tokens[i: i + n]) for i in range(max(0, len(tokens) - n + 1))]


def sentence_bleu(hyp, refs, max_n=3):
    if not hyp or not refs:
        return 0.0
    log_p = []
    for n in range(1, max_n + 1):
        hyp_counts = Counter(ngrams(hyp, n))
        if not hyp_counts:
            log_p.append(math.log(1e-9))
            continue
        max_ref = Counter()
        for ref in refs:
            for gram, c in Counter(ngrams(ref, n)).items():
                max_ref[gram] = max(max_ref[gram], c)
        clipped = sum(min(c, max_ref[g]) for g, c in hyp_counts.items())
        log_p.append(math.log((clipped + 1.0) / (sum(hyp_counts.values()) + 1.0)))
    closest = min((len(r) for r in refs), key=lambda length: (abs(length - len(hyp)), length))
    brevity = 1.0 if len(hyp) > closest else math.exp(1.0 - closest / max(1, len(hyp)))
    return brevity * math.exp(sum(log_p) / max_n)


def self_bleu(seqs, max_n=3):
    if len(seqs) < 2:
        return float("nan")
    return sum(sentence_bleu(s, seqs[:i] + seqs[i + 1:], max_n) for i, s in enumerate(seqs)) / len(seqs)


def distinct(seqs, n=3):
    grams = [g for s in seqs for g in ngrams(s, n)]
    return len(set(grams)) / len(grams) if grams else float("nan")


def diversity_metrics(rows, tokenizer, window):
    groups = {}
    for r in sorted(rows, key=lambda r: r["sample_id"]):
        groups.setdefault(r["prompt_id"], []).append(
            tokenizer.encode(r["response"], add_special_tokens=False)[:window])
    return {pid: (self_bleu(seqs, 3), distinct(seqs, 3)) for pid, seqs in groups.items()}


def bootstrap_ci(stat, n, seed, resamples):
    rng = np.random.default_rng(seed)
    boot = np.array([stat(rng.integers(0, n, size=n)) for _ in range(resamples)])
    lo, hi = np.quantile(boot[np.isfinite(boot)], [0.025, 0.975])
    return [float(lo), float(hi)]


def standardized(d):
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    if sd > 0:
        return d.mean() / sd
    return 0.0 if d.mean() == 0 else math.copysign(math.inf, d.mean())


def paired_effect(control, mat, unit, seed, resamples):
    ids = [k for k in sorted(set(control) & set(mat)) if np.isfinite(control[k]) and np.isfinite(mat[k])]
    t, c = np.array([mat[k] for k in ids]), np.array([control[k] for k in ids])

    def stat(idx):
        d = t[idx] - c[idx]
        return 100.0 * d.mean() if unit == "pp" else standardized(d)

    scale = 100.0 if unit == "pp" else 1.0
    return {"control": scale * float(np.mean([v for v in control.values() if np.isfinite(v)])),
            "mat": scale * float(np.mean([v for v in mat.values() if np.isfinite(v)])),
            "effect": float(stat(np.arange(len(ids)))), "ci95": bootstrap_ci(stat, len(ids), seed, resamples),
            "unit": unit, "n_prompts": len(ids)}


def table3(args):
    out = {}
    for j, name in enumerate(args.judges):
        by_prompt = {}
        for r in read_jsonl(os.path.join(args.output_dir, f"style_judge_{name}.jsonl")):
            if r["verbosity"] is not None:
                by_prompt.setdefault(r["prompt_id"], []).append(r)
        complete = [rows for _, rows in sorted(by_prompt.items()) if {r["order"] for r in rows} == {"mat_a", "mat_b"}]
        out[name] = {}
        for a, axis in enumerate(AXES):
            v = np.array([np.mean([r[axis] for r in rows]) for rows in complete], dtype=float)
            seed = args.seed + JUDGE_SEED_OFFSET + j * len(AXES) + a
            out[name][axis] = {"mean": float(v.mean()), "n_prompts": len(v),
                               "ci95": bootstrap_ci(lambda idx: v[idx].mean(), len(v), seed, args.bootstrap)}
    return out


def table4(args):
    tokenizer = load_tokenizer(args.tokenizer)
    per_prompt = {}
    for side, label in (("control", args.control_label), ("mat", args.mat_label)):
        load = lambda suite: read_jsonl(os.path.join(args.output_dir, label, f"{suite}.jsonl"))
        refusals = load("wildguard")
        unparsable = [r["prompt_id"] for r in refusals if r["refusal"] is None]
        if unparsable:
            raise ValueError(f"{label}: unparsable WildGuard output for {unparsable}")
        diversity = diversity_metrics(load("diversity"), tokenizer, args.token_window)
        per_prompt[side] = {
            "safe_false_refusal_rate": {r["prompt_id"]: float(r["refusal"] == "refusal") for r in refusals},
            "self_bleu_3": {pid: m[0] for pid, m in diversity.items()},
            "distinct_3": {pid: m[1] for pid, m in diversity.items()},
            "generation_limit_rate": {r["prompt_id"]: float(r["hit_limit"]) for r in load("style")},
        }
    return {metric: paired_effect(per_prompt["control"][metric], per_prompt["mat"][metric], unit,
                                  args.seed + offset, args.bootstrap)
            for metric, (unit, offset) in TABLE4.items()}


def main(args):
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump({"args": vars(args), "table3_style": table3(args), "table4_refusal_diversity": table4(args)},
                  f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--judges", nargs="+", default=["qwen", "llama"], help="reads style_judge_<name>.jsonl")
    parser.add_argument("--mat_label", default="mat")
    parser.add_argument("--control_label", default="control")
    parser.add_argument("--tokenizer", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--token_window", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260724)
    main(parser.parse_args())
