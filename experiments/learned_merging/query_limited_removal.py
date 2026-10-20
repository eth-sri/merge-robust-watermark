import argparse
import json
import os

import numpy as np
import torch

from mat.config import TrainConfig
from mat.evaluation import detect_batched, sample_from_prefixes, web_texts
from mat.utils import load_tokenizer
from mat.watermarks import build_watermark
from regional_merge import RegionalMerge, heldout_nll, utility_sequences


def lower_group_direction(q, j):
    d = np.full(len(q), q[j] / (q.sum() - q[j]))
    d[j] = -1.0
    return d / np.sqrt(np.dot(q, d ** 2))


def max_step(a, d):
    return min((1.0 - x) / dx if dx > 0 else x / -dx for x, dx in zip(a, d) if dx != 0)


def main(args):
    trace_path = os.path.join(args.output_dir, "search_trace.json")
    if os.path.exists(trace_path):
        print(f"{trace_path} exists, skipping")
        return
    with open(os.path.join(args.start, "coefficients.json")) as f:
        start = json.load(f)
    torch.manual_seed(args.seed)
    tokenizer = load_tokenizer(start["base"])
    _, heldout = utility_sequences(tokenizer, **start["utility_data"])
    rm = RegionalMerge(start["base"], start["wm_model"], start["partner"])
    groups, q = rm.groups, rm.wm_energy
    a = np.array([start["wm_coefficients"][g] for g in groups])
    b = np.array([start["partner_coefficients"][g] for g in groups])
    assert abs(np.dot(q, a) - start["wm_weight"]) < 1e-5, "start does not match the global mixture"

    def utility(wm_coefficients):
        rm.apply(wm_coefficients, b)
        return heldout_nll(lambda ids: rm.model(ids, use_cache=False).logits, heldout)

    watermark = build_watermark(TrainConfig.from_yaml(args.watermark_config).watermark, tokenizer)

    def query(prefixes, seed):
        torch.manual_seed(seed)
        completions = sample_from_prefixes(rm.model, tokenizer, prefixes, args.max_new_tokens,
                                           batch_size=len(prefixes))
        return detect_batched(watermark, completions)[0]

    n = args.prompts_per_round
    probes = web_texts("en", tokenizer, n * len(args.round_steps), args.prefix_length + args.max_new_tokens,
                       args.prefix_length, skip=args.probe_skip)

    start_nll = utility(a)
    rounds, n_scored = [], 0
    for r, nominal in enumerate(args.round_steps, start=1):
        directions = [lower_group_direction(q, j) for j in range(len(groups))]
        step = min(nominal, 0.95 * min(max_step(a, d) for d in directions))
        assert step > 1e-8, "no feasible move"
        candidates = [("current", a)] + [(f"lower_{g}", a + step * d) for g, d in zip(groups, directions)]
        rows = []
        for name, coefficients in candidates:
            nll = utility(coefficients)
            increase = {domain: nll[domain] - start_nll[domain] for domain in nll}
            z = query(probes[(r - 1) * n: r * n], args.seed + 100 * r)
            n_scored += len(z)
            rows.append(dict(candidate=name, wm_coefficients=dict(zip(groups, coefficients.tolist())),
                             heldout_nll=nll, nll_increase=increase, z_scores=z, mean_z=float(np.mean(z)),
                             eligible=all(v <= args.max_nll_increase + 1e-12 for v in increase.values())))
            print(f"round {r} {name}: mean z {rows[-1]['mean_z']:.3f}, NLL increase {increase}", flush=True)
        eligible = [row for row in rows if row["eligible"]]
        best = min(eligible, key=lambda row: (row["mean_z"], row["candidate"] != "current"))
        a = np.array([best["wm_coefficients"][g] for g in groups])
        rounds.append(dict(round=r, step=step, candidates=rows, selected=best["candidate"]))
        print(f"round {r}: selected {best['candidate']}", flush=True)
    assert n_scored == (len(groups) + 1) * n * len(args.round_steps), n_scored

    final_nll = utility(a)
    model_dir = os.path.join(args.output_dir, "model")
    rm.model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    trace = {
        "start": args.start, "variant": start["variant"], "wm_weight": start["wm_weight"],
        "partner_weight": start["partner_weight"], "wm_energy": dict(zip(groups, q.tolist())),
        "partner_coefficients": dict(zip(groups, b.tolist())),
        "start_wm_coefficients": start["wm_coefficients"], "start_heldout_nll": start_nll,
        "rounds": rounds, "selected_wm_coefficients": dict(zip(groups, a.tolist())),
        "selected_heldout_nll": final_nll, "scored_completions": n_scored, "args": vars(args),
    }
    with open(trace_path, "w") as f:
        json.dump(trace, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", required=True, help="learned_merge.py output_dir (same variant, mixture)")
    parser.add_argument("--watermark_config", required=True, help="watermark training config (detector)")
    parser.add_argument("--round_steps", type=float, nargs="+", default=[0.10, 0.05])
    parser.add_argument("--prompts_per_round", type=int, default=10)
    parser.add_argument("--max_nll_increase", type=float, default=0.02, help="nats/token, per domain")
    parser.add_argument("--probe_skip", type=int, default=1000)
    parser.add_argument("--prefix_length", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    main(parser.parse_args())
