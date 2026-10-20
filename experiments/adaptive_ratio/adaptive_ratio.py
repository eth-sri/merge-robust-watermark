import argparse
import json
import os
import shutil

import numpy as np
import scipy.stats
import torch
import yaml

from mat.config import TrainConfig
from mat.evaluation import detect_batched, detection_metrics, sample_from_prefixes, web_prefixes
from mat.merging import merge, pairwise_config
from mat.utils import load_model, load_tokenizer
from mat.watermarks import build_watermark


def generate_scores(args, cfg, ratio_dirs):
    todo = {a: d for a, d in ratio_dirs.items() if not os.path.exists(os.path.join(d, "scores.json"))}
    if not todo:
        return
    models = {**cfg["unwatermarked"], **cfg["variants"][args.variant]}
    spec = cfg["scenarios"][args.scenario]
    tokenizer_source = models[spec["tokenizer_source"]] if "tokenizer_source" in spec else None
    config_source = models[spec["config_source"]] if "config_source" in spec else None
    config = TrainConfig.from_yaml(cfg["variants"][args.variant]["watermark_config"])
    tokenizer = load_tokenizer(config.base_model)
    watermark = build_watermark(config.watermark, tokenizer)
    prefixes = web_prefixes("en", tokenizer, args.probe_pool_size + args.holdout_size,
                            args.prefix_length, args.max_new_tokens)

    for alpha, out in todo.items():
        model_dir = os.path.join(out, "model")
        if not os.path.exists(os.path.join(model_dir, "config.json")):
            merge(pairwise_config("linear", models[spec["A"]], models[spec["B"]], alpha), model_dir,
                  tokenizer_source=tokenizer_source, config_source=config_source)
        model = load_model(model_dir)
        torch.manual_seed(args.seed)
        completions = sample_from_prefixes(model, tokenizer, prefixes, args.max_new_tokens, batch_size=16)
        z, p = detect_batched(watermark, completions, batch_size=16)
        with open(os.path.join(out, "scores.json"), "w") as f:
            json.dump({"alpha": alpha, "metrics": detection_metrics(p, z), "z_scores": z, "p_values": p,
                       "completion_ids": completions.tolist()}, f)
        del model
        torch.cuda.empty_cache()
        if not args.keep_models:
            shutil.rmtree(model_dir)


def select_ratios(args, ratio_dirs):
    alphas = sorted(ratio_dirs)
    z, p = {}, {}
    for a in alphas:
        with open(os.path.join(ratio_dirs[a], "scores.json")) as f:
            scores = json.load(f)
        z[a], p[a] = np.asarray(scores["z_scores"]), np.asarray(scores["p_values"])
    n = args.probe_pool_size + args.holdout_size
    assert all(len(z[a]) == n for a in alphas), f"cached scores must cover {n} prompts"

    perm = np.random.default_rng(args.seed).permutation(n)
    pool, heldout = perm[: args.probe_pool_size], perm[args.probe_pool_size:]
    detected = np.stack([p[a][heldout] < 0.01 for a in alphas]).astype(float)

    repeats, chosen = [], []
    for r in range(args.n_repeats):
        probe = np.random.default_rng(r).choice(pool, args.probe_size, replace=False)
        probe_z = [float(z[a][probe].mean()) for a in alphas]
        i = min(range(len(alphas)), key=lambda j: (probe_z[j], -alphas[j]))
        chosen.append(i)
        repeats.append({"repeat": r, "selected_alpha": alphas[i],
                        "heldout_tpr_at_1pct_fpr": float(detected[i].mean()),
                        "probe_mean_z": dict(zip(map(str, alphas), probe_z)),
                        "probe_indices": probe.tolist()})
    chosen = np.array(chosen)
    tprs = detected[chosen].mean(axis=1)
    mean = float(tprs.mean())

    half = float(scipy.stats.t.ppf(0.975, len(tprs) - 1) * tprs.std(ddof=1) / np.sqrt(len(tprs)))
    rng = np.random.default_rng(args.seed)
    boot = []
    for _ in range(args.n_bootstrap):
        rows = chosen[rng.integers(0, len(chosen), len(chosen))]
        cols = rng.integers(0, len(heldout), len(heldout))
        boot.append(detected[rows][:, cols].mean())

    return {
        "variant": args.variant, "scenario": args.scenario, "merge_method": "linear",
        "alphas_watermarked_fraction": alphas,
        "probe_pool_size": args.probe_pool_size, "holdout_size": args.holdout_size,
        "queries_per_ratio": args.probe_size, "queries_per_repeat": args.probe_size * len(alphas),
        "n_repeats": args.n_repeats, "seed": args.seed,
        "selection": "lowest mean probe z-score; ties to the larger alpha",
        "mean_heldout_tpr_at_1pct_fpr": mean,
        "ci95_t_over_repeats": [mean - half, mean + half],
        "ci95_bootstrap_repeats_and_prompts": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "n_bootstrap": args.n_bootstrap,
        "selected_counts": {str(a): int((chosen == i).sum()) for i, a in enumerate(alphas)},
        "fixed_ratio_heldout_tpr_at_1pct_fpr": {str(a): float(detected[i].mean()) for i, a in enumerate(alphas)},
        "repeats": repeats,
    }


def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    root = os.path.join(args.output_root, args.variant, args.scenario)
    ratio_dirs = {a: os.path.join(root, f"alpha_{a}") for a in args.alphas}
    for d in ratio_dirs.values():
        os.makedirs(d, exist_ok=True)
    generate_scores(args, cfg, ratio_dirs)
    result = select_ratios(args, ratio_dirs)

    print(f"{args.variant}: mean held-out TPR@1% = {result['mean_heldout_tpr_at_1pct_fpr']:.3f} "
          f"(95% t {result['ci95_t_over_repeats'][0]:.3f}-{result['ci95_t_over_repeats'][1]:.3f}; "
          f"95% bootstrap {result['ci95_bootstrap_repeats_and_prompts'][0]:.3f}-"
          f"{result['ci95_bootstrap_repeats_and_prompts'][1]:.3f}), {args.n_repeats} repeats")
    print("selected counts:", result["selected_counts"])
    print("fixed-ratio held-out TPR@1%:", result["fixed_ratio_heldout_tpr_at_1pct_fpr"])
    with open(os.path.join(root, "adaptive_selection.json"), "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/merge/scenarios.yaml")
    parser.add_argument("--variant", required=True, help="kgwd or mat (entries of scenarios.yaml)")
    parser.add_argument("--scenario", default="BFp_OpenMath", help="pairwise scenario; alpha = weight on B")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.7, 0.75, 0.8, 0.85, 0.9])
    parser.add_argument("--probe_size", type=int, default=50, help="detector queries per ratio per repeat")
    parser.add_argument("--n_repeats", type=int, default=20)
    parser.add_argument("--probe_pool_size", type=int, default=200)
    parser.add_argument("--holdout_size", type=int, default=300)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--prefix_length", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42, help="sampling seed and probe/held-out split seed")
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--output_root", default="output/adaptive_ratio")
    parser.add_argument("--keep_models", action="store_true")
    main(parser.parse_args())
