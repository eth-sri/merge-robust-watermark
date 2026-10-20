import argparse
import json
import math
import os

import numpy as np
import torch
import yaml

from mat.config import TrainConfig
from mat.evaluation import web_texts
from mat.utils import DTYPES, load_model, load_tokenizer
from mat.watermarks import build_watermark


@torch.no_grad()
def green_mass(model, contexts, watermark, batch_size):
    out = []
    for i in range(0, len(contexts), batch_size):
        ids = contexts[i: i + batch_size].to(model.device)
        mask = watermark._greenlist_mask(watermark._context_hashes(ids))[:, :-1]
        logits = model(input_ids=ids, use_cache=False).logits[:, watermark.k - 1: -1, : mask.shape[-1]]
        out.append((logits.float().softmax(-1) * mask).sum(-1).double().cpu())
    return torch.cat(out)


@torch.no_grad()
def set_weights(model, theta0, theta1, lam):
    for name, p in model.named_parameters():
        p.copy_(theta0[name])
        p.lerp_(theta1[name].to(p.device, p.dtype), lam)


def delta_norm(theta0, theta1, device):
    total = 0.0
    for name in theta0:
        d = theta1[name].to(device, torch.float32) - theta0[name].to(device, torch.float32)
        total += torch.linalg.vector_norm(d).item() ** 2
    return math.sqrt(total)


def compute_scores(args, models, todo, out_file):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    base_path = models[todo[0]]["B_prime"]
    tokenizer = load_tokenizer(base_path)
    watermark = build_watermark(TrainConfig.from_yaml(models[todo[0]]["watermark_config"]).watermark, tokenizer)
    contexts = torch.stack(web_texts("en", tokenizer, args.n_contexts, args.min_tokens, args.context_length))
    assert len(contexts) == args.n_contexts, f"only {len(contexts)} qualifying documents"

    model = load_model(base_path, dtype="bfloat16", device_map=None)
    theta0 = {n: p.detach().clone() for n, p in model.named_parameters()}
    model.to(device="cuda", dtype=DTYPES[args.dtype])
    p0 = green_mass(model, contexts, watermark, args.batch_size)
    factor = math.sqrt(p0.shape[1] / (watermark.gamma * (1 - watermark.gamma)))

    for variant in todo:
        ckpt = models[variant]["B"]
        theta1 = {n: p.detach() for n, p in load_model(ckpt, dtype="bfloat16", device_map=None).named_parameters()}
        assert theta1.keys() == theta0.keys() and all(theta1[n].shape == theta0[n].shape for n in theta0), \
            f"{ckpt} and {base_path} have different parameters"
        norm = delta_norm(theta0, theta1, "cuda")
        print(f"[{variant}] ||dtheta||_2 = {norm:.4f}  ({ckpt} - {base_path})")

        scores = {}
        for lam in args.lambdas:
            set_weights(model, theta0, theta1, lam)
            s = factor * (green_mass(model, contexts, watermark, args.batch_size) - p0).mean(dim=1)
            scores[str(lam)] = s.tolist()
            print(f"[{variant}] lambda={lam:.2f}  mean score {s.mean():.4f}")
        with open(out_file(variant), "w") as f:
            json.dump({"variant": variant, "checkpoint": ckpt, "base": base_path,
                       "watermark_config": models[variant]["watermark_config"], "delta_l2_norm": norm,
                       "n_contexts": args.n_contexts, "context_length": args.context_length,
                       "dtype": args.dtype, "score_factor": factor, "scores": scores}, f, indent=1)
        del theta1


def percentile_ci(draws):
    return [float(v) for v in np.percentile(draws, [2.5, 97.5])]


def summarize(args, results):
    rows = []
    for variant, res in results.items():
        scores = {float(lam): np.asarray(s) for lam, s in res["scores"].items()}
        end = scores[1.0]
        idx = np.random.default_rng(args.seed).integers(0, len(end), size=(args.bootstrap, len(end)))
        for lam in args.lambdas:
            x = scores[lam]
            rows.append({
                "variant": variant, "lambda": lam, "n_texts": len(x),
                "mean_score": float(x.mean()), "mean_score_ci95": percentile_ci(x[idx].mean(1)),
                "pct_retained": float(100 * x.mean() / end.mean()),
                "pct_retained_ci95": percentile_ci(100 * x[idx].mean(1) / end[idx].mean(1)),
            })

    print("\n||dtheta||_2: " + ", ".join(f"{v} {r['delta_l2_norm']:.4f}" for v, r in results.items()))
    print(f"% of full-checkpoint score retained (95% percentile bootstrap over {rows[0]['n_texts']} texts, "
          f"{args.bootstrap} resamples)")
    print("lambda" + "".join(f"{v:>26s}" for v in results))
    for lam in args.lambdas:
        cells = [r for r in rows if r["lambda"] == lam]
        print(f"{lam:6.2f}" + "".join(f"{r['pct_retained']:11.1f} [{r['pct_retained_ci95'][0]:5.1f}, "
                                      f"{r['pct_retained_ci95'][1]:5.1f}]" for r in cells))
    for r in rows:
        if r["lambda"] == 1.0:
            lo, hi = r["mean_score_ci95"]
            print(f"full-checkpoint score {r['variant']}: {r['mean_score']:.3f} [{lo:.3f}, {hi:.3f}]")

    with open(os.path.join(args.output_dir, "table16.json"), "w") as f:
        json.dump({"args": vars(args), "delta_l2_norm": {v: r["delta_l2_norm"] for v, r in results.items()},
                   "rows": rows}, f, indent=2)


def main(args):
    args.lambdas = sorted(set(args.lambdas) | {1.0})
    with open(args.scenarios) as f:
        cfg = yaml.safe_load(f)
    models = {v: {**cfg["unwatermarked"], **cfg["variants"][v]} for v in args.variants}
    assert len({m["B_prime"] for m in models.values()}) == 1, "variants must share the base theta_0"
    detectors = {(w.type, w.gamma, w.seeding_scheme) for w in
                 (TrainConfig.from_yaml(m["watermark_config"]).watermark for m in models.values())}
    assert len(detectors) == 1 and next(iter(detectors))[0] == "kgw", "the score needs one shared KGW green list"

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = lambda v: os.path.join(args.output_dir, f"scores_{v}.json")
    todo = [v for v in args.variants if not os.path.exists(out_file(v))]
    if todo:
        compute_scores(args, models, todo, out_file)
    results = {}
    for v in args.variants:
        with open(out_file(v)) as f:
            results[v] = json.load(f)
    summarize(args, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", default="configs/merge/scenarios.yaml")
    parser.add_argument("--variants", nargs="+", default=["kgwd", "mat"])
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.1, 0.25, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--n_contexts", type=int, default=48)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--min_tokens", type=int, default=256, help="documents shorter than this are skipped")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="output/mechanism")
    main(parser.parse_args())
