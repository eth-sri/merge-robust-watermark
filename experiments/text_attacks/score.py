import argparse
import json
import os

import torch

from mat.config import TrainConfig
from mat.evaluation import detection_metrics
from mat.utils import load_tokenizer
from mat.watermarks import build_watermark

LABELS = {
    "clean": "Clean",
    "llama_paraphrase": "Llama paraphrasing",
    "dipper": "DIPPER paraphrasing",
    "backtranslation": "Backtranslation",
    "word_deletion_10": "Word deletion (10%)",
    "word_deletion_20": "Word deletion (20%)",
}


def score_texts(watermark, tokenizer, texts, max_tokens):
    z, p = [], []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
        if len(ids) <= watermark.k:
            z.append(None)
            p.append(None)
            continue
        res = watermark.detect(torch.tensor([ids], device="cuda"))
        z.append(res["z_score"].item())
        p.append(res["p_value"].item())
    return z, p


def score_dir(directory, max_tokens):
    with open(os.path.join(directory, "generations.json")) as f:
        gen = json.load(f)
    config = TrainConfig.from_yaml(gen["watermark_config"])
    tokenizer = load_tokenizer(config.base_model)
    watermark = build_watermark(config.watermark, tokenizer)

    results = {}
    for state in LABELS:
        if state == "clean":
            texts = gen["texts"]
        else:
            path = os.path.join(directory, f"attacked_{state}.json")
            if not os.path.exists(path):
                print(f"MISSING {path}")
                continue
            with open(path) as f:
                texts = json.load(f)["texts"]
        z, p = score_texts(watermark, tokenizer, texts, max_tokens)
        valid = [i for i, v in enumerate(p) if v is not None]
        assert valid, f"no scorable texts for {state} in {directory}"
        metrics = detection_metrics([p[i] for i in valid], [z[i] for i in valid])
        metrics["n_texts"] = len(texts)
        metrics["n_unscorable"] = len(texts) - len(valid)
        metrics["tpr_at_1pct_fpr_unscorable_as_miss"] = sum(p[i] < 0.01 for i in valid) / len(texts)
        results[state] = {"metrics": metrics, "p_values": p, "z_scores": z}

    with open(os.path.join(directory, "scores.json"), "w") as f:
        json.dump({"watermark_config": gen["watermark_config"], "tokenizer": config.base_model,
                   "max_score_tokens": max_tokens, "states": results}, f)
    return results


def main(args):
    all_results = {os.path.basename(os.path.normpath(d)): score_dir(d, args.max_score_tokens) for d in args.dirs}
    names = list(all_results)
    print(f"\nTPR@1% (%), Wald 95% CI, n scored\n{'Text transformation':<22}" +
          "".join(f"{n:>31}" for n in names))
    for state, label in LABELS.items():
        cells = []
        for name in names:
            res = all_results[name].get(state)
            if res is None:
                cells.append(f"{'MISSING':>31}")
                continue
            m = res["metrics"]
            lo, hi = m["tpr_at_1pct_fpr_ci95"]
            cells.append(f"{100 * m['tpr_at_1pct_fpr']:>9.1f} [{100 * lo:5.1f}, {100 * hi:5.1f}] n={m['n_samples']:<4}")
        print(f"{label:<22}" + "".join(cells))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dirs", nargs="+", required=True, help="one output directory per variant")
    parser.add_argument("--max_score_tokens", type=int, default=200)
    main(parser.parse_args())
