import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from mat.evaluation import sample_from_prefixes, wald_ci, web_texts
from mat.utils import load_model, load_tokenizer
from mat.watermarks.gaussmark import GaussMarkSecret, apply_secret, sequence_logprob


@torch.no_grad()
def logprobs(model, tokens, batch_size=8):
    return torch.cat([sequence_logprob(model, tokens[i: i + batch_size].to(model.device)).cpu()
                      for i in tqdm(range(0, len(tokens), batch_size), desc="Scoring")])


def main(args):
    torch.manual_seed(args.seed)
    tokenizer = load_tokenizer(args.base)
    n, L = args.n_samples, args.prefix_length + args.max_new_tokens
    prefixes = web_texts("en", tokenizer, n, L, args.prefix_length)
    null_prefixes = web_texts("en", tokenizer, n, L, args.prefix_length, skip=n)

    model = load_model(args.model)
    positives = sample_from_prefixes(model, tokenizer, prefixes, args.max_new_tokens, batch_size=8)
    del model
    torch.cuda.empty_cache()

    base = load_model(args.base)
    null = sample_from_prefixes(base, tokenizer, null_prefixes, args.max_new_tokens, batch_size=8)
    base_pos, base_null = logprobs(base, positives), logprobs(base, null)
    apply_secret(base, GaussMarkSecret.load(args.secret))
    t_pos = (logprobs(base, positives) - base_pos).numpy()
    t_null = (logprobs(base, null) - base_null).numpy()

    threshold = float(np.quantile(t_null, 0.99))
    tpr = float((t_pos > threshold).mean())
    metrics = {
        "n_samples": len(t_pos), "n_null": len(t_null), "threshold_1pct": threshold,
        "tpr_at_1pct_fpr": tpr, "tpr_at_1pct_fpr_ci95": wald_ci(tpr, len(t_pos)),
        "tpr_at_5pct_fpr": float((t_pos > np.quantile(t_null, 0.95)).mean()),
        "auroc": float(roc_auc_score(np.r_[np.ones(len(t_pos)), np.zeros(len(t_null))], np.r_[t_pos, t_null])),
    }
    print(json.dumps(metrics, indent=2))
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "detection_en.json"), "w") as f:
        json.dump({"args": vars(args), "metrics": metrics, "T_pos": t_pos.tolist(), "T_null": t_null.tolist()}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--secret", required=True)
    parser.add_argument("--base", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--prefix_length", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    main(parser.parse_args())
