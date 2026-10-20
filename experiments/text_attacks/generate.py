import argparse
import json
import os
import shutil

import torch
import yaml

from mat.config import TrainConfig
from mat.evaluation import sample_from_prefixes, web_prefixes
from mat.merging import merge, pairwise_config
from mat.utils import load_model, load_tokenizer


def main(args):
    out = os.path.join(args.output_dir, "generations.json")
    if os.path.exists(out):
        print(f"{out} exists, skipping")
        return
    with open(args.scenarios) as f:
        cfg = yaml.safe_load(f)
    models = {**cfg["unwatermarked"], **cfg["variants"][args.variant]}

    model_dir = os.path.join(args.output_dir, "merged_model")
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        merge(pairwise_config("linear", models["B_prime"], models["B"], args.alpha), model_dir)

    tokenizer = load_tokenizer(TrainConfig.from_yaml(models["watermark_config"]).base_model)
    prefixes = web_prefixes("en", tokenizer, args.n_samples, args.prefix_length, args.max_new_tokens)
    torch.manual_seed(args.seed)
    completions = sample_from_prefixes(load_model(model_dir), tokenizer, prefixes, args.max_new_tokens, batch_size=16)

    with open(out, "w") as f:
        json.dump({"args": vars(args), "watermark_config": models["watermark_config"],
                   "prefix_texts": [tokenizer.decode(p.tolist(), skip_special_tokens=True) for p in prefixes],
                   "texts": tokenizer.batch_decode(completions.tolist(), skip_special_tokens=True),
                   "completion_ids": completions.tolist()}, f)
    print(f"Wrote {len(prefixes)} continuations to {out}")
    if not args.keep_model:
        shutil.rmtree(model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", required=True, help="variant in configs/merge/scenarios.yaml (kgwd, mat)")
    parser.add_argument("--scenarios", default="configs/merge/scenarios.yaml")
    parser.add_argument("--alpha", type=float, default=0.5, help="weight on the watermarked parent B")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--prefix_length", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep_model", action="store_true")
    parser.add_argument("--output_dir", required=True)
    main(parser.parse_args())
