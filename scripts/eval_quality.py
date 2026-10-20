import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from mat.config import TrainConfig
from mat.evaluation import (KGWLogitsProcessor, gsm8k_prompts, sample_from_chat_prompts, sample_from_prefixes,
                            web_prefixes)
from mat.utils import load_model, load_tokenizer
from mat.watermarks import build_watermark


def seq_rep_3(ids):
    if len(ids) < 3:
        return 0.0
    grams = [tuple(ids[i: i + 3]) for i in range(len(ids) - 2)]
    return 1.0 - len(set(grams)) / len(grams)


@torch.no_grad()
def reference_ppl(ref_model, ref_tokenizer, src_tokenizer, samples):
    ppls = []
    for prompt_ids, completion_ids in tqdm(samples, desc="Scoring"):
        prompt = src_tokenizer.decode(prompt_ids, skip_special_tokens=True)
        completion = src_tokenizer.decode(completion_ids, skip_special_tokens=True)
        n_prompt = len(ref_tokenizer.encode(prompt, add_special_tokens=False))
        full = ref_tokenizer.encode(prompt + completion, add_special_tokens=False)
        if len(full) - n_prompt < 2:
            ppls.append(float("inf"))
            continue
        ids = torch.tensor([full], device=ref_model.device)
        logits = ref_model(ids).logits[0, n_prompt - 1: -1]
        ppls.append(torch.exp(F.cross_entropy(logits.float(), ids[0, n_prompt:])).item())
    return ppls


def main(args):
    torch.manual_seed(args.seed)
    config = TrainConfig.from_yaml(args.watermark_config)
    tokenizer = load_tokenizer(config.base_model)
    if args.logit_processor:
        model = load_model(config.base_model)
        processor = KGWLogitsProcessor(build_watermark(config.watermark, tokenizer))
    else:
        model, processor = load_model(args.model), None

    if args.domain == "math":
        samples = sample_from_chat_prompts(model, tokenizer, gsm8k_prompts(tokenizer, args.n_samples),
                                           args.max_new_tokens, batch_size=8, logits_processor=processor)
    else:
        prefixes = web_prefixes(args.domain, tokenizer, args.n_samples, args.prefix_length, args.max_new_tokens)
        completions = sample_from_prefixes(model, tokenizer, prefixes, args.max_new_tokens, batch_size=16,
                                           logits_processor=processor)
        samples = list(zip(prefixes, completions))
    del model
    torch.cuda.empty_cache()

    ref_tokenizer = load_tokenizer(args.reference_model)
    ref_model = load_model(args.reference_model)
    ppls = reference_ppl(ref_model, ref_tokenizer, tokenizer, samples)
    reps = [seq_rep_3(c.tolist()) for _, c in samples]
    metrics = {"n_samples": len(ppls), "n_unscorable": int(np.sum(~np.isfinite(ppls))),
               "median_ppl": float(np.median(ppls)), "seq_rep_3": float(np.mean(reps))}
    print(json.dumps(metrics, indent=2))

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"quality_{args.domain}.json"), "w") as f:
        json.dump({"args": vars(args), "metrics": metrics, "ppl": ppls, "seq_rep_3": reps}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="checkpoint to evaluate (not needed with --logit_processor)")
    parser.add_argument("--watermark_config", required=True, help="training config (fixes tokenizer, base model)")
    parser.add_argument("--logit_processor", action="store_true")
    parser.add_argument("--domain", choices=["en", "de", "fr", "math"], default="en")
    parser.add_argument("--reference_model", default="meta-llama/Llama-2-13b-hf")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--prefix_length", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    main(parser.parse_args())
