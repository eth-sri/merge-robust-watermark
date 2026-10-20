import argparse
import json
import os

import torch

from mat.config import TrainConfig
from mat.evaluation import (KGWLogitsProcessor, detect_batched, detection_metrics, gsm8k_human_answers,
                            gsm8k_prompts, sample_from_chat_prompts, sample_from_prefixes, web_prefixes,
                            web_texts)
from mat.utils import load_model, load_tokenizer
from mat.watermarks import build_watermark


def main(args):
    torch.manual_seed(args.seed)
    config = TrainConfig.from_yaml(args.watermark_config)
    tokenizer = load_tokenizer(config.base_model)
    watermark = build_watermark(config.watermark, tokenizer)
    if args.logit_processor:
        assert config.watermark.type == "kgw", "decoding-time reference is implemented for KGW"
        model, processor = load_model(config.base_model), KGWLogitsProcessor(watermark)
    else:
        model, processor = load_model(args.model, quantization=args.quantization), None

    if args.domain == "math":
        samples = sample_from_chat_prompts(model, tokenizer, gsm8k_prompts(tokenizer, args.n_samples),
                                           args.max_new_tokens, batch_size=8, logits_processor=processor)
        completions = torch.stack([c for _, c in samples])
    else:
        prefixes = web_prefixes(args.domain, tokenizer, args.n_samples, args.prefix_length, args.max_new_tokens)
        completions = sample_from_prefixes(model, tokenizer, prefixes, args.max_new_tokens, batch_size=16,
                                           logits_processor=processor)
    z, p = detect_batched(watermark, completions, batch_size=16)

    null_p = None
    if args.auroc:
        human = (gsm8k_human_answers(tokenizer, args.n_samples, args.max_new_tokens) if args.domain == "math" else
                 web_texts(args.domain, tokenizer, args.n_samples, args.max_new_tokens, args.max_new_tokens,
                           skip=args.n_samples))
        _, null_p = detect_batched(watermark, human)
    metrics = detection_metrics(p, z, null_p)
    print(json.dumps(metrics, indent=2))

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"detection_{args.domain}.json"), "w") as f:
        json.dump({"args": vars(args), "metrics": metrics, "p_values": p, "z_scores": z,
                   "null_p_values": null_p, "completion_ids": completions.tolist()}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="checkpoint to evaluate (not needed with --logit_processor)")
    parser.add_argument("--watermark_config", required=True)
    parser.add_argument("--domain", choices=["en", "de", "math"], default="en")
    parser.add_argument("--quantization", choices=["int8", "fp4", "nf4"])
    parser.add_argument("--logit_processor", action="store_true")
    parser.add_argument("--auroc", action="store_true")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--prefix_length", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    main(parser.parse_args())
