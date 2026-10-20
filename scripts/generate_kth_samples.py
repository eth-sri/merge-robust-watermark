import argparse
import itertools
import json
import os

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList

from mat.config import TrainConfig
from mat.data import distillation_mixture
from mat.utils import check_rope_config, dtype_kwarg, load_tokenizer
from mat.watermarks import build_watermark


class KTHDecoding(LogitsProcessor):
    def __init__(self, gumbel):
        self.gumbel = gumbel

    def __call__(self, input_ids, scores):
        noise = self.gumbel[input_ids.shape[1] % self.gumbel.shape[0]]
        v = min(noise.shape[-1], scores.shape[-1])
        scores[..., :v] = scores[..., :v] + noise[:v].to(scores.dtype)
        return scores


def prompt_stream(config, tokenizer, rank, world_size):
    samples = config.samples
    mixture = distillation_mixture(tokenizer, config.datasets, config.proportions,
                                   max(samples.prompt_length + 1, 256), seed=samples.prompt_seed)
    index = 0
    for example in mixture:
        ids = example["input_ids"][:samples.prompt_length]
        if len(ids) != samples.prompt_length:
            continue
        if index % world_size == rank:
            yield ids
        index += 1


def main(args):
    config = TrainConfig.from_yaml(args.config)
    samples = config.samples
    rank, world_size = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}")
    if world_size > 1:
        dist.init_process_group("gloo")
    torch.manual_seed(config.watermark.kth_seed + rank)

    os.makedirs(samples.data_dir, exist_ok=True)
    shard = os.path.join(samples.data_dir, f"shard_{rank:03d}.jsonl")
    target = samples.num_samples // world_size + int(rank < samples.num_samples % world_size)
    done = sum(1 for _ in open(shard)) if os.path.exists(shard) else 0

    tokenizer = load_tokenizer(config.base_model)
    check_rope_config(config.base_model)
    model = AutoModelForCausalLM.from_pretrained(config.base_model, **dtype_kwarg(config.dtype)).to(device).eval()
    watermark = build_watermark(config.watermark, tokenizer)
    processor = LogitsProcessorList([KTHDecoding(watermark.gumbel.to(device))])

    prompts = itertools.islice(prompt_stream(config, tokenizer, rank, world_size), done, None)
    with open(shard, "a") as f:
        while done < target:
            batch = list(itertools.islice(prompts, min(samples.batch_size, target - done)))
            if not batch:
                break
            input_ids = torch.tensor(batch, device=device)
            with torch.no_grad():
                out = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                     do_sample=False, min_new_tokens=samples.completion_length,
                                     max_new_tokens=samples.completion_length, logits_processor=processor,
                                     pad_token_id=tokenizer.eos_token_id)
            for row in out[:, input_ids.shape[1]:]:
                f.write(json.dumps({"text": tokenizer.decode(row.tolist(), skip_special_tokens=True)}) + "\n")
            f.flush()
            done += len(batch)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate KTH-watermarked samples for sampling distillation.")
    parser.add_argument("--config", required=True)
    main(parser.parse_args())
