import argparse
import json
import math
import os
import random

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from mat.utils import load_model, load_tokenizer

SEED = 20260724


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize(text):
    return " ".join(text.lower().split())


def stratified_counts(sizes, n):
    total = sum(sizes.values())
    exact = {k: n * s / total for k, s in sizes.items()}
    counts = {k: math.floor(v) for k, v in exact.items()}
    order = sorted(sizes, key=lambda k: (exact[k] - counts[k], sizes[k], k), reverse=True)
    for k in order[: n - sum(counts.values())]:
        counts[k] += 1
    return counts


def style_prompts(n=300, seed=SEED):
    with open(hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")) as f:
        rows = json.load(f)
    groups, seen = {}, set()
    for row in rows:
        prompt = row["instruction"].strip()
        if prompt and normalize(prompt) not in seen:
            seen.add(normalize(prompt))
            groups.setdefault(row["dataset"], []).append(prompt)
    counts = stratified_counts({k: len(v) for k, v in groups.items()}, n)
    rng = random.Random(seed)
    chosen = []
    for source in sorted(groups):
        rng.shuffle(groups[source])
        chosen += groups[source][: counts[source]]
    rng.shuffle(chosen)
    return [{"prompt_id": f"style_{i:04d}", "prompt": p} for i, p in enumerate(chosen)]


def xstest_safe_prompts():
    rows = [r for r in load_dataset("Paul/XSTest", split="train") if r["label"].strip().lower() == "safe"]
    assert len(rows) == 250, f"expected 250 safe XSTest prompts, found {len(rows)}"
    return [{"prompt_id": f"xstest_{int(r['id']):04d}", "prompt": r["prompt"].strip()} for r in rows]


def eli5_prompts(n=100):
    out, seen = [], set()
    for row in load_dataset("sentence-transformers/eli5", split="train", streaming=True):
        q = row["question"].strip()
        if q and normalize(q) not in seen:
            seen.add(normalize(q))
            out.append({"prompt_id": f"diversity_{len(out):04d}", "prompt": q})
            if len(out) == n:
                return out
    raise ValueError(f"only {len(out)} unique ELI5 questions")


def chat(tokenizer, prompt):
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                         add_generation_prompt=True)


def stop_ids(model, tokenizer):
    eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
    return eos if isinstance(eos, list) else [eos]


def decode_until_stop(tokenizer, ids, stops):
    ids, stops = ids.tolist(), set(stops) | {tokenizer.pad_token_id}
    end = next((i for i, t in enumerate(ids) if t in stops), None)
    ids = ids if end is None else ids[:end]
    return tokenizer.decode(ids, skip_special_tokens=True).strip(), len(ids), end is None


@torch.no_grad()
def batched_generate(model, tokenizer, texts, max_new_tokens, batch_size, max_length=2048, **sampling):
    tokenizer.padding_side = "left"
    device = model.get_input_embeddings().weight.device
    stops, out = stop_ids(model, tokenizer), []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating"):
        enc = tokenizer(texts[i: i + batch_size], return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length, add_special_tokens=False).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=stops, **sampling)
        out += [decode_until_stop(tokenizer, g, stops) for g in gen[:, enc["input_ids"].shape[1]:]]
    return out


def greedy_suite(model, tokenizer, prompts, args):
    results = batched_generate(model, tokenizer, [chat(tokenizer, p["prompt"]) for p in prompts],
                               args.greedy_max_new_tokens, args.batch_size, do_sample=False)
    return [{**p, "response": r, "n_tokens": n, "hit_limit": h} for p, (r, n, h) in zip(prompts, results)]


@torch.no_grad()
def diversity_suite(model, tokenizer, prompts, args):
    stops, rows = stop_ids(model, tokenizer), []
    for i, p in enumerate(tqdm(prompts, desc="Sampling")):
        torch.manual_seed(args.seed + 10_000 * i)
        enc = tokenizer([chat(tokenizer, p["prompt"])], return_tensors="pt", add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, do_sample=True, temperature=0.7, top_p=0.9, top_k=50,
                             num_return_sequences=args.n_samples, max_new_tokens=args.diversity_max_new_tokens,
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=stops)
        for j, g in enumerate(gen[:, enc["input_ids"].shape[1]:]):
            r, n, h = decode_until_stop(tokenizer, g, stops)
            rows.append({**p, "sample_id": j, "response": r, "n_tokens": n, "hit_limit": h})
    return rows


def main(args):
    tokenizer = load_tokenizer(args.tokenizer)
    model = load_model(args.model)
    suites = {
        "style": lambda: greedy_suite(model, tokenizer, style_prompts(args.n_style), args),
        "xstest": lambda: greedy_suite(model, tokenizer, xstest_safe_prompts(), args),
        "diversity": lambda: diversity_suite(model, tokenizer, eli5_prompts(args.n_diversity), args),
    }
    for name, run in suites.items():
        path = os.path.join(args.output_dir, args.label, f"{name}.jsonl")
        if os.path.exists(path):
            print(f"skip {path} (exists)")
            continue
        rows = run()
        write_jsonl(path, rows)
        print(f"{path}: {len(rows)} responses, {sum(r['hit_limit'] for r in rows)} hit the token limit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="checkpoint to evaluate")
    parser.add_argument("--label", required=True, help="output subfolder, e.g. mat or control")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tokenizer", default="meta-llama/Llama-3.1-8B-Instruct", help="shared chat template")
    parser.add_argument("--n_style", type=int, default=300)
    parser.add_argument("--n_diversity", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=10, help="diversity samples per prompt")
    parser.add_argument("--greedy_max_new_tokens", type=int, default=256)
    parser.add_argument("--diversity_max_new_tokens", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=SEED)
    main(parser.parse_args())
