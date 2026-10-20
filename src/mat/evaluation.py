import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import LogitsProcessor, LogitsProcessorList

WEB_SOURCES = {
    "en": dict(path="allenai/c4", name="realnewslike", split="validation"),
    "de": dict(path="HuggingFaceFW/fineweb-2", name="deu_Latn", split="train"),
    "fr": dict(path="HuggingFaceFW/fineweb-2", name="fra_Latn", split="train"),
}
TRAINING_RANGE = {"de": 100_000}

GSM8K_FEW_SHOT = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.\n#### 6"),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.\n#### 5"),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
     "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.\n#### 39"),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
     "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.\n#### 8"),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
     "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.\n#### 9"),
    ("There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
     "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29.\n#### 29"),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
     "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33.\n#### 33"),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 = 8 dollars left.\n#### 8"),
]


def web_texts(domain, tokenizer, n, min_tokens, keep_tokens, skip=0):
    out, skipped = [], 0
    for row in load_dataset(**WEB_SOURCES[domain], streaming=True).skip(TRAINING_RANGE.get(domain, 0)):
        tokens = tokenizer.encode(row["text"], add_special_tokens=False)
        if len(tokens) < min_tokens:
            continue
        if skipped < skip:
            skipped += 1
            continue
        out.append(torch.tensor(tokens[:keep_tokens]))
        if len(out) >= n:
            break
    return out


def web_prefixes(domain, tokenizer, n, prefix_length=50, completion_length=200):
    return web_texts(domain, tokenizer, n, prefix_length + completion_length, prefix_length)


def gsm8k_prompts(tokenizer, n, n_shots=8):
    """GSM8K test questions after the 8-shot chain-of-thought exemplars of Wei et al. (2022)."""
    prompts = []
    for row in load_dataset("openai/gsm8k", "main", split="test"):
        messages = []
        for q, a in GSM8K_FEW_SHOT[:n_shots]:
            messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        messages.append({"role": "user", "content": row["question"]})
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        if len(prompts) >= n:
            break
    return prompts


def gsm8k_human_answers(tokenizer, n, num_tokens=200):
    out = []
    for row in load_dataset("openai/gsm8k", "main", split="train"):
        tokens = tokenizer.encode(row["answer"], add_special_tokens=False)
        if len(tokens) >= num_tokens:
            out.append(torch.tensor(tokens[:num_tokens]))
            if len(out) >= n:
                break
    return out


SAMPLING = dict(do_sample=True, temperature=1.0, top_k=50, top_p=1.0, repetition_penalty=1.0)


class KGWLogitsProcessor(LogitsProcessor):

    def __init__(self, watermark):
        self.watermark = watermark

    def __call__(self, input_ids, scores):
        return self.watermark.watermark_logits(input_ids[:, -self.watermark.k:], scores.unsqueeze(1)).squeeze(1)


def _generate(model, logits_processor, **kwargs):
    if logits_processor is not None:
        kwargs["logits_processor"] = LogitsProcessorList([logits_processor])
    return model.generate(**kwargs, **SAMPLING)

@torch.no_grad()
def sample_from_prefixes(model, tokenizer, prefixes, max_new_tokens=200, batch_size=16, logits_processor=None):
    outs = []
    for i in tqdm(range(0, len(prefixes), batch_size), desc="Generating"):
        input_ids = torch.stack(prefixes[i: i + batch_size]).to(model.device)
        out = _generate(model, logits_processor, input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                        max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id)
        outs.append(out[:, input_ids.shape[1]:])
    return torch.cat(outs)


@torch.no_grad()
def sample_from_chat_prompts(model, tokenizer, prompts, max_new_tokens=200, batch_size=8, logits_processor=None):
    tokenizer.padding_side = "left"
    samples = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        enc = tokenizer(prompts[i: i + batch_size], return_tensors="pt", padding=True,
                        truncation=True, max_length=2048).to(model.device)
        out = _generate(model, logits_processor, **enc, max_new_tokens=max_new_tokens,
                        min_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
        L = enc["input_ids"].shape[1]
        for j in range(out.shape[0]):
            samples.append((enc["input_ids"][j][enc["attention_mask"][j].bool()], out[j, L:]))
    return samples


def detect_batched(watermark, token_seqs, batch_size=64, device="cuda"):
    z, p = [], []
    for i in range(0, len(token_seqs), batch_size):
        batch = token_seqs[i: i + batch_size]
        batch = torch.stack(list(batch)) if not torch.is_tensor(batch) else batch
        res = watermark.detect(batch.to(device))
        z += res["z_score"].cpu().tolist()
        p += res["p_value"].cpu().tolist()
    return z, p


def wald_ci(rate, n, z=1.96):
    half = z * np.sqrt(rate * (1 - rate) / n)
    return [float(max(0.0, rate - half)), float(min(1.0, rate + half))]


def bootstrap_ci(detected, resamples=10_000, seed=42):
    rng = np.random.default_rng(seed)
    rates = detected[rng.integers(0, len(detected), size=(resamples, len(detected)))].mean(axis=1)
    return [float(q) for q in np.quantile(rates, [0.025, 0.975])]


def detection_metrics(p_values, z_scores, null_p_values=None):
    p = np.asarray(p_values)
    tpr1 = float((p < 0.01).mean())
    metrics = {
        "n_samples": len(p),
        "tpr_at_1pct_fpr": tpr1,
        "tpr_at_1pct_fpr_ci95": wald_ci(tpr1, len(p)),
        "tpr_at_1pct_fpr_bootstrap_ci95": bootstrap_ci((p < 0.01).astype(float)),
        "tpr_at_5pct_fpr": float((p < 0.05).mean()),
        "median_p_value": float(np.median(p)),
        "mean_z_score": float(np.mean(z_scores)),
        "median_z_score": float(np.median(z_scores)),
    }
    if null_p_values:
        null = np.asarray(null_p_values[: len(p)])
        y = np.concatenate([np.zeros(len(p)), np.ones(len(null))])
        metrics["auroc"] = float(roc_auc_score(y, np.concatenate([p, null])))
    return metrics
