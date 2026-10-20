import argparse
import json
import os
import random
import re

import torch
from tqdm import tqdm

from mat.utils import dtype_kwarg, load_model, load_tokenizer

ATTACKS = ["llama_paraphrase", "backtranslation", "dipper", "word_deletion_10", "word_deletion_20"]

LLAMA_MAX_NEW_TOKENS = 280
DIPPER_LEX_DIVERSITY, DIPPER_ORDER_DIVERSITY, DIPPER_WINDOW = 60, 0, 3
DIPPER_TOP_P, DIPPER_MAX_LENGTH = 0.75, 512
DELETION_RATIOS = {"word_deletion_10": 0.10, "word_deletion_20": 0.20}

SYSTEM = ("You transform passages faithfully. Preserve meaning, factual details, "
          "tone, and approximate length. Return only the requested transformed text.")
PARAPHRASE = ("Rewrite the passage using substantially different wording and sentence "
              "structure. Preserve its meaning, factual details, tone, and approximate "
              "length. Do not summarize or comment on it.\n\nPASSAGE:\n{text}")
TRANSLATE = ("Translate the passage into {language}. Preserve every factual detail and "
             "return only the translation.\n\nPASSAGE:\n{text}")


class LlamaRewriter:

    def __init__(self, model_name, batch_size):
        self.tokenizer = load_tokenizer(model_name)
        self.tokenizer.padding_side = "left"
        self.model = load_model(model_name)
        self.batch_size = batch_size

    @torch.no_grad()
    def __call__(self, user_prompts):
        outputs = []
        for i in tqdm(range(0, len(user_prompts), self.batch_size), desc="Rewriting"):
            chats = [self.tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True) for p in user_prompts[i: i + self.batch_size]]
            enc = self.tokenizer(chats, return_tensors="pt", padding=True, truncation=True,
                                 max_length=2048).to(self.model.device)
            out = self.model.generate(**enc, do_sample=False, max_new_tokens=LLAMA_MAX_NEW_TOKENS,
                                      pad_token_id=self.tokenizer.pad_token_id,
                                      eos_token_id=self.tokenizer.eos_token_id)
            new = out[:, enc["input_ids"].shape[1]:]
            outputs += [t.strip() for t in self.tokenizer.batch_decode(new, skip_special_tokens=True)]
        return outputs


def sentence_windows(text, size):
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", " ".join(text.split())) if s.strip()]
    return [" ".join(sentences[i: i + size]) for i in range(0, len(sentences), size)]


class Dipper:

    def __init__(self, model_name):
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name, **dtype_kwarg("bfloat16"), device_map="cuda").eval()
        self.codes = f"lexical = {100 - DIPPER_LEX_DIVERSITY}, order = {100 - DIPPER_ORDER_DIVERSITY}"

    @torch.no_grad()
    def __call__(self, text, prefix):
        prefix = " ".join(prefix.split())
        outputs = []
        for window in sentence_windows(text, DIPPER_WINDOW):
            prompt = f"{self.codes} {prefix} <sent> {window} </sent>" if prefix else \
                f"{self.codes} <sent> {window} </sent>"
            enc = self.tokenizer([prompt], return_tensors="pt", truncation=True, max_length=1024).to(self.model.device)
            out = self.model.generate(**enc, do_sample=True, top_p=DIPPER_TOP_P, top_k=0,
                                      max_length=DIPPER_MAX_LENGTH)
            rewritten = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
            if rewritten:
                outputs.append(rewritten)
                prefix = f"{prefix} {rewritten}".strip()
        return " ".join(outputs)


def delete_words(text, ratio, rng):
    words = text.split()
    kept = [w for w in words if rng.random() >= ratio]
    if words and not kept:
        kept = [rng.choice(words)]
    return " ".join(kept)


def main(args):
    with open(os.path.join(args.input_dir, "generations.json")) as f:
        gen = json.load(f)
    texts, prefixes = gen["texts"], gen["prefix_texts"]
    attacks = ATTACKS if args.attacks == ["all"] else args.attacks
    llama = None

    for attack in attacks:
        out = os.path.join(args.input_dir, f"attacked_{attack}.json")
        if os.path.exists(out):
            print(f"{out} exists, skipping")
            continue
        result = {"attack": attack, "seed": args.seed}
        if attack in DELETION_RATIOS:
            rng = random.Random(args.seed)
            result["texts"] = [delete_words(t, DELETION_RATIOS[attack], rng) for t in texts]
        elif attack == "dipper":
            llama = None
            torch.cuda.empty_cache()
            dipper = Dipper(args.dipper_model)
            torch.manual_seed(args.seed)
            result["texts"] = [dipper(t, p) for t, p in tqdm(list(zip(texts, prefixes)), desc="DIPPER")]
            del dipper
            torch.cuda.empty_cache()
        else:
            if llama is None:
                llama = LlamaRewriter(args.llama_model, args.batch_size)
            if attack == "llama_paraphrase":
                result["texts"] = llama([PARAPHRASE.format(text=t) for t in texts])
            else:
                french = llama([TRANSLATE.format(language="French", text=t) for t in texts])
                result["intermediate_texts"] = french
                result["texts"] = llama([TRANSLATE.format(language="English", text=t) for t in french])
        result["texts"] = [t.strip() for t in result["texts"]]
        print(f"{attack}: {sum(not t for t in result['texts'])} empty outputs of {len(texts)}")
        with open(out, "w") as f:
            json.dump(result, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", required=True, help="directory with generations.json (generate.py)")
    parser.add_argument("--attacks", nargs="+", default=["all"], choices=["all", *ATTACKS])
    parser.add_argument("--llama_model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dipper_model", default="kalpeshk2011/dipper-paraphraser-xxl")
    parser.add_argument("--batch_size", type=int, default=8, help="Llama rewriting batch size")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
