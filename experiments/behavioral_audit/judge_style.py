import argparse
import json
import os
import re

from generate import batched_generate, chat, read_jsonl, write_jsonl
from mat.utils import load_model, load_tokenizer

AXES = ("verbosity", "formality", "structuring", "hedging_or_caution")
RUBRIC = """Compare two assistant responses to the same user instruction.
Judge presentation style only. Ignore factual correctness, helpfulness, safety,
and which answer you prefer.

For each axis, return an integer from -2 to 2:
-2 = A is much more; -1 = A is somewhat more; 0 = no directional difference;
1 = B is somewhat more; 2 = B is much more.

Axes:
- verbosity
- formality
- structuring (headings, lists, explicit organization)
- hedging_or_caution

Also decide whether the pair has a material overall style difference. Ordinary
wording/content variation without a systematic presentation difference is false.

User instruction:
{prompt}

Response A:
{response_a}

Response B:
{response_b}

Return exactly one JSON object with this schema:
{{"verbosity": 0, "formality": 0, "structuring": 0,
  "hedging_or_caution": 0, "material_style_difference": false}}
"""


def parse_judgment(text):
    for candidate in reversed(re.findall(r"\{.*?\}", text, flags=re.DOTALL)):
        try:
            value = json.loads(candidate)
            scores = {axis: int(value[axis]) for axis in AXES}
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if all(-2 <= s <= 2 for s in scores.values()) and isinstance(value.get("material_style_difference"), bool):
            return scores
    return None


def main(args):
    out_path = os.path.join(args.output_dir, f"style_judge_{args.name}.jsonl")
    if os.path.exists(out_path):
        print(f"skip {out_path} (exists)")
        return
    mat = {r["prompt_id"]: r for r in read_jsonl(os.path.join(args.output_dir, args.mat_label, "style.jsonl"))}
    ctl = {r["prompt_id"]: r for r in read_jsonl(os.path.join(args.output_dir, args.control_label, "style.jsonl"))}
    assert mat.keys() == ctl.keys() and all(mat[k]["prompt"] == ctl[k]["prompt"] for k in mat), "prompt mismatch"

    requests = []
    for pid in sorted(mat):
        for order in ("mat_a", "mat_b"):
            a, b = (mat[pid], ctl[pid]) if order == "mat_a" else (ctl[pid], mat[pid])
            requests.append({"prompt_id": pid, "order": order, "judge_prompt": RUBRIC.format(
                prompt=mat[pid]["prompt"], response_a=a["response"], response_b=b["response"])})

    tokenizer, model = load_tokenizer(args.judge_model), load_model(args.judge_model, device_map="auto")
    outputs = batched_generate(model, tokenizer, [chat(tokenizer, r["judge_prompt"]) for r in requests],
                               args.max_new_tokens, args.batch_size, max_length=4096, do_sample=False)
    rows = []
    for req, (text, _, _) in zip(requests, outputs):
        scores = parse_judgment(text)
        sign = -1 if req["order"] == "mat_a" else 1
        rows.append({"prompt_id": req["prompt_id"], "order": req["order"], "judge_model": args.judge_model,
                     "judge_output": text,
                     **{axis: None if scores is None else sign * scores[axis] for axis in AXES}})
    write_jsonl(out_path, rows)
    print(f"{out_path}: {len(rows)} judgments, {sum(r['verbosity'] is None for r in rows)} unparsable")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge_model", required=True)
    parser.add_argument("--name", required=True, help="short judge name used in the output file")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mat_label", default="mat")
    parser.add_argument("--control_label", default="control")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    main(parser.parse_args())
