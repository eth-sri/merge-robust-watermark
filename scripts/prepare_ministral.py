import argparse
import json
import os
import shutil

from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

PREFIX = "language_model."
TOKENIZER_FILES = ("chat_template.jinja", "generation_config.json", "special_tokens_map.json",
                   "tokenizer.json", "tokenizer_config.json")


def main(args):
    src = snapshot_download(args.source, allow_patterns=["config.json", "model*.safetensors",
                                                         "model.safetensors.index.json", *TOKENIZER_FILES])
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(src, "config.json")) as f:
        outer = json.load(f)
    with open(os.path.join(src, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]

    new_map = {}
    for shard in sorted(set(weight_map.values())):
        tensors = {}
        with safe_open(os.path.join(src, shard), framework="pt") as fh:
            for name in fh.keys():
                if name.startswith(PREFIX):
                    tensors[name[len(PREFIX):]] = fh.get_tensor(name)
        if tensors:
            save_file(tensors, os.path.join(args.output, shard), metadata={"format": "pt"})
            new_map.update({name: shard for name in tensors})
    with open(os.path.join(args.output, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": new_map}, f, indent=2)

    config = dict(outer["text_config"])
    config.update(architectures=["Ministral3ForCausalLM"], bos_token_id=outer.get("bos_token_id", 1),
                  eos_token_id=outer.get("eos_token_id", 2), pad_token_id=outer.get("pad_token_id", 11),
                  dtype=outer.get("dtype", "bfloat16"))
    with open(os.path.join(args.output, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    for name in TOKENIZER_FILES:
        if os.path.isfile(os.path.join(src, name)):
            shutil.copy2(os.path.join(src, name), args.output)
    path = os.path.join(args.output, "tokenizer_config.json")
    with open(path) as f:
        tok_cfg = json.load(f)
    tok_cfg["fix_mistral_regex"] = True
    with open(path, "w") as f:
        json.dump(tok_cfg, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="mistralai/Ministral-3-3B-Instruct-2512-BF16")
    parser.add_argument("--output", default="output/ministral3b_text")
    main(parser.parse_args())
