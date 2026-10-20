import argparse
import os

import yaml
from transformers import AutoModelForCausalLM

from mat.utils import dtype_kwarg, load_tokenizer
from mat.watermarks.gaussmark import apply_secret, make_secret


def main(cfg):
    os.makedirs(cfg["output_dir"], exist_ok=True)
    tokenizer = load_tokenizer(cfg["base_model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], **dtype_kwarg("bfloat16"))
    secret = make_secret(model, cfg["layer"], sigma=cfg["sigma"], seed=cfg["seed"])
    apply_secret(model, secret)
    model.save_pretrained(cfg["output_dir"], safe_serialization=True)
    tokenizer.save_pretrained(cfg["output_dir"])
    secret.save(os.path.join(cfg["output_dir"], "gaussmark_secret.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    with open(parser.parse_args().config) as f:
        main(yaml.safe_load(f))
