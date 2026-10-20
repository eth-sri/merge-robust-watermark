import argparse
import os
import subprocess

import yaml


def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_args = ",".join([f"pretrained={args.model}"] + [f"{k}={v}" for k, v in cfg["model_args"].items()])
    for bench in cfg["benchmarks"]:
        if args.task and bench["task"] != args.task:
            continue
        subprocess.run([
            "lm_eval", "--model", "hf", "--model_args", model_args,
            "--tasks", bench["task"], "--num_fewshot", str(bench["num_fewshot"]),
            "--batch_size", str(cfg["batch_size"]), "--device", "cuda",
            "--output_path", os.path.join(args.output_dir, bench["task"]),
        ], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default="configs/eval/benchmarks.yaml")
    parser.add_argument("--task", help="run only this task")
    main(parser.parse_args())
