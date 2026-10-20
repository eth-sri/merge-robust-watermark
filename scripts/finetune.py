import argparse
import glob
import os

import yaml
from accelerate import PartialState
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

from mat.data import SFT_CORPORA
from mat.utils import (add_legacy_rope_keys, check_rope_config, copy_tokenizer_files, dtype_kwarg,
                       gradient_accumulation_steps, load_tokenizer)


def main(cfg, resume: bool):
    tokenizer = load_tokenizer(cfg["base_model"])
    if tokenizer.chat_template is None:
        raise SystemExit(f"{cfg['base_model']} has no chat template, but the SFT examples are chat-formatted.")
    with PartialState().main_process_first():
        dataset = SFT_CORPORA[cfg["dataset"]](**cfg.get("dataset_kwargs", {}))

    check_rope_config(cfg["base_model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], **dtype_kwarg("float32"))
    accumulation = gradient_accumulation_steps(cfg["effective_batch_size"],
                                               cfg["training_args"]["per_device_train_batch_size"])
    os.environ["ACCELERATE_GRADIENT_ACCUMULATION_STEPS"] = str(accumulation)
    args = SFTConfig(output_dir=cfg["output_dir"], max_length=cfg["sequence_length"], packing=False,
                     gradient_accumulation_steps=accumulation, **cfg["training_args"])
    if args.n_gpu > 1:
        raise SystemExit("Several GPUs in one process (DataParallel) would multiply the batch size: "
                         "launch one process per GPU with accelerate launch, or make one GPU visible")
    trainer = SFTTrainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer)
    trainer.train(resume_from_checkpoint=resume or None)
    trainer.save_model()
    if PartialState().is_main_process:
        for d in [cfg["output_dir"], *glob.glob(os.path.join(cfg["output_dir"], "checkpoint-*"))]:
            add_legacy_rope_keys(d)
            copy_tokenizer_files(cfg["base_model"], d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="per-domain recipe (configs/sft/*.yaml)")
    parser.add_argument("--base_model", required=True, help="watermarked checkpoint to finetune")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg.update(base_model=args.base_model, output_dir=args.output_dir)
    main(cfg, args.resume)
