import argparse
import glob
import os

from accelerate import PartialState
from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, default_data_collator

from mat.config import TrainConfig
from mat.trainer import SampleMATTrainer
from mat.utils import check_rope_config, dtype_kwarg, gradient_accumulation_steps, load_tokenizer


def main(config, resume: bool):
    shards = sorted(glob.glob(os.path.join(config.samples.data_dir, "shard_*.jsonl")))
    if not shards:
        raise SystemExit(f"No shard_*.jsonl in {config.samples.data_dir}: run scripts/generate_kth_samples.py first")
    tokenizer = load_tokenizer(config.base_model)

    def tokenize(batch):
        out = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=config.sequence_length)
        out["labels"] = out["input_ids"].copy()
        return out

    with PartialState().main_process_first():
        raw = load_dataset("json", data_files=shards, split="train")
        dataset = raw.map(tokenize, batched=True, remove_columns=raw.column_names)

    check_rope_config(config.base_model)
    model = AutoModelForCausalLM.from_pretrained(config.base_model, **dtype_kwarg("float32"))
    reference = None
    if config.merge_adversary is not None:
        reference = AutoModelForCausalLM.from_pretrained(config.merge_reference or config.base_model,
                                                         **dtype_kwarg(config.dtype))
        reference.requires_grad_(False).eval()
        if not config.teacher_on_cpu:
            reference.to(f"cuda:{os.environ.get('LOCAL_RANK', 0)}")

    accumulation = gradient_accumulation_steps(config.effective_batch_size,
                                               config.training_args["per_device_train_batch_size"], accumulate=False)
    os.environ["ACCELERATE_GRADIENT_ACCUMULATION_STEPS"] = str(accumulation)
    args = TrainingArguments(output_dir=config.output_dir, gradient_accumulation_steps=accumulation,
                             **config.training_args)
    if args.n_gpu > 1:
        raise SystemExit("Several GPUs in one process (DataParallel) would multiply the batch size: "
                         "launch one process per GPU with accelerate launch, or make one GPU visible")
    if reference is None:
        trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=default_data_collator,
                          processing_class=tokenizer)
    else:
        trainer = SampleMATTrainer(merge_adversary=config.merge_adversary, reference_model=reference,
                                   checkpoint_merge=config.checkpoint_merge, model=model, args=args,
                                   train_dataset=dataset, data_collator=default_data_collator,
                                   processing_class=tokenizer)
    trainer.train(resume_from_checkpoint=resume or None)
    trainer.save_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sampling distillation: train on generated watermarked samples.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(TrainConfig.from_yaml(args.config), args.resume)
