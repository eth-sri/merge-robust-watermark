import argparse
import os

from transformers import AutoModelForCausalLM, TrainingArguments, default_data_collator

from mat.config import TrainConfig
from mat.data import distillation_mixture
from mat.trainer import MATTrainer
from mat.utils import check_rope_config, dtype_kwarg, gradient_accumulation_steps, load_tokenizer
from mat.watermarks import build_watermark


def main(config: TrainConfig, resume: bool = False):
    tokenizer = load_tokenizer(config.base_model)
    check_rope_config(config.base_model)

    train_ds = distillation_mixture(tokenizer, config.datasets, config.proportions,
                                    config.sequence_length, streaming=config.streaming)

    student = AutoModelForCausalLM.from_pretrained(config.base_model, **dtype_kwarg("float32"))
    teacher = AutoModelForCausalLM.from_pretrained(config.base_model, **dtype_kwarg(config.dtype))
    teacher.requires_grad_(False).eval()
    reference = None
    if config.merge_adversary is not None and config.merge_reference not in (None, config.base_model):
        reference = AutoModelForCausalLM.from_pretrained(config.merge_reference, **dtype_kwarg(config.dtype))
        reference.requires_grad_(False).eval()
    if not config.teacher_on_cpu:
        device = f"cuda:{os.environ.get('LOCAL_RANK', 0)}"
        teacher.to(device)
        if reference is not None:
            reference.to(device)

    accumulation = gradient_accumulation_steps(config.effective_batch_size,
                                               config.training_args["per_device_train_batch_size"], accumulate=False)
    os.environ["ACCELERATE_GRADIENT_ACCUMULATION_STEPS"] = str(accumulation)
    training_args = TrainingArguments(output_dir=config.output_dir, gradient_accumulation_steps=accumulation,
                                      **config.training_args)
    if training_args.n_gpu > 1:
        raise SystemExit("Several GPUs in one process (DataParallel) would multiply the batch size: "
                         "launch one process per GPU with accelerate launch, or make one GPU visible")
    trainer = MATTrainer(
        watermark=build_watermark(config.watermark, tokenizer),
        teacher_model=teacher,
        merge_adversary=config.merge_adversary,
        reference_model=reference,
        checkpoint_merge=config.checkpoint_merge,
        model=student,
        args=training_args,
        train_dataset=train_ds,
        data_collator=default_data_collator,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=resume or None)
    trainer.save_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--resume", action="store_true", help="resume from the last checkpoint in output_dir")
    args = parser.parse_args()
    main(TrainConfig.from_yaml(args.config, args.override), resume=args.resume)
