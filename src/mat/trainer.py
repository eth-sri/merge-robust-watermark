import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import Trainer


def _clean_name(name: str) -> str:
    return name.removeprefix("module.")


class MATTrainer(Trainer):
    def __init__(self, watermark, teacher_model, merge_adversary=None, reference_model=None,
                 checkpoint_merge: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.watermark = watermark
        self.teacher = teacher_model.eval() if teacher_model is not None else None
        self.merge_adversary = merge_adversary
        self.checkpoint_merge = checkpoint_merge
        reference = reference_model if reference_model is not None else teacher_model
        self.reference_params = dict(reference.named_parameters()) if merge_adversary else None

    @staticmethod
    def _masked_mean(per_token, attention_mask):
        mask = attention_mask.float()
        return ((per_token * mask).sum(-1) / mask.sum(-1).clamp(min=1)).mean()

    def _kl_to_target(self, student_logits, target_logits, attention_mask):
        kl = F.kl_div(F.log_softmax(student_logits, dim=-1), F.log_softmax(target_logits, dim=-1),
                      reduction="none", log_target=True).sum(-1)
        return self._masked_mean(kl, attention_mask)

    def _ce_to_tokens(self, student_logits, target_tokens, attention_mask):
        nll = -F.log_softmax(student_logits, dim=-1).gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)
        return self._masked_mean(nll, attention_mask)

    def _watermark_loss(self, student_logits, teacher_logits, input_ids, attention_mask):
        if getattr(self.watermark, "use_argmax_loss", False):
            with torch.no_grad():
                targets = self.watermark.watermark_logits_argmax(input_ids, teacher_logits)
            return self._ce_to_tokens(student_logits, targets, attention_mask)
        target_logits = self.watermark.watermark_logits(input_ids, teacher_logits.clone())
        return self._kl_to_target(student_logits, target_logits, attention_mask)

    def _merged_forward(self, model, inputs, alpha: float):
        module = model.module if hasattr(model, "module") else model

        def build():
            return {
                _clean_name(n): alpha * p + (1.0 - alpha)
                * self.reference_params[_clean_name(n)].to(device=p.device, dtype=p.dtype)
                for n, p in model.named_parameters()
            }

        def forward(kwargs):
            return torch.func.functional_call(module, build(), args=(), kwargs=kwargs)

        if self.checkpoint_merge and self.model.training:
            return torch.utils.checkpoint.checkpoint(forward, inputs, use_reentrant=False)
        return forward(inputs)

    def _forward(self, model, inputs):
        if self.merge_adversary is None:
            return model(**inputs)
        alpha = torch.empty(1).uniform_(self.merge_adversary.alpha_min, self.merge_adversary.alpha_max).item()
        self._last_alpha = alpha
        return self._merged_forward(model, inputs, alpha)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs.pop("labels", None)
        attention_mask = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))

        device = next(model.parameters()).device
        offloaded = next(self.teacher.parameters()).device.type == "cpu"
        if offloaded:
            self.teacher.to(device)
        with torch.no_grad():
            teacher_logits = self.teacher(**inputs).logits
        if offloaded:
            self.teacher.to("cpu")
            torch.cuda.empty_cache()

        outputs = self._forward(model, inputs)
        loss = self._watermark_loss(outputs.logits, teacher_logits, inputs["input_ids"], attention_mask)
        return (loss, outputs) if return_outputs else loss

    def train(self, resume_from_checkpoint=None, *args, **kwargs):
        zero3 = self.is_deepspeed_enabled and self.accelerator.state.deepspeed_plugin.zero_stage == 3
        wrapped = self.is_fsdp_enabled or zero3 or (
            (self.args.world_size > 1 or self.args.n_gpu > 1) and not self.is_deepspeed_enabled)
        if self.merge_adversary is not None and wrapped:
            raise SystemExit("MAT's merged forward pass bypasses DDP, DataParallel and FSDP: "
                             "launch it with DeepSpeed ZeRO-1/2")
        if resume_from_checkpoint and self.is_deepspeed_enabled:
            raise SystemExit("--resume is not supported with DeepSpeed: its checkpoints store model weights only")
        return super().train(resume_from_checkpoint, *args, **kwargs)

    def log(self, logs, *args, **kwargs):
        if hasattr(self, "_last_alpha"):
            logs["alpha"] = self._last_alpha
        super().log(logs, *args, **kwargs)

    def _save_checkpoint(self, model, trial, metrics=None):
        if not self.is_deepspeed_enabled:
            return super()._save_checkpoint(model, trial)
        output_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.state.global_step}")
        self.save_model(output_dir, _internal_call=True)
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if not self.is_deepspeed_enabled:
            return super().save_model(output_dir, _internal_call)
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        state_dict = self.accelerator.get_state_dict(self.model_wrapped)
        if self.args.should_save and state_dict is not None:
            self.model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=True)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()


class SampleMATTrainer(MATTrainer):
    def __init__(self, merge_adversary, reference_model, checkpoint_merge: bool = True, *args, **kwargs):
        super().__init__(None, None, merge_adversary, reference_model, checkpoint_merge, *args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        return Trainer.compute_loss(self, lambda **batch: self._forward(model, batch), inputs, return_outputs, **kwargs)
