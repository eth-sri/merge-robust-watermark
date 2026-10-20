import json
import os
import shutil

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
TRANSFORMERS_MAJOR = int(transformers.__version__.split(".")[0])


def dtype_kwarg(dtype):
    return {("dtype" if TRANSFORMERS_MAJOR >= 5 else "torch_dtype"): DTYPES[dtype]}


def _config_dict(path):
    if os.path.isdir(path):
        file = os.path.join(path, "config.json")
    else:
        from huggingface_hub import hf_hub_download
        file = hf_hub_download(path, "config.json")
    with open(file) as f:
        return json.load(f)


def check_rope_config(path):
    cfg = _config_dict(path)
    if TRANSFORMERS_MAJOR < 5 and "rope_parameters" in cfg and "rope_theta" not in cfg:
        raise RuntimeError(
            f"{path} stores RoPE settings as `rope_parameters` (transformers >= 5), which "
            f"transformers {transformers.__version__} ignores. Run "
            f"`python -c \"from mat.utils import add_legacy_rope_keys; add_legacy_rope_keys('{path}')\"` "
            f"on a local copy, or use transformers >= 5."
        )


def add_legacy_rope_keys(model_dir):
    file = os.path.join(model_dir, "config.json")
    with open(file) as f:
        cfg = json.load(f)
    rope = cfg.get("rope_parameters")
    if rope is None or "rope_theta" in cfg:
        return
    cfg["rope_theta"] = rope["rope_theta"]
    if rope.get("rope_type", "default") != "default":
        cfg["rope_scaling"] = {k: v for k, v in rope.items() if k != "rope_theta"}
    with open(file, "w") as f:
        json.dump(cfg, f, indent=2)


TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "tokenizer.model", "added_tokens.json", "chat_template.jinja")


def copy_tokenizer_files(src, dst):
    if not os.path.isdir(src):
        from huggingface_hub import snapshot_download
        src = snapshot_download(src, allow_patterns=list(TOKENIZER_FILES))
    for name in TOKENIZER_FILES:
        if os.path.exists(os.path.join(dst, name)):
            os.remove(os.path.join(dst, name))
        if os.path.exists(os.path.join(src, name)):
            shutil.copy(os.path.join(src, name), os.path.join(dst, name))


def gradient_accumulation_steps(effective_batch_size, per_device_batch_size, accumulate: bool = True):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    steps, remainder = divmod(effective_batch_size, per_device_batch_size * world_size)
    if steps == 0 or remainder:
        raise SystemExit(f"effective_batch_size {effective_batch_size} is not a multiple of "
                         f"per_device_train_batch_size {per_device_batch_size} x {world_size} processes")
    if steps > 1 and not accumulate:
        raise SystemExit(f"effective_batch_size {effective_batch_size} would need {steps} gradient accumulation steps, "
                         f"which the pinned transformers version scales inconsistently: set "
                         f"per_device_train_batch_size to {effective_batch_size // world_size} or use more processes")
    return steps


def load_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(path, dtype="bfloat16", quantization=None, device_map="cuda"):
    check_rope_config(path)
    kwargs = dict(**dtype_kwarg(dtype), device_map=device_map)
    if quantization == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization in ("fp4", "nf4"):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type=quantization,
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False)
    elif quantization is not None:
        raise ValueError(f"Unknown quantization: {quantization}")
    return AutoModelForCausalLM.from_pretrained(path, **kwargs).eval()
