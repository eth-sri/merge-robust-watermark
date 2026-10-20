import inspect
import json
import os
import shutil

import torch
import mergekit.plan
from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge
from pydantic import BaseModel

for _, _cls in inspect.getmembers(mergekit.plan, inspect.isclass):
    if issubclass(_cls, BaseModel) and _cls is not BaseModel:
        try:
            _cls.model_rebuild()
        except Exception:
            pass

TASK_VECTOR_METHODS = ("ties", "dare_ties", "breadcrumbs", "breadcrumbs_ties")


def _num_layers(model: str) -> int:
    if os.path.isdir(model):
        path = os.path.join(model, "config.json")
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(model, "config.json")
    with open(path) as f:
        return json.load(f)["num_hidden_layers"]


def mergekit_config(method, models, weights=None, t=None, base_model=None, density=None,
                    gamma=None, dtype="bfloat16") -> dict:
    if method == "linear":
        assert weights is not None and len(weights) == len(models)
        return {"merge_method": "linear", "dtype": dtype, "base_model": models[0],
                "models": [{"model": m, "parameters": {"weight": w}} for m, w in zip(models, weights)]}

    if method == "slerp":
        assert len(models) == 2 and t is not None and base_model in models
        n_layers = _num_layers(models[0])
        return {"merge_method": "slerp", "dtype": dtype, "base_model": base_model, "parameters": {"t": t},
                "slices": [{"sources": [{"model": m, "layer_range": [0, n_layers]} for m in models]}]}

    if method in TASK_VECTOR_METHODS:
        assert base_model is not None and density is not None
        others = [m for m in models if m != base_model]
        assert weights is not None and len(weights) == len(others)
        params = {"density": density}
        if method.startswith("breadcrumbs"):
            params["gamma"] = gamma
        return {"merge_method": method, "dtype": dtype, "base_model": base_model,
                "parameters": params,
                "models": [{"model": m, "parameters": {"weight": w, **params}} for m, w in zip(others, weights)]}

    raise ValueError(f"Unknown merge method: {method}")


def pairwise_config(method, a, b, alpha, density=None, gamma=None, dtype="bfloat16") -> dict:
    if method == "linear":
        return mergekit_config("linear", [a, b], weights=[1.0 - alpha, alpha], dtype=dtype)
    if method == "slerp":
        return mergekit_config("slerp", [a, b], t=alpha, base_model=a, dtype=dtype)
    return mergekit_config(method, [a, b], weights=[alpha], base_model=a, density=density,
                           gamma=gamma, dtype=dtype)


def anchored_config(method, anchor, models, weights, density=None, gamma=None, dtype="bfloat16") -> dict:
    return mergekit_config(method, [anchor, *models], weights=weights, base_model=anchor, density=density,
                           gamma=gamma, dtype=dtype)


def merge(config: dict, output_dir: str, tokenizer_source: str = None, config_source: str = None,
          seed: int = 42) -> None:
    torch.manual_seed(seed)
    if tokenizer_source is not None:
        config = {**config, "tokenizer": {"source": tokenizer_source}}
    os.makedirs(output_dir, exist_ok=True)
    run_merge(MergeConfiguration.model_validate(config), out_path=output_dir,
              options=MergeOptions(cuda=torch.cuda.is_available(), copy_tokenizer=True,
                                   lazy_unpickle=True, low_cpu_memory=False))
    if config_source is not None:
        if os.path.isdir(config_source):
            source = os.path.join(config_source, "config.json")
        else:
            from huggingface_hub import hf_hub_download
            source = hf_hub_download(config_source, "config.json")
        shutil.copy(source, os.path.join(output_dir, "config.json"))
