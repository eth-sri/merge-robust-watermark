from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class WatermarkConfig(BaseModel):
    type: Literal["kgw", "aar", "kth"] = "kgw"

    gamma: float = 0.25
    delta: float = 2.0
    seeding_scheme: str = "simple_1"
    top_k: Optional[int] = None

    aar_context_size: int = 2
    aar_seed: int = 42

    kth_key_size: int = 256
    kth_seed: int = 42
    kth_n_runs: int = 100


class MergeAdversaryConfig(BaseModel):
    alpha_min: float = 0.1
    alpha_max: float = 1.0

    @model_validator(mode="after")
    def _check(self):
        if not 0.0 < self.alpha_min <= self.alpha_max <= 1.0:
            raise ValueError("need 0 < alpha_min <= alpha_max <= 1")
        return self


class SampleConfig(BaseModel):
    data_dir: str
    num_samples: int = 640_000
    prompt_length: int = 50
    completion_length: int = 256
    batch_size: int = 32
    prompt_seed: int = 1042


class TrainConfig(BaseModel):
    base_model: str
    dtype: str = "bfloat16"
    sequence_length: int = 512
    effective_batch_size: int = 64
    streaming: bool = True

    watermark: WatermarkConfig = Field(default_factory=WatermarkConfig)
    merge_adversary: Optional[MergeAdversaryConfig] = None
    merge_reference: Optional[str] = None
    datasets: List[str]
    proportions: List[float]

    teacher_on_cpu: bool = False
    checkpoint_merge: bool = True
    samples: Optional[SampleConfig] = None

    training_args: Dict[str, Any] = Field(default_factory=dict)
    output_dir: str

    @classmethod
    def from_yaml(cls, path: str, overrides: Optional[List[str]] = None) -> "TrainConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        for item in overrides or []:
            key, value = item.split("=", 1)
            node = data
            *parents, leaf = key.split(".")
            for p in parents:
                node = node.setdefault(p, {})
            node[leaf] = _parse_scalar(value)
        return cls.model_validate(data)


def _parse_scalar(value: str):
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    if value.lower() in ("null", "none"):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value
