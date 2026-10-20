from dataclasses import dataclass

import torch


@dataclass
class GaussMarkSecret:
    layer_name: str
    sigma: float
    delta: torch.Tensor
    seed: int

    def save(self, path: str) -> None:
        torch.save({"layer_name": self.layer_name, "sigma": self.sigma,
                    "delta": self.delta.cpu(), "seed": self.seed}, path)

    @classmethod
    def load(cls, path: str) -> "GaussMarkSecret":
        d = torch.load(path, map_location="cpu", weights_only=False)
        return cls(layer_name=d["layer_name"], sigma=float(d["sigma"]),
                   delta=d["delta"], seed=int(d["seed"]))


def make_secret(model: torch.nn.Module, layer_name: str, sigma: float, seed: int = 1234) -> GaussMarkSecret:
    params = dict(model.named_parameters())
    if layer_name not in params:
        raise KeyError(f"Layer '{layer_name}' not found in model.")
    g = torch.Generator(device="cpu").manual_seed(seed)
    delta = torch.randn(params[layer_name].shape, generator=g, dtype=torch.float32) * sigma
    return GaussMarkSecret(layer_name=layer_name, sigma=sigma, delta=delta, seed=seed)


@torch.no_grad()
def apply_secret(model: torch.nn.Module, secret: GaussMarkSecret, sign: float = 1.0) -> None:
    w = dict(model.named_parameters())[secret.layer_name]
    w.add_(sign * secret.delta.to(device=w.device, dtype=w.dtype))


@torch.no_grad()
def sequence_logprob(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits
    log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    return log_probs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1).sum(dim=-1)
