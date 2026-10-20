import math
import re

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset

from mat.evaluation import web_texts
from mat.utils import load_model

LAYER = re.compile(r"\.layers\.(\d+)\.")
MATH_SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                 "number_theory", "prealgebra", "precalculus")


def variant_models(variant, config="configs/merge/scenarios.yaml"):
    with open(config) as f:
        cfg = yaml.safe_load(f)
    models = {**cfg["unwatermarked"], **cfg["variants"][variant]}
    return models["B_prime"], models["B"], models["OpenMath"]


def parameter_group(name, n_layers, n_blocks=4):
    match = LAYER.search(name)
    if match is None:
        return "embedding_head_other"
    size = math.ceil(n_layers / n_blocks)
    start = min(int(match.group(1)) // size, n_blocks - 1) * size
    return f"layers_{start:02d}_{min(n_layers, start + size) - 1:02d}"


def task_vector(path, base_model):
    other = dict(load_model(path, device_map="cpu").named_parameters())
    base = dict(base_model.named_parameters())
    assert other.keys() == base.keys(), f"{path} does not match the base architecture"
    return {name: (other[name].to(p.device) - p).detach() for name, p in base.items()}


class RegionalMerge:
    def __init__(self, base, wm_model, partner):
        self.model = load_model(base)
        self.model.requires_grad_(False)
        self.wm_delta = task_vector(wm_model, self.model)
        self.partner_delta = task_vector(partner, self.model)
        n_layers = self.model.config.num_hidden_layers
        group_of = {name: parameter_group(name, n_layers) for name, _ in self.model.named_parameters()}
        self.groups = sorted(set(group_of.values()))
        self.index = {name: self.groups.index(g) for name, g in group_of.items()}
        self.wm_energy = self._energy(self.wm_delta)
        self.partner_energy = self._energy(self.partner_delta)
        self.base = None

    def _energy(self, delta):
        energy = np.zeros(len(self.groups))
        for name, d in delta.items():
            energy[self.index[name]] += d.float().square().sum().item()
        return energy / energy.sum()

    def merged_parameters(self, a, b):
        assert self.base is None, "merged_parameters() needs the unmodified B' weights: call it before apply()"
        return {name: p + a[self.index[name]].to(p.dtype) * self.wm_delta[name]
                + b[self.index[name]].to(p.dtype) * self.partner_delta[name]
                for name, p in self.model.named_parameters()}

    @torch.no_grad()
    def apply(self, a, b):
        if self.base is None:
            self.base = {name: p.detach().cpu() for name, p in self.model.named_parameters()}
        for name, p in self.model.named_parameters():
            g = self.index[name]
            merged = self.base[name].to(p.device, torch.float32)
            merged.add_(self.wm_delta[name].float(), alpha=float(a[g]))
            merged.add_(self.partner_delta[name].float(), alpha=float(b[g]))
            p.copy_(merged)


def utility_sequences(tokenizer, n=128, length=128, en_skip=4000, seed=42):
    bos = torch.tensor([tokenizer.bos_token_id])
    english = [torch.cat([bos, t]) for t in web_texts("en", tokenizer, n, length - 1, length - 1, skip=en_skip)]
    per_subject = []
    for i, subject in enumerate(MATH_SUBJECTS):
        rows = load_dataset("EleutherAI/hendrycks_math", subject, split="train").shuffle(seed=seed + i)
        ids = (tokenizer(row["solution"]).input_ids for row in rows)
        per_subject.append([torch.tensor(x[:length]) for x in ids if len(x) >= length])
    math_rows = [rows[j] for j in range(max(map(len, per_subject))) for rows in per_subject if j < len(rows)][:n]

    train, heldout = {}, {}
    for i, (domain, rows) in enumerate([("en", english), ("math", math_rows)]):
        assert len(rows) == n, f"only {len(rows)} usable {domain} sequences"
        order = torch.randperm(n, generator=torch.Generator().manual_seed(seed + i))
        ids = torch.stack(rows)[order]
        train[domain], heldout[domain] = ids[: n // 2], ids[n // 2:]
    return train, heldout


def next_token_nll(logits, ids):
    return F.cross_entropy(logits[:, :-1].float().transpose(1, 2), ids[:, 1:])


@torch.no_grad()
def heldout_nll(logits_fn, sequences):
    result = {}
    for domain, rows in sequences.items():
        losses = []
        for x in rows:
            ids = x[None].cuda()
            losses.append(next_token_nll(logits_fn(ids), ids).item())
        result[domain] = float(np.mean(losses))
    return result
