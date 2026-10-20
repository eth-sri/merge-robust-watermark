import argparse
import json
import os
import random

import torch
from torch.func import functional_call

from mat.utils import load_tokenizer
from regional_merge import RegionalMerge, heldout_nll, next_token_nll, utility_sequences, variant_models


def bounded(raw, target, radius, weights):
    t = torch.tanh(raw)
    centered = t - (t * weights).sum()
    return target + radius * centered / (1.0 + centered.abs().max())


def main(args):
    report_path = os.path.join(args.output_dir, "coefficients.json")
    if os.path.exists(report_path):
        print(f"{report_path} exists, skipping")
        return
    torch.manual_seed(args.seed)
    base, wm_model, partner = variant_models(args.variant, args.config)
    tokenizer = load_tokenizer(base)
    utility_data = dict(n=args.n_sequences, length=args.sequence_length, en_skip=args.en_skip, seed=args.seed)
    train, heldout = utility_sequences(tokenizer, **utility_data)
    rm = RegionalMerge(base, wm_model, partner)

    targets = (args.wm_weight, 1.0 - args.wm_weight)
    weights = [torch.tensor(e, dtype=torch.float32, device="cuda") for e in (rm.wm_energy, rm.partner_energy)]
    radii = [args.radius_scale * min(t, 1.0 - t) for t in targets]
    raw = [torch.zeros(len(rm.groups), device="cuda", requires_grad=True) for _ in targets]

    def coefficients():
        return [bounded(r, t, rad, w) for r, t, rad, w in zip(raw, targets, radii, weights)]

    with torch.no_grad():
        params = rm.merged_parameters(*coefficients())
        nll_uniform = heldout_nll(
            lambda ids: functional_call(rm.model, params, (ids,), {"use_cache": False}).logits, heldout)
        del params
    print(f"held-out NLL, uniform merge: {nll_uniform}")

    optimizer = torch.optim.Adam(raw, lr=args.lr)
    rng = random.Random(args.seed)
    domains = sorted(train)
    history = []
    for step in range(args.steps):
        domain = domains[step % len(domains)]
        ids = rng.choice(train[domain])[None].cuda()
        a, b = coefficients()
        logits = functional_call(rm.model, rm.merged_parameters(a, b), (ids,), {"use_cache": False}).logits
        nll = next_token_nll(logits, ids)
        pull = sum((w * (c - t) ** 2).sum() for w, c, t in zip(weights, (a, b), targets))
        loss = nll + args.coefficient_l2 * pull
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw, args.max_grad_norm)
        optimizer.step()
        if step % 10 == 0 or step == args.steps - 1:
            row = dict(step=step, domain=domain, nll=nll.item(), wm=a.tolist(), partner=b.tolist())
            history.append(row)
            print(row, flush=True)

    a, b = [c.detach().double().cpu().numpy() for c in coefficients()]
    rm.apply(a, b)
    nll_learned = heldout_nll(lambda ids: rm.model(ids, use_cache=False).logits, heldout)
    print(f"held-out NLL, learned merge: {nll_learned}")

    model_dir = os.path.join(args.output_dir, "model")
    rm.model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    report = {
        "variant": args.variant, "base": base, "wm_model": wm_model, "partner": partner,
        "wm_weight": targets[0], "partner_weight": targets[1], "utility_data": utility_data,
        "wm_energy": dict(zip(rm.groups, rm.wm_energy.tolist())),
        "partner_energy": dict(zip(rm.groups, rm.partner_energy.tolist())),
        "wm_coefficients": dict(zip(rm.groups, a.tolist())),
        "partner_coefficients": dict(zip(rm.groups, b.tolist())),
        "heldout_nll_uniform": nll_uniform, "heldout_nll_learned": nll_learned,
        "history": history, "args": vars(args),
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", required=True, help="variant in configs/merge/scenarios.yaml (kgwd, mat)")
    parser.add_argument("--config", default="configs/merge/scenarios.yaml")
    parser.add_argument("--wm_weight", type=float, default=0.7, help="watermarked share (OpenMath: 1 - it)")
    parser.add_argument("--n_sequences", type=int, default=128, help="per domain, half of them held out")
    parser.add_argument("--sequence_length", type=int, default=128)
    parser.add_argument("--en_skip", type=int, default=4000, help="C4 texts skipped before the English data")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--coefficient_l2", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--radius_scale", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    main(parser.parse_args())
