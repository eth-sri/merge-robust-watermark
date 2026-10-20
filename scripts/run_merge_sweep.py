import argparse
import os
import shutil
import subprocess
import sys

import yaml

from mat.merging import anchored_config, merge, pairwise_config

HERE = os.path.dirname(os.path.abspath(__file__))


def sweep_tag(scenario, method, params):
    return f"{scenario}_{method}" + (f"_d{params['density']}" if "density" in params else "")


class Resolver:

    def __init__(self, cfg, variant, method, method_params, output_root):
        self.cfg, self.method, self.params = cfg, method, method_params
        self.models = {**cfg["unwatermarked"], **cfg["variants"][variant]}
        self.inner_root = os.path.join(output_root, variant, "_inner")

    def path(self, token):
        if token in self.models:
            return self.models[token]
        if token in self.cfg["scenarios"]:
            out = os.path.join(self.inner_root, f"{token}_linear_{self.cfg['inner_alpha']}")
            if not os.path.exists(os.path.join(out, "config.json")):
                self.build(token, self.cfg["inner_alpha"], out, method="linear", params={})
            return out
        raise KeyError(f"Unknown model or scenario: {token}")

    def build(self, scenario, alpha, out, method=None, params=None):
        method = method or self.method
        params = self.params if params is None else params
        spec = self.cfg["scenarios"][scenario]
        if "anchor" in spec:
            models = list(spec["vectors"])
            weights = [alpha if w == "alpha" else float(w) for w in spec["vectors"].values()]
            config = anchored_config(method, self.path(spec["anchor"]), [self.path(m) for m in models],
                                     weights, **params)
        else:
            a, b = spec["A"], spec["B"]
            if method == "ties" and spec.get("ties_base") == b:
                a, b = b, a
            config = pairwise_config(method, self.path(a), self.path(b), alpha, **params)
        tokenizer = self.path(spec["tokenizer_source"]) if "tokenizer_source" in spec else None
        config_source = self.path(spec["config_source"]) if "config_source" in spec else None
        merge(config, out, tokenizer_source=tokenizer, config_source=config_source)


def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    params = dict(cfg["methods"][args.method])
    if args.density is not None:
        params["density"] = args.density
    resolver = Resolver(cfg, args.variant, args.method, params, args.output_root)
    variant = cfg["variants"][args.variant]
    domains = ["en"] if "gaussmark_secret" in variant else args.domains

    for alpha in args.alphas:
        out = os.path.join(args.output_root, args.variant, sweep_tag(args.scenario, args.method, params),
                           f"alpha_{alpha}")
        model_dir = os.path.join(out, "model")
        todo = [d for d in domains if not os.path.exists(os.path.join(out, "eval", f"detection_{d}.json"))]
        if (todo or args.keep_models) and not os.path.exists(os.path.join(model_dir, "config.json")):
            resolver.build(args.scenario, alpha, model_dir)
        for domain in todo:
            if "gaussmark_secret" in variant:
                cmd = ["eval_gaussmark.py", "--secret", variant["gaussmark_secret"]]
            else:
                cmd = ["eval_detection.py", "--watermark_config", variant["watermark_config"], "--domain", domain]
            subprocess.run([sys.executable, os.path.join(HERE, cmd[0]), *cmd[1:], "--model", model_dir,
                            "--n_samples", str(args.n_samples), "--output_dir", os.path.join(out, "eval")],
                           check=True)
        if not args.keep_models and os.path.exists(model_dir):
            shutil.rmtree(model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/merge/scenarios.yaml")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--method", required=True,
                        choices=["linear", "slerp", "ties", "dare_ties", "breadcrumbs", "breadcrumbs_ties"])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--density", type=float, help="override the method's density (e.g. DARE-TIES 0.3)")
    parser.add_argument("--domains", nargs="+", default=["en", "de", "math"])
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--output_root", default="output/merges")
    parser.add_argument("--keep_models", action="store_true")
    main(parser.parse_args())
