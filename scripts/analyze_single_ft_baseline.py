import argparse
import json
import math
import os


def load(path, domain):
    with open(os.path.join(path, f"detection_{domain}.json")) as f:
        m = json.load(f)["metrics"]
    return m["tpr_at_1pct_fpr"], m["n_samples"]


def main(args):
    standalone = {}
    for item in args.standalone:
        key, path = item.split("=")
        variant, model = key.split(":")
        standalone.setdefault(variant, {})[model] = path

    for variant, models in standalone.items():
        for domain in args.domains:
            (p_math, n), (p_de, _) = load(models["F_math"], domain), load(models["F_german"], domain)
            base = max(p_math, p_de)
            se = math.sqrt(base * (1 - base) / n)
            for method in args.methods:
                for a in args.alphas:
                    merged = os.path.join(args.merge_root, variant, f"FF_{method}", f"alpha_{a}", "eval")
                    tpr, _ = load(merged, domain)
                    status = "above" if tpr > base + se else "below" if tpr < base - se else "within"
                    print(f"{variant:5s} {domain:5s} {method:6s} alpha={a}: FF={100 * tpr:5.1f}  "
                          f"baseline={100 * base:5.1f} (SE {100 * se:.1f})  {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--standalone", nargs="+", required=True,
                        help="variant:F_math=<eval dir> and variant:F_german=<eval dir> entries")
    parser.add_argument("--merge_root", default="output/merges")
    parser.add_argument("--methods", nargs="+", default=["linear", "slerp", "ties_d0.5"])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--domains", nargs="+", default=["en", "de", "math"])
    main(parser.parse_args())
