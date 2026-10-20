import argparse
import json
import os


def tpr(root, variant, scenario, method, alpha, domain):
    path = os.path.join(root, variant, f"{scenario}_{method}", f"alpha_{alpha}", "eval", f"detection_{domain}.json")
    with open(path) as f:
        return json.load(f)["metrics"]["tpr_at_1pct_fpr"]


def main(args):
    rows, holds = [], 0
    for variant in args.variants:
        for domain in args.domains:
            for a in args.alphas:
                a_eff = round(0.5 * a, 4)
                fbf = tpr(args.merge_root, variant, args.fbf, args.method, a, domain)
                bb = tpr(args.merge_root, variant, args.bb, args.method, a_eff, domain)
                ok = bb <= fbf
                holds += ok
                rows.append(dict(variant=variant, domain=domain, alpha_eff=a_eff, tpr_fbf=fbf, tpr_bb=bb, bound_holds=ok))
    for r in rows:
        print(f"{r['variant']:6s} {r['domain']:5s} a_eff={r['alpha_eff']:.2f}  FBF={r['tpr_fbf']:.3f}  "
              f"BB'={r['tpr_bb']:.3f}  {'holds' if r['bound_holds'] else 'FAILS'}")
    print(f"\nbound holds in {holds}/{len(rows)} configurations ({100 * holds / len(rows):.1f}%)")
    with open(args.output, "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--merge_root", default="output/merges")
    parser.add_argument("--method", default="slerp")
    parser.add_argument("--variants", nargs="+", default=["kgwd", "mat"])
    parser.add_argument("--domains", nargs="+", default=["en", "de", "math"])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--fbf", default="FBF", help="FBF_code for the code-finetune test")
    parser.add_argument("--bb", default="BBp")
    parser.add_argument("--output", default="lower_bound.json")
    main(parser.parse_args())
