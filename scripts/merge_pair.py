import argparse

from mat.merging import merge, pairwise_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--method", default="linear", choices=["linear", "slerp", "ties", "dare_ties"])
    parser.add_argument("--alpha", type=float, required=True, help="weight on B")
    parser.add_argument("--density", type=float, help="ties / dare_ties density")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    merge(pairwise_config(args.method, args.a, args.b, args.alpha, density=args.density), args.output_dir)
