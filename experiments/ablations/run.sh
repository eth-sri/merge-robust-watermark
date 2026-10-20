#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
C=configs/watermark/llama1b_ablation.yaml

run() {
  local name=$1; shift
  accelerate launch scripts/train_watermark.py --config $C --override output_dir=output/ablations/$name "$@"
  for ckpt in output/ablations/$name/checkpoint-*; do
    python scripts/eval_detection.py --model $ckpt --watermark_config $C --domain en --output_dir $ckpt/eval
    python scripts/eval_quality.py --model $ckpt --watermark_config $C --domain en --output_dir $ckpt/eval
    python - "$ckpt" <<'PY'
import os, sys, yaml
cfg = yaml.safe_load(open("configs/merge/scenarios.yaml"))
cfg["variants"]["llama1b_ablation"]["B"] = sys.argv[1]
os.makedirs(f"{sys.argv[1]}/merges", exist_ok=True)
yaml.safe_dump(cfg, open(f"{sys.argv[1]}/merges/scenarios.yaml", "w"))
PY
    python scripts/run_merge_sweep.py --config $ckpt/merges/scenarios.yaml --variant llama1b_ablation \
      --scenario BBp --method linear --alphas 0.1 0.3 0.5 0.7 0.9 --domains en --output_root $ckpt/merges
  done
}

run base
run no_topk watermark.top_k=null
for k in 50 100 300 500; do run k$k watermark.top_k=$k; done
for d in 1.0 2.3 2.5 2.7; do run delta$d watermark.delta=$d; done
for a in 0.3 0.5; do run amin$a merge_adversary.alpha_min=$a; done
for d in 2.0 2.2 2.3 2.5; do run k100_delta$d watermark.top_k=100 watermark.delta=$d; done
run fixed_alpha0.3 merge_adversary.alpha_min=0.3 merge_adversary.alpha_max=0.3
