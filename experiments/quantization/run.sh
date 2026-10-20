#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=output/quantization

for v in kgwd mat; do
  C=configs/watermark/llama8b_${v}.yaml
  for s in BBp BF FF; do
    python scripts/run_merge_sweep.py --variant $v --scenario $s --method linear --alphas 0.3 0.5 0.7 \
      --domains en --keep_models --output_root $ROOT
    for a in 0.3 0.5 0.7; do
      M=$ROOT/$v/${s}_linear/alpha_$a/model
      for q in int8 fp4 nf4; do
        python scripts/eval_detection.py --model $M --watermark_config $C --domain en --quantization $q \
          --output_dir $ROOT/$v/${s}_linear/alpha_$a/eval_$q
      done
      rm -rf $M
    done
  done
done
