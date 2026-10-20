#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
SFT_ACCELERATE=${SFT_ACCELERATE:?"set SFT_ACCELERATE to the accelerate executable of the requirements-sft.txt environment (transformers 5 + TRL), e.g. /path/to/sft-env/bin/accelerate"}
SFT=("$SFT_ACCELERATE" launch)
B_PRIME=meta-llama/Llama-3.1-8B-Instruct
ROOT=output/post_merge_sft

for v in kgwd mat; do
  C=configs/watermark/llama8b_${v}.yaml
  for s in BBp BF FF; do
    python scripts/run_merge_sweep.py --variant $v --scenario $s --method linear --alphas 0.5 --keep_models --output_root $ROOT/merges
    M=$ROOT/merges/$v/${s}_linear/alpha_0.5/model
    SFT_DIR=$ROOT/sft/$v/$s
    "${SFT[@]}" scripts/finetune.py --config configs/sft/french.yaml --base_model $M --output_dir $SFT_DIR
    for step in 500 1000 1500 1814; do
      CK=$SFT_DIR/checkpoint-$step
      python scripts/merge_pair.py --a $B_PRIME --b $CK --method linear --alpha 0.5 --output_dir $CK-remerged
      for d in en de math; do
        python scripts/eval_detection.py --model $CK --watermark_config $C --domain $d --output_dir $CK/eval
        python scripts/eval_detection.py --model $CK-remerged --watermark_config $C --domain $d --output_dir $CK-remerged/eval
      done
    done
  done
done
