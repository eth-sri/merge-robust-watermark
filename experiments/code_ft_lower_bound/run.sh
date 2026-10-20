#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
SFT_ACCELERATE=${SFT_ACCELERATE:?"set SFT_ACCELERATE to the accelerate executable of the requirements-sft.txt environment (transformers 5 + TRL), e.g. /path/to/sft-env/bin/accelerate"}
SFT=("$SFT_ACCELERATE" launch)

for v in kgwd mat; do
  "${SFT[@]}" scripts/finetune.py --config configs/sft/code.yaml \
    --base_model output/watermark/llama8b_${v}/checkpoint-5000 --output_dir output/sft/llama8b_${v}_code
  for m in linear slerp; do
    python scripts/run_merge_sweep.py --variant $v --scenario FBF_code --method $m
  done
done
for m in linear slerp; do
  python scripts/analyze_lower_bound.py --fbf FBF_code --method $m --output results/lower_bound_code_$m.json
done
