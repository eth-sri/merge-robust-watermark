#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python -c "import sys, transformers, mergekit; sys.exit(int(transformers.__version__.split('.')[0]) < 5)" || {
  echo "Ministral-3 needs the requirements-sft.txt environment (transformers >= 5) with mergekit installed via: uv pip install --no-deps mergekit==0.1.4" >&2
  exit 1
}

python scripts/prepare_ministral.py --output output/ministral3b_text
accelerate launch scripts/train_watermark.py --config configs/watermark/ministral3b_kgwd.yaml
accelerate launch scripts/train_watermark.py --config configs/watermark/ministral3b_mat.yaml
for v in ministral3b_kgwd ministral3b_mat; do
  python scripts/run_merge_sweep.py --variant $v --scenario BBp --method slerp --alphas 0.5 --domains en de
done
