#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

accelerate launch scripts/train_watermark.py --config configs/watermark/qwen3b_kgwd.yaml
accelerate launch scripts/train_watermark.py --config configs/watermark/qwen3b_mat.yaml
for v in qwen3b_kgwd qwen3b_mat; do
  python scripts/run_merge_sweep.py --variant $v --scenario BBp --method slerp --domains en
done
python scripts/eval_detection.py --logit_processor --watermark_config configs/watermark/qwen3b_kgwd.yaml --domain en --output_dir results/standalone/qwen3b_kgw_logit_processor
python scripts/eval_quality.py --logit_processor --watermark_config configs/watermark/qwen3b_kgwd.yaml --domain en --output_dir results/quality/qwen3b_kgw_logit_processor
