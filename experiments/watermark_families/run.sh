#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

accelerate launch scripts/train_watermark.py --config configs/watermark/llama8b_aar_d.yaml
accelerate launch scripts/train_watermark.py --config configs/watermark/llama8b_aar_mat.yaml
accelerate launch scripts/generate_kth_samples.py --config configs/watermark/llama8b_kth_d.yaml
accelerate launch scripts/train_on_samples.py --config configs/watermark/llama8b_kth_d.yaml
accelerate launch scripts/train_on_samples.py --config configs/watermark/llama8b_kth_mat.yaml
python scripts/embed_gaussmark.py --config configs/watermark/llama8b_gaussmark.yaml

for v in aar_d aar_mat kth_d kth_mat; do
  python scripts/eval_detection.py --model output/watermark/llama8b_$v/checkpoint-5000 --watermark_config configs/watermark/llama8b_$v.yaml --domain en --output_dir results/standalone/llama8b_$v
  python scripts/eval_quality.py --model output/watermark/llama8b_$v/checkpoint-5000 --watermark_config configs/watermark/llama8b_$v.yaml --domain en --output_dir results/quality/llama8b_$v
done
python scripts/eval_gaussmark.py --model output/watermark/llama8b_gaussmark --secret output/watermark/llama8b_gaussmark/gaussmark_secret.pt --output_dir results/standalone/llama8b_gaussmark
python scripts/eval_quality.py --model output/watermark/llama8b_gaussmark --watermark_config configs/watermark/llama8b_kgwd.yaml --domain en --output_dir results/quality/llama8b_gaussmark

for v in aar_d aar_mat kth_d kth_mat gaussmark; do
  for m in linear slerp ties; do
    python scripts/run_merge_sweep.py --variant $v --scenario BBp --method $m --domains en
  done
done
