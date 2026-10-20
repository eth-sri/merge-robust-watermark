#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."

OUT=${OUT:-output/behavioral_audit}
CONTROL=${CONTROL:-output/watermark/llama8b_kgwd_topk100/checkpoint-5000}
MAT=${MAT:-output/watermark/llama8b_mat/checkpoint-5000}
DIR=experiments/behavioral_audit

[ -d "$CONTROL" ] || accelerate launch scripts/train_watermark.py --config configs/watermark/llama8b_kgwd_topk100.yaml
python "$DIR/generate.py" --model "$CONTROL" --label control --output_dir "$OUT"
python "$DIR/generate.py" --model "$MAT" --label mat --output_dir "$OUT"

python "$DIR/classify_refusals.py" --output_dir "$OUT" --labels control mat

python "$DIR/judge_style.py" --judge_model Qwen/Qwen2.5-32B-Instruct --name qwen --output_dir "$OUT"
python "$DIR/judge_style.py" --judge_model meta-llama/Llama-3.1-70B-Instruct --name llama --output_dir "$OUT"

python "$DIR/summarize.py" --output_dir "$OUT" --judges qwen llama
