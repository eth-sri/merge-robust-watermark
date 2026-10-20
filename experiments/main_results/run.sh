#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
SFT_ACCELERATE=${SFT_ACCELERATE:?"set SFT_ACCELERATE to the accelerate executable of the requirements-sft.txt environment (transformers 5 + TRL), e.g. /path/to/sft-env/bin/accelerate"}
SFT=("$SFT_ACCELERATE" launch)

accelerate launch scripts/train_watermark.py --config configs/watermark/llama8b_kgwd.yaml
accelerate launch scripts/train_watermark.py --config configs/watermark/llama8b_mat.yaml

for v in kgwd mat; do
  B=output/watermark/llama8b_${v}/checkpoint-5000
  "${SFT[@]}" scripts/finetune.py --config configs/sft/math.yaml --base_model $B --output_dir output/sft/llama8b_${v}_math
  "${SFT[@]}" scripts/finetune.py --config configs/sft/german.yaml --base_model $B --output_dir output/sft/llama8b_${v}_german
  "${SFT[@]}" scripts/finetune.py --config configs/sft/french.yaml --base_model $B --output_dir output/sft/llama8b_${v}_french
done

for v in kgwd mat; do
  C=configs/watermark/llama8b_${v}.yaml
  for m in watermark/llama8b_${v}/checkpoint-5000 sft/llama8b_${v}_math sft/llama8b_${v}_german sft/llama8b_${v}_french; do
    for d in en de math; do python scripts/eval_detection.py --model output/$m --watermark_config $C --domain $d --output_dir results/standalone/$m; done
    for d in en de fr; do python scripts/eval_quality.py --model output/$m --watermark_config $C --domain $d --output_dir results/quality/$m; done
  done
done
for d in en de math; do python scripts/eval_detection.py --logit_processor --watermark_config configs/watermark/llama8b_kgwd.yaml --domain $d --output_dir results/standalone/kgw_logit_processor; done
for d in en de fr; do
  python scripts/eval_quality.py --logit_processor --watermark_config configs/watermark/llama8b_kgwd.yaml --domain $d --output_dir results/quality/kgw_logit_processor
  python scripts/eval_quality.py --model meta-llama/Llama-3.1-8B-Instruct --watermark_config configs/watermark/llama8b_kgwd.yaml --domain $d --output_dir results/quality/instruct
done

python scripts/eval_benchmarks.py --model meta-llama/Llama-3.1-8B-Instruct --output_dir results/benchmarks/instruct
for v in kgwd mat; do
  for m in watermark/llama8b_${v}/checkpoint-5000 sft/llama8b_${v}_math sft/llama8b_${v}_german sft/llama8b_${v}_french; do
    python scripts/eval_benchmarks.py --model output/$m --output_dir results/benchmarks/$m
  done
done

for v in kgwd mat; do
  for s in FF BF F2BF FBF F2F BBp BFp_FuseChat BFp_OpenMath BFp_Tulu; do
    for m in linear slerp ties; do
      python scripts/run_merge_sweep.py --variant $v --scenario $s --method $m --output_root output/merges
    done
  done
done

for v in kgwd mat; do
  for s in FF BF; do
    python scripts/run_merge_sweep.py --variant $v --scenario $s --method slerp --alphas 0.5 0.7 --keep_models
    for a in 0.5 0.7; do
      M=output/merges/$v/${s}_slerp/alpha_$a/model
      python scripts/eval_benchmarks.py --model $M --output_dir results/benchmarks/merges/$v/${s}_slerp_$a
      for d in en de fr; do python scripts/eval_quality.py --model $M --watermark_config configs/watermark/llama8b_${v}.yaml --domain $d --output_dir results/quality/merges/$v/${s}_slerp_$a; done
    done
  done
done

for v in kgwd mat; do
  for m in linear slerp ties; do
    python scripts/run_merge_sweep.py --variant $v --scenario BBp --method $m --alphas 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45
  done
done
for m in linear slerp ties_d0.5; do python scripts/analyze_lower_bound.py --method $m --output results/lower_bound_$m.json; done

python scripts/analyze_single_ft_baseline.py \
  --standalone kgwd:F_math=results/standalone/sft/llama8b_kgwd_math kgwd:F_german=results/standalone/sft/llama8b_kgwd_german \
               mat:F_math=results/standalone/sft/llama8b_mat_math mat:F_german=results/standalone/sft/llama8b_mat_german
