#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
accelerate launch scripts/train_watermark.py --config configs/watermark/llama8b_mat_fusechat_ref.yaml

python - <<'PY'
import yaml
cfg = yaml.safe_load(open("configs/merge/scenarios.yaml"))
cfg["variants"]["mat_fusechat_ref"] = {"watermark_config": "configs/watermark/llama8b_mat_fusechat_ref.yaml",
                                       "B": "output/watermark/llama8b_mat_fusechat_ref/checkpoint-5000"}
yaml.safe_dump(cfg, open("output/scenarios_reference_ablation.yaml", "w"))
PY
for v in mat mat_fusechat_ref; do
  for s in BFp_FuseChat BFp_Tulu; do
    python scripts/run_merge_sweep.py --config output/scenarios_reference_ablation.yaml --variant $v \
      --scenario $s --method linear --alphas 0.3 0.5 0.7 --domains en --output_root output/reference_ablation
  done
done
