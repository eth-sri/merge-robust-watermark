#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."

EXP=experiments/learned_merging
OUT=${OUT:-output/learned_merging}

for variant in kgwd mat; do
    wm_config=configs/watermark/llama8b_${variant}.yaml
    for w in 0.7 0.5; do
        python "$EXP/learned_merge.py" --variant $variant --wm_weight $w --output_dir "$OUT/$variant/learned_wm$w"
        python "$EXP/query_limited_removal.py" --start "$OUT/$variant/learned_wm$w" --watermark_config $wm_config \
            --output_dir "$OUT/$variant/query_wm$w"
    done
    for run in learned_wm0.7 query_wm0.7 query_wm0.5; do
        for domain in en de math; do
            if [ ! -f "$OUT/$variant/$run/eval/detection_$domain.json" ]; then
                python scripts/eval_detection.py --model "$OUT/$variant/$run/model" --watermark_config $wm_config \
                    --domain $domain --n_samples 500 --output_dir "$OUT/$variant/$run/eval"
            fi
        done
    done
done

python - "$OUT" <<'EOF'
import json, sys

def tpr(variant, run, domain):
    with open(f"{sys.argv[1]}/{variant}/{run}/eval/detection_{domain}.json") as f:
        return 100 * json.load(f)["metrics"]["tpr_at_1pct_fpr"]

print(f"{'Setting':28s}{'English':>15s}{'German':>15s}{'Math':>15s}")
for label, run in [("Learned merge 70/30", "learned_wm0.7"), ("Query-limited removal 70/30", "query_wm0.7"),
                   ("Query-limited removal 50/50", "query_wm0.5")]:
    cells = [f"{tpr('kgwd', run, d):.1f} / {tpr('mat', run, d):.1f}" for d in ("en", "de", "math")]
    print(f"{label:28s}" + "".join(f"{c:>15s}" for c in cells))
EOF
