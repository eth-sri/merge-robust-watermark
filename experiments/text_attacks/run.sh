#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-output/text_attacks}

for variant in kgwd mat; do
    python experiments/text_attacks/generate.py --variant "$variant" --alpha 0.5 --n_samples 500 \
        --prefix_length 50 --max_new_tokens 200 --output_dir "$OUT/$variant"
    python experiments/text_attacks/attack.py --input_dir "$OUT/$variant" --attacks all
done
python experiments/text_attacks/score.py --dirs "$OUT/kgwd" "$OUT/mat" --max_score_tokens 200
