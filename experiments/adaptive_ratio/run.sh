#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."

for variant in kgwd mat; do
    python experiments/adaptive_ratio/adaptive_ratio.py \
        --variant "$variant" \
        --scenario BFp_OpenMath \
        --alphas 0.7 0.75 0.8 0.85 0.9 \
        --probe_size 50 \
        --n_repeats 20 \
        --probe_pool_size 200 \
        --holdout_size 300
done
