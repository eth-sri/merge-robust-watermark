#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."

python experiments/mechanism/interpolation_score.py \
    --variants kgwd mat \
    --lambdas 0.1 0.25 0.4 0.6 0.8 1.0 \
    --n_contexts 48 --context_length 128 --min_tokens 256 \
    --dtype float32 --bootstrap 5000 --seed 42 \
    --output_dir output/mechanism
