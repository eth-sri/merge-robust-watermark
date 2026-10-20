#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

for v in kgwd mat; do
  for s in BBp BF FF; do
    for d in 0.3 0.5; do
      python scripts/run_merge_sweep.py --variant $v --scenario $s --method dare_ties --density $d
    done
  done
  for m in breadcrumbs breadcrumbs_ties; do
    python scripts/run_merge_sweep.py --variant $v --scenario BFp_OpenMath_anchored --method $m --alphas 0.5 0.7 0.9
  done
done
