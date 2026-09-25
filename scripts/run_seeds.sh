#!/usr/bin/env bash
# Extra training seeds for LCR vs. LCR without geometry features (full data), then evaluation.
set -u
PY=${PY:-python}
cd "$(dirname "$0")/.."
echo "START $(date)"
for seed in 1 2; do
  OMP_NUM_THREADS=2 $PY scripts/train.py --model lcr --epochs 20 --train train+train_off1 --tag full_s$seed --seed $seed > results/train_lcr_full_s$seed.log 2>&1 &
  p1=$!
  OMP_NUM_THREADS=2 $PY scripts/train.py --model lcr_nogeom --epochs 20 --train train+train_off1 --tag full_s$seed --seed $seed > results/train_lcr_nogeom_full_s$seed.log 2>&1 &
  p2=$!
  wait $p1; echo "lcr s$seed exit=$? $(date)"
  wait $p2; echo "lcr_nogeom s$seed exit=$? $(date)"
done
for seed in 1 2; do
  OMP_NUM_THREADS=1 $PY scripts/evaluate.py --methods lcr,lcr_nogeom --suffix _full_s$seed --out results/eval_full_s$seed.json 2>&1 | grep -v WARNING > results/eval_full_s$seed.log
  echo "eval s$seed done $(date)"
done
echo "ALL_DONE $(date)"
