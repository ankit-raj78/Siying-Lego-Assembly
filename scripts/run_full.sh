#!/usr/bin/env bash
# Full-data (train + train_off1, ~400k samples) training and evaluation.
# Waits on PIDs with `wait` (a pgrep on the command text would match this script itself).
set -u
PY=${PY:-python}
cd "$(dirname "$0")/.."
echo "START $(date)"
OMP_NUM_THREADS=2 $PY scripts/train.py --model lcr --epochs 20 --train train+train_off1 --tag full > results/train_lcr_full.log 2>&1 &
p1=$!
OMP_NUM_THREADS=2 $PY scripts/train.py --model lcr_nogeom --epochs 20 --train train+train_off1 --tag full > results/train_lcr_nogeom_full.log 2>&1 &
p2=$!
wait $p1; echo "lcr_full exit=$? $(date)"
wait $p2; echo "lcr_nogeom_full exit=$? $(date)"
OMP_NUM_THREADS=4 $PY scripts/train.py --model global --epochs 20 --train train+train_off1 --tag full > results/train_global_full.log 2>&1
echo "global_full exit=$? $(date)"
OMP_NUM_THREADS=1 $PY scripts/evaluate.py --methods lcr,lcr_nogeom,global --suffix _full --out results/eval_full.json 2>&1 | grep -v WARNING > results/eval_full.log
echo "eval_full done $(date)"
OMP_NUM_THREADS=1 $PY scripts/evaluate.py --methods base,blackbox --out results/eval_ref.json 2>&1 | grep -v WARNING > results/eval_ref.log
echo "ALL_DONE $(date)"
