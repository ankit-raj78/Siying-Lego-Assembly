#!/usr/bin/env bash
# Fine-tune LCR on original + on-policy data, then evaluate. Usage: run_dagger.sh <init> <round-data> <tag>
set -u
PY=${PY:-python}
cd "$(dirname "$0")/.."
INIT=$1; DATA=$2; TAG=$3
echo "START $TAG $(date)"
OMP_NUM_THREADS=2 $PY scripts/train.py --model lcr --init $INIT --epochs 8 --lr 3e-4 --train train+train_off1+$DATA --tag $TAG > results/train_lcr_$TAG.log 2>&1
echo "train exit=$? $(date)"
OMP_NUM_THREADS=1 $PY scripts/evaluate.py --methods lcr --suffix _${TAG}_last --out results/eval_$TAG.json 2>&1 | grep -v WARNING > results/eval_$TAG.log
echo "eval done $(date)"
echo "ALL_DONE $TAG $(date)"
