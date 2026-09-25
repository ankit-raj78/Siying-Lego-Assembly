#!/usr/bin/env bash
# Train SkinNet on the s3 caches (features computed on the fly), then evaluate. Usage: run_skin.sh <tag> [epochs]
set -u
PY=${PY:-python}
cd "$(dirname "$0")/.."
TAG=$1; EPOCHS=${2:-20}; MODEL=${MODEL:-skin}
export LCR_SIMPARAMS=results/simparams_sysid4.json
echo "START $TAG $(date)"
OMP_NUM_THREADS=4 $PY scripts/train.py --model $MODEL --epochs $EPOCHS --train s3_train+s3_train_off1 --val s3_val --tag $TAG > results/train_skin_$TAG.log 2>&1
echo "train exit=$? $(date)"
OMP_NUM_THREADS=1 $PY scripts/evaluate.py --methods $MODEL --suffix _$TAG --out results/eval_skin_$TAG.json 2>&1 | grep -v WARNING > results/eval_skin_$TAG.log
echo "eval done $(date)"
echo "ALL_DONE $TAG $(date)"
