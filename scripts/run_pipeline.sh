#!/usr/bin/env bash
# Full LCR pipeline for one base-sim parameter set:
#   precompute (val + both training halves) -> train LCR -> evaluate -> N DAgger rounds (collect, fine-tune, evaluate)
# Usage: PY=<python> scripts/run_pipeline.sh <prefix> <params.json> [n_dagger_rounds=1]
set -eu
PY=${PY:-python}
cd "$(dirname "$0")/.."
PREFIX=$1; PARAMS=$2; ROUNDS=${3:-1}
export LCR_SIMPARAMS=$PARAMS
log() { echo "$1 $(date)"; }
log "START prefix=$PREFIX params=$PARAMS"
$PY scripts/precompute.py --which val --prefix ${PREFIX}_ 2>&1 | grep -v WARNING
$PY scripts/precompute.py --which train --prefix ${PREFIX}_ 2>&1 | grep -v WARNING
$PY scripts/precompute.py --which train --offset 1 --prefix ${PREFIX}_ 2>&1 | grep -v WARNING
log PRECOMPUTE_DONE
TR=${PREFIX}_train+${PREFIX}_train_off1; VA=${PREFIX}_val
OMP_NUM_THREADS=4 $PY scripts/train.py --model lcr --epochs 20 --train $TR --val $VA --tag $PREFIX > results/train_lcr_$PREFIX.log 2>&1
log TRAIN_DONE
OMP_NUM_THREADS=1 $PY scripts/evaluate.py --methods lcr --suffix _$PREFIX --out results/eval_$PREFIX.json 2>&1 | grep -v WARNING > results/eval_$PREFIX.log
log EVAL_DONE
prev=lcr_$PREFIX; data=""
for r in $(seq 1 $ROUNDS); do
  $PY scripts/onpolicy.py --model $prev --out cache/${PREFIX}_onpolicy_r$r.npz --starts 8 --horizon 12 --seed $r 2>&1 | grep -v WARNING
  data="${data:+$data+}${PREFIX}_onpolicy_r$r"
  OMP_NUM_THREADS=4 $PY scripts/train.py --model lcr --init $prev --epochs 8 --lr 3e-4 --train $TR+$data --val $VA --tag ${PREFIX}_dagger_r$r > results/train_lcr_${PREFIX}_dagger_r$r.log 2>&1
  prev=lcr_${PREFIX}_dagger_r${r}_last
  OMP_NUM_THREADS=1 $PY scripts/evaluate.py --methods lcr --suffix _${PREFIX}_dagger_r${r}_last --out results/eval_${PREFIX}_dagger_r$r.json 2>&1 | grep -v WARNING > results/eval_${PREFIX}_dagger_r$r.log
  log DAGGER_R${r}_DONE
done
log ALL_DONE
