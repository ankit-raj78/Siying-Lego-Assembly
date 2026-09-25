#!/usr/bin/env bash
# LCR on the refined base-sim parameters (results/simparams_sysid2.json), then two DAgger rounds.
set -eu
PY=${PY:-python}
cd "$(dirname "$0")/.."
export LCR_SIMPARAMS=results/simparams_sysid2.json
log() { echo "$1 $(date)"; }
log START
$PY scripts/precompute.py --which val --prefix s2_ 2>&1 | grep -v WARNING
$PY scripts/precompute.py --which train --prefix s2_ 2>&1 | grep -v WARNING
$PY scripts/precompute.py --which train --offset 1 --prefix s2_ 2>&1 | grep -v WARNING
log PRECOMPUTE_DONE
OMP_NUM_THREADS=4 $PY scripts/train.py --model lcr --epochs 20 --train s2_train+s2_train_off1 --val s2_val --tag s2 > results/train_lcr_s2.log 2>&1
log TRAIN_DONE
OMP_NUM_THREADS=1 $PY scripts/evaluate.py --methods lcr --suffix _s2 --out results/eval_s2.json 2>&1 | grep -v WARNING > results/eval_s2.log
log EVAL_S2_DONE
prev=lcr_s2; data=""
for r in 1 2; do
  $PY scripts/onpolicy.py --model $prev --out cache/s2_onpolicy_r$r.npz --starts 8 --horizon 12 --seed $r 2>&1 | grep -v WARNING
  data="${data:+$data+}s2_onpolicy_r$r"
  OMP_NUM_THREADS=4 $PY scripts/train.py --model lcr --init $prev --epochs 8 --lr 3e-4 --train s2_train+s2_train_off1+$data --val s2_val --tag s2_dagger_r$r > results/train_lcr_s2_dagger_r$r.log 2>&1
  prev=lcr_s2_dagger_r${r}_last
  OMP_NUM_THREADS=1 $PY scripts/evaluate.py --methods lcr --suffix _s2_dagger_r${r}_last --out results/eval_s2_dagger_r$r.json 2>&1 | grep -v WARNING > results/eval_s2_dagger_r$r.log
  log DAGGER_R${r}_DONE
done
log ALL_DONE
