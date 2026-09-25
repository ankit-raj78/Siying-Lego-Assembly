#!/usr/bin/env bash
# sysid3 (cylinder inflation + socket offset) -> sysid_free (per-axis M,D through the sim) -> base-sim test-set check.
set -u
PY=${PY:-python}
cd "$(dirname "$0")/.."
echo "START $(date)"
LCR_SIMPARAMS=results/simparams_sysid2.json OMP_NUM_THREADS=1 $PY scripts/sysid3.py 2>&1 | grep -v WARNING > results/sysid3.log
echo "sysid3 exit=$? $(date)"
LCR_SIMPARAMS=results/simparams_sysid3.json OMP_NUM_THREADS=1 $PY scripts/sysid_free.py 2>&1 | grep -v WARNING > results/sysid_free.log
echo "sysid_free exit=$? $(date)"
LCR_SIMPARAMS=results/simparams_sysid_free.json OMP_NUM_THREADS=1 $PY scripts/evaluate.py --methods base --out results/eval_base_sysid3.json 2>&1 | grep -v WARNING > results/eval_base_sysid3.log
echo "base eval done $(date)"
echo "CHAIN_DONE $(date)"
