#!/usr/bin/env bash
# After the tool-friction fix: redo sysid4 (geometry, orientation scored) and test-set base-sim checks.
set -u
PY=${PY:-python}
cd "$(dirname "$0")/.."
echo "START $(date)"
LCR_SIMPARAMS=results/simparams_sysid2.json OMP_NUM_THREADS=1 $PY scripts/sysid4.py 2>&1 | grep -v WARNING > results/sysid4.log
echo "sysid4 exit=$? $(date)"
for P in sysid2 sysid4 final; do
  LCR_SIMPARAMS=results/simparams_$P.json OMP_NUM_THREADS=1 $PY scripts/evaluate.py --methods base --out results/eval_base_${P}_v2.json 2>&1 | grep -v WARNING > results/eval_base_${P}_v2.log
  echo "base $P eval done $(date)"
done
echo "RECHECK_DONE $(date)"
