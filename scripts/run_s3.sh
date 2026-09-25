#!/usr/bin/env bash
# Base-sim check with the final params, then the full pipeline (prefix s3, 1 DAgger round).
set -u
PY=${PY:-python}
cd "$(dirname "$0")/.."
echo "START $(date)"
LCR_SIMPARAMS=results/simparams_final.json OMP_NUM_THREADS=1 $PY scripts/evaluate.py --methods base --out results/eval_base_final.json 2>&1 | grep -v WARNING > results/eval_base_final.log
echo "BASE_FINAL_DONE $(date)"
PY=$PY scripts/run_pipeline.sh s3 results/simparams_final.json 1
echo "S3_DONE $(date)"
