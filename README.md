# Siying-Lego-Assembly

Prelim experiment for CAD-conditioned contact dynamics: **LCR (Local Contact Residual)** —
a rigid MuJoCo simulator handles geometry/non-penetration, and a small network adds
per-contact residual forces conditioned on local geometry. Evaluated on the Act-FIGNet
real-world peg-insertion dataset (Zenodo 10.5281/zenodo.22802802).

## Pipeline
```bash
pip install -r requirements.txt
scripts/download_data.sh                 # ~180 MB into data/
python scripts/sysid.py                  # stage 0: contact params of the base sim
python scripts/precompute.py             # stage 1: features + linearised sim (cache/)
python scripts/train.py --model lcr      # stage 2 (also: global, lcr_nogeom, blackbox)
python scripts/evaluate.py               # 100-step rollouts, one-step F/T, timing
```
`analysis/` holds the data checks that motivated the design.
