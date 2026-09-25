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

## Prelim results (2026-09-25)
100-step open-loop rollouts, per-axis RMSE (position mm / orientation deg). LCR = mean of 3 training seeds.

| method | seen tools | square (unseen tool) | expert policy (OOD) | circle policy (OOD) |
|---|---|---|---|---|
| MuJoCo-Adm (identified base sim) | 5.34 / 1.17 | 5.71 / 1.13 | 10.37 / 1.67 | 9.50 / 2.19 |
| + global MLP residual | 12.05 / 3.78 | 13.40 / 2.80 | 15.08 / 2.95 | 9.52 / 2.20 |
| black-box MLP | 8.35 / 1.92 | 8.63 / 2.18 | 16.79 / 1.76 | 17.13 / 1.88 |
| **LCR** | **4.73 / 0.96** | **5.07 / 0.91** | 9.08 / 1.22 | 4.36 / 0.51 |
| Act-FIGNet (paper, seen tools) | 2.87 / 0.64 | – | – | – |

Full tables (incl. one-step F/T, timing, per-seed ablation): `results/summary.md`.
