"""Merge evaluation files into results/summary.json and print markdown tables.
Rollout errors are reported per axis (matches the paper's MuJoCo numbers) and as vector norms."""
import json
import os

import numpy as np

SETS = ["test_seen", "square_unseen", "expert_ood", "circle_ood"]
PAPER = {  # Table III of arXiv:2509.12151 (real-world random-policy test set)
    "MuJoCo (paper)": (5.2059, 0.9231, 3.5420, 0.0920),
    "MuJoCo+res (paper)": (6.6267, 1.4698, 1.1178, 0.0454),
    "Act-FIGNet (paper)": (2.8659, 0.6431, 0.9182, 0.0339),
}


def load(p):
    return json.load(open(p)) if os.path.exists(p) else {}


ref, full = load("results/eval_ref.json"), load("results/eval_full.json")
seeds = {"lcr": [full.get("lcr_full")], "lcr_nogeom": [full.get("lcr_nogeom_full")]}
for s in (1, 2):
    e = load(f"results/eval_full_s{s}.json")
    for m in seeds:
        seeds[m].append(e.get(f"{m}_full_s{s}"))
seeds = {m: [x for x in v if x] for m, v in seeds.items()}

KEYS = {"pos_axis": lambda r: r["rollout_pos_rmse_axis_mm"], "ori_axis": lambda r: r["rollout_ori_rmse_axis_deg"],
        "pos_vec": lambda r: r["rollout_pos_rmse_mm"], "ori_vec": lambda r: r["rollout_ori_rmse_deg"],
        "roll_force": lambda r: r["rollout_ft"]["force_axis_mean"],
        "one_force": lambda r: r["one_step"]["force_axis_mean"], "one_torque": lambda r: r["one_step"]["torque_axis_mean"],
        "ms": lambda r: r["ms_per_step"]}


def stats(runs, ts):
    out = {}
    for k, f in KEYS.items():
        v = np.array([f(r[ts]) for r in runs])
        out[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v))}
    out["pos_curve"] = list(np.mean([r[ts]["pos_err_curve_mm"] for r in runs], 0))
    return out


summary = {"paper": PAPER, "sets": {}}
methods = {"base": [ref["base"]], "global": [full["global_full"]], "blackbox": [ref["blackbox"]],
           "lcr_nogeom": seeds["lcr_nogeom"], "lcr": seeds["lcr"]}
for ts in SETS:
    summary["sets"][ts] = {m: stats(r, ts) for m, r in methods.items()}
    summary["sets"][ts]["linear"] = {"one_force": ref["linear_admittance"][ts]["one_step"]["force_axis_mean"],
                                     "one_torque": ref["linear_admittance"][ts]["one_step"]["torque_axis_mean"]}
json.dump(summary, open("results/summary.json", "w"), indent=1)

fmt = lambda s: f"{s['mean']:.2f}" + (f" ± {s['std']:.2f}" if s["n"] > 1 else "")
for ts in SETS:
    print(f"\n### {ts}\n| method | seeds | pos axis mm | ori axis deg | rollout F N | 1-step F N | 1-step T Nm | ms/step |")
    print("|---|---|---|---|---|---|---|---|")
    for m, s in summary["sets"][ts].items():
        if m == "linear":
            print(f"| linear admittance | – | – | – | – | {s['one_force']:.2f} | {s['one_torque']:.4f} | – |")
            continue
        print(f"| {m} | {s['pos_axis']['n']} | {fmt(s['pos_axis'])} | {fmt(s['ori_axis'])} | {fmt(s['roll_force'])} | "
              f"{fmt(s['one_force'])} | {s['one_torque']['mean']:.4f} | {s['ms']['mean']:.2f} |")
