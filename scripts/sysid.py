"""Stage 0: identify contact parameters of MuJoCo-Adm on the sysid splits (as in the paper).
Free-space mass/damping come from analysis/free_space_fit.py and are fixed here.

Objective per config: 10-step rollout position RMSE (mm) + one-step F/T force error (N),
each divided by the value of the default config."""
import itertools
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lcr import data as D                     # noqa: E402
from lcr.rollout import rollout, pose_errors, segment_starts  # noqa: E402
from lcr.sim import AdmSim, SimParams         # noqa: E402

H = 10


def evaluate(cfg):
    mu, tc = cfg
    rng = np.random.default_rng(0)
    pe, Winst, Wavg, O0, O1 = [], [], [], [], []
    for split in D.SYSID_SPLITS:
        sim = AdmSim(split, SimParams(friction=mu, solref_tc=tc))
        for ep in D.load_episodes(split):
            for t0 in segment_starts(len(ep["pos"]), H, 8, rng):
                P, Q, W, bad = rollout(sim, ep, t0, H)
                e, _ = pose_errors(P, Q, ep, t0)
                pe.append(e)
            for t in range(2, len(ep["pos"]) - 2, 5):          # one-step wrench checks
                v, w = D.backward_velocity(ep["pos"], ep["quat"], t)
                sim.set_state(ep["pos"][t], ep["quat"][t], v, w)
                Winst.append(sim.forward(ep["action"][t]))
                Wavg.append(sim.step(ep["action"][t]))
                O0.append(ep["obs"][t]); O1.append(ep["obs"][t + 1])
    pe = np.array(pe)
    Winst, Wavg, O0, O1 = map(np.array, (Winst, Wavg, O0, O1))
    rm = lambda A, B: float(np.sqrt(((A[:, :3] - B[:, :3]) ** 2).mean(0)).mean())
    return {"mu": mu, "tc": tc, "pos_rmse_mm": float(np.sqrt((pe ** 2).mean()) * 1000),
            "F_inst_vs_o_t": rm(Winst, O0), "F_avg_vs_o_t": rm(Wavg, O0), "F_avg_vs_o_t1": rm(Wavg, O1)}


if __name__ == "__main__":
    grid = list(itertools.product([0.1, 0.3, 0.6, 1.0], [0.05, 0.1, 0.2, 0.4]))
    with Pool(4) as pool:
        res = pool.map(evaluate, grid)
    base = next(r for r in res if r["mu"] == 1.0 and r["tc"] == 0.05)
    for r in res:
        r["score"] = r["pos_rmse_mm"] / base["pos_rmse_mm"] + r["F_avg_vs_o_t"] / base["F_avg_vs_o_t"]
    res.sort(key=lambda r: r["score"])
    for r in res:
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    os.makedirs("results", exist_ok=True)
    json.dump({"best": res[0], "all": res}, open("results/sysid.json", "w"), indent=1)
    print("BEST", res[0])
