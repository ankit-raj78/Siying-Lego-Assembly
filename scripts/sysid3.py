"""Stage 0c: identify cylinder mesh inflation and the training-session socket pose offset on a subset of
TRAIN episodes (the test sets share this socket pose; sysid splits use a different one).
Base params: LCR_SIMPARAMS (sysid2). Objective: 30-step rollout per-axis pos RMSE + one-step force error,
each normalised by the no-offset config."""
import json, os, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lcr import data as D
from lcr.rollout import rollout, segment_starts
from lcr.sim import AdmSim, params_from_cfg
H, N_EPS = 30, 30
BASE = json.load(open(os.environ["LCR_SIMPARAMS"]))

def evaluate(c):
    cfg = dict(BASE, **c)
    rng = np.random.default_rng(0)
    pv, Wp, O = [], [], []
    for split in D.TRAIN_SPLITS:
        sim = AdmSim(split, params_from_cfg(cfg))
        for ep in D.load_episodes(split)[:N_EPS]:
            for t0 in segment_starts(len(ep["pos"]), H, 4, rng):
                P, Q, W, bad = rollout(sim, ep, t0, H)
                pv.append(P - ep["pos"][t0 + 1:t0 + H + 1])
            for t in range(2, len(ep["pos"]) - 2, 10):
                v, w = D.backward_velocity(ep["pos"], ep["quat"], t)
                sim.set_state(ep["pos"][t], ep["quat"][t], v, w)
                Wp.append(sim.step(ep["action"][t])); O.append(ep["obs"][t])
    pv, Wp, O = np.array(pv), np.array(Wp), np.array(O)
    return dict(c, pos_axis_mm=float(np.sqrt((pv ** 2).mean()) * 1000),
                force=float(np.sqrt(((Wp[:, :3] - O[:, :3]) ** 2).mean(0)).mean()))

if __name__ == "__main__":
    rng = np.random.default_rng(3)
    cands = [{"inflate_cylinder": 0.0, "dx": 0.0, "dy": 0.0, "dz": 0.0, "dyaw": 0.0}]
    for _ in range(47):
        cands.append({"inflate_cylinder": float(rng.uniform(0, 4e-4)), "dx": float(rng.uniform(-5e-4, 5e-4)),
                      "dy": float(rng.uniform(-5e-4, 5e-4)), "dz": float(rng.uniform(-5e-4, 5e-4)),
                      "dyaw": float(rng.uniform(-0.02, 0.02))})
    with Pool(4) as pool:
        res = pool.map(evaluate, cands)
    b = res[0]
    for r in res:
        r["score"] = r["pos_axis_mm"] / b["pos_axis_mm"] + r["force"] / b["force"]
    res.sort(key=lambda r: r["score"])
    for r in res[:8]:
        print({k: (round(v, 6) if abs(v) < 1 else round(v, 4)) for k, v in r.items()})
    print("DEFAULT", {k: round(v, 4) for k, v in b.items()})
    json.dump({"best": res[0], "default": b, "all": res}, open("results/sysid3.json", "w"), indent=1)
    json.dump(dict(BASE, **{k: res[0][k] for k in ("inflate_cylinder", "dx", "dy", "dz", "dyaw")}),
              open("results/simparams_sysid3.json", "w"), indent=1)
    print("SYSID3_DONE")
