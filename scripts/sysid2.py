"""Stage 0, refined: random search over 6 base-sim parameters on the sysid splits.
Objective: 30-step rollout position RMSE (mm, per axis) + one-step F/T force error (N),
each normalised by the current default parameters."""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lcr import data as D                                    # noqa: E402
from lcr.rollout import rollout, segment_starts              # noqa: E402
from lcr.sim import AdmSim, SimParams                        # noqa: E402

H = 30
BASE = SimParams()


def make(c):
    sd, sm = c["damp_scale"], c["mass_scale"]
    dmp = tuple(np.array(BASE.damping) * np.array([sd, sd, sd, 1, 1, 1]))
    arm = tuple(np.array(BASE.armature) * np.array([sm, sm, sm, 1, 1, 1]))
    return SimParams(damping=dmp, armature=arm, friction=c["friction"], solref_tc=c["tc"], solref_dr=c["dr"],
                     solimp=(c["d0"], 0.99, 0.001, 0.5, 2.0))


def evaluate(c):
    rng = np.random.default_rng(0)
    pv, Wp, O = [], [], []
    for split in D.SYSID_SPLITS:
        sim = AdmSim(split, make(c))
        for ep in D.load_episodes(split):
            for t0 in segment_starts(len(ep["pos"]), H, 4, rng):
                P, Q, W, bad = rollout(sim, ep, t0, H)
                pv.append(P - ep["pos"][t0 + 1:t0 + H + 1])
            for t in range(2, len(ep["pos"]) - 2, 5):
                v, w = D.backward_velocity(ep["pos"], ep["quat"], t)
                sim.set_state(ep["pos"][t], ep["quat"][t], v, w)
                Wp.append(sim.step(ep["action"][t])); O.append(ep["obs"][t])
    pv, Wp, O = np.array(pv), np.array(Wp), np.array(O)
    return dict(c, pos_axis_mm=float(np.sqrt((pv ** 2).mean()) * 1000),
                force=float(np.sqrt(((Wp[:, :3] - O[:, :3]) ** 2).mean(0)).mean()))


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    default = {"friction": BASE.friction, "tc": BASE.solref_tc, "dr": BASE.solref_dr, "d0": BASE.solimp[0],
               "damp_scale": 1.0, "mass_scale": 1.0}
    cands = [default]
    for _ in range(47):
        cands.append({"friction": float(rng.uniform(0.2, 1.2)), "tc": float(np.exp(rng.uniform(np.log(0.03), np.log(0.4)))),
                      "dr": float(rng.uniform(0.5, 2.0)), "d0": float(rng.uniform(0.8, 0.99)),
                      "damp_scale": float(rng.uniform(0.8, 1.25)), "mass_scale": float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))})
    with Pool(4) as pool:
        res = pool.map(evaluate, cands)
    b = res[0]
    for r in res:
        r["score"] = r["pos_axis_mm"] / b["pos_axis_mm"] + r["force"] / b["force"]
    res.sort(key=lambda r: r["score"])
    for r in res[:8]:
        print({k: round(v, 4) for k, v in r.items()})
    print("DEFAULT", {k: round(v, 4) for k, v in b.items()})
    json.dump({"best": res[0], "default": b, "all": res}, open("results/sysid2.json", "w"), indent=1)
