"""Stage 0c': grid over the two physically justified geometry corrections (cylinder mesh inflation,
socket height) with position, ORIENTATION and force all in the objective. Base: LCR_SIMPARAMS (sysid2)."""
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
    pv, rv, Wp, O = [], [], [], []
    for split in D.TRAIN_SPLITS:
        sim = AdmSim(split, params_from_cfg(cfg))
        for ep in D.load_episodes(split)[:N_EPS]:
            for t0 in segment_starts(len(ep["pos"]), H, 4, rng):
                P, Q, W, bad = rollout(sim, ep, t0, H)
                pv.append(P - ep["pos"][t0 + 1:t0 + H + 1])
                rv.append(D.rot_delta_world(ep["quat"][t0 + 1:t0 + H + 1], Q))
            for t in range(2, len(ep["pos"]) - 2, 10):
                v, w = D.backward_velocity(ep["pos"], ep["quat"], t)
                sim.set_state(ep["pos"][t], ep["quat"][t], v, w)
                Wp.append(sim.step(ep["action"][t])); O.append(ep["obs"][t])
    pv, rv, Wp, O = map(np.array, (pv, rv, Wp, O))
    return dict(c, pos_axis_mm=float(np.sqrt((pv ** 2).mean()) * 1e3), ori_axis_deg=float(np.degrees(np.sqrt((rv ** 2).mean()))),
                force=float(np.sqrt(((Wp[:, :3] - O[:, :3]) ** 2).mean(0)).mean()))

if __name__ == "__main__":
    cands = [{"inflate_cylinder": a, "dz": z} for a in (0.0, 1.0e-3, 1.25e-3, 1.5e-3, 2.0e-3) for z in (0.0, 1.0e-3, 1.25e-3, 1.5e-3, 2.0e-3)]
    with Pool(4) as pool:
        res = pool.map(evaluate, cands)
    b = res[0]
    for r in res:
        r["score"] = r["pos_axis_mm"] / b["pos_axis_mm"] + r["ori_axis_deg"] / b["ori_axis_deg"] + r["force"] / b["force"]
    res.sort(key=lambda r: r["score"])
    for r in res[:6]:
        print({k: round(v, 5) for k, v in r.items()})
    print("DEFAULT", {k: round(v, 4) for k, v in b.items()})
    json.dump({"best": res[0], "default": b, "all": res}, open("results/sysid4_ext2.json", "w"), indent=1)
    json.dump(dict(BASE, **{k: res[0][k] for k in ("inflate_cylinder", "dz")}), open("results/simparams_sysid4_ext2.json", "w"), indent=1)
    print("SYSID4_DONE")
