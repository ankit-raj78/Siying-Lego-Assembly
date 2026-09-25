"""Stage 0d: per-axis virtual mass/damping identified THROUGH the simulator on free-space training steps.
Axes decouple in free space, so each (mass_k, damp_k) scale pair is a 2-D grid search minimising the
one-step error of the actual MuJoCo step (integration scheme included). Base params: LCR_SIMPARAMS."""
import json, os, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lcr import data as D
from lcr.sim import AdmSim, params_from_cfg
BASE = json.load(open(os.environ["LCR_SIMPARAMS"]))
MS = [0.5, 0.65, 0.8, 1.0, 1.25, 1.6, 2.0]
DS = [0.8, 0.88, 0.94, 1.0, 1.06, 1.12, 1.25]
_S = {}

def samples():
    if "s" not in _S:
        out = []
        for s in D.TRAIN_SPLITS:
            for ep in D.load_episodes(s)[:120]:
                O = ep["obs"]
                for t in range(3, len(ep["pos"]) - 2, 3):
                    if max(np.linalg.norm(O[t-1:t+2, :3], axis=1)) < 0.3:
                        out.append((s, ep, t))
        _S["s"] = out
    return _S["s"]

def evaluate(job):
    k, ms, ds = job
    cfg = dict(BASE); cfg["mass_axis"] = [1.0] * 6; cfg["damp_axis"] = [1.0] * 6
    cfg["mass_axis"][k] = ms; cfg["damp_axis"][k] = ds
    sims = {s: AdmSim(s, params_from_cfg(cfg)) for s in D.TRAIN_SPLITS}
    errs = []
    for s, ep, t in samples():
        sim = sims[s]
        v, w = D.backward_velocity(ep["pos"], ep["quat"], t)
        sim.set_state(ep["pos"][t], ep["quat"][t], v, w); sim.step(ep["action"][t])
        p, q = sim.pose()
        e = p - ep["pos"][t + 1] if k < 3 else D.rot_delta_world(ep["quat"][t + 1], q)
        errs.append(e[k] if k < 3 else e[k - 3])
    return k, ms, ds, float(np.sqrt(np.mean(np.square(errs))))

if __name__ == "__main__":
    print("free-space samples:", len(samples()))
    jobs = [(k, ms, ds) for k in range(6) for ms in MS for ds in DS]
    with Pool(4) as pool:
        res = pool.map(evaluate, jobs, chunksize=6)
    mass_axis, damp_axis = [1.0] * 6, [1.0] * 6
    for k in range(6):
        rk = [r for r in res if r[0] == k]
        base = next(r for r in rk if r[1] == 1.0 and r[2] == 1.0)
        best = min(rk, key=lambda r: r[3])
        mass_axis[k], damp_axis[k] = best[1], best[2]
        print(f"axis {k}: base {base[3]*1e3:.4f} -> best {best[3]*1e3:.4f} (x1e3)  mass x{best[1]} damp x{best[2]}")
    out = dict(BASE, mass_axis=mass_axis, damp_axis=damp_axis)
    json.dump(out, open("results/simparams_sysid_free.json", "w"), indent=1)
    json.dump({"grid": res}, open("results/sysid_free.json", "w"))
    print("SYSID_FREE_DONE")
