"""Stage 1: precompute everything the simulator contributes, so training never runs MuJoCo.

For every sample (true state s_t, action a_t):
  * per-contact LCR features at s_t
  * base step (dW = 0): predicted next pose error e0 (vs. s_{t+1}) and mean contact wrench W0
  * finite-difference Jacobians of the one-step map w.r.t. the residual wrench dW:
      J  : d[next pos, next rot] / d dW        (6x6)
      JW : d[predicted F/T]      / d dW        (6x6)
    so that, to first order,  err(dW) = e0 + J dW  and  F/T(dW) = W0 + JW dW.
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lcr import data as D                     # noqa: E402
from lcr.sim import AdmSim, SimParams, MAX_CONTACTS, N_FEAT, N_GLOBAL  # noqa: E402

EPS = np.array([2.0, 2.0, 2.0, 0.05, 0.05, 0.05])


def one_step(sim, ep, t, dW):
    v, w = D.backward_velocity(ep["pos"], ep["quat"], t)
    sim.set_state(ep["pos"][t], ep["quat"][t], v, w)
    W = sim.step(ep["action"][t], dW)
    p, q = sim.pose()
    return p, q, W, sim.unstable


_CACHE = {}


def _get(split):
    """Per-worker cache of episodes and simulator (loading a 57 MB split per job is slow)."""
    if split not in _CACHE:
        _CACHE[split] = (D.load_episodes(split), AdmSim(split))
    return _CACHE[split]


def process(job):
    split, ep_idx, stride, offset = job
    eps, sim = _get(split)
    ref = sim.socket_ref()
    ep = eps[ep_idx]
    rows = []
    T = len(ep["pos"])
    for t in range(2 + offset, T - 1, stride):
        v, w = D.backward_velocity(ep["pos"], ep["quat"], t)
        sim.set_state(ep["pos"][t], ep["quat"][t], v, w)
        sim.forward(ep["action"][t])
        f = sim.features(ep["action"][t], v, w)
        p0, q0, W0, bad = one_step(sim, ep, t, np.zeros(6))
        J = np.zeros((6, 6)); JW = np.zeros((6, 6))
        for j in range(6):
            dW = np.zeros(6); dW[j] = EPS[j]
            pj, qj, Wj, b = one_step(sim, ep, t, dW)
            bad |= b
            J[:3, j] = (pj - p0) / EPS[j]
            J[3:, j] = D.rot_delta_world(q0, qj) / EPS[j]
            JW[:, j] = (Wj - W0) / EPS[j]
        if bad:
            continue
        e0 = np.concatenate([p0 - ep["pos"][t + 1], D.rot_delta_world(ep["quat"][t + 1], q0)])
        Rt = D.quat_to_mat(ep["quat"][t])
        pose_feat = np.concatenate([(ep["pos"][t] - ref) / 0.05, Rt[:, 0], Rt[:, 1]])
        true_delta = np.concatenate([ep["pos"][t + 1] - ep["pos"][t],
                                     D.rot_delta_world(ep["quat"][t], ep["quat"][t + 1])])
        rows.append(dict(feats=f["feats"], R=f["R"], r=f["r"], lam=f["lam"], mask=f["mask"], glob=f["glob"],
                         pose=pose_feat.astype(np.float32), W0=W0, JW=JW, e0=e0, J=J, obs=ep["obs"][t],
                         delta=true_delta, ep=ep_idx, t=t))
    return rows


def run(splits, stride, out, workers=4, offset=0):
    jobs = [(s, i, stride, offset) for s in splits for i in range(D.manifest()[s]["n_episodes"])]
    with Pool(workers) as pool:
        chunks = pool.map(process, jobs, chunksize=4)
    rows = [r for c in chunks for r in c]
    arr = {k: np.stack([r[k] for r in rows]).astype(np.float32 if k not in ("ep", "t") else np.int32)
           for k in rows[0]}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, **arr)
    print(f"{out}: {len(rows)} samples, contacts/sample {arr['mask'].sum(1).mean():.2f}, "
          f"with contact {np.mean(arr['mask'].sum(1) > 0):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all")
    ap.add_argument("--offset", type=int, default=0, help="start step offset (1 = the other half at stride 2)")
    ap.add_argument("--prefix", default="", help="cache file prefix, e.g. s2_ for another base-sim parameter set")
    a = ap.parse_args()
    todo = {
        "train": (D.TRAIN_SPLITS, 2), "val": (D.VAL_SPLITS, 2),
        "test_seen": (D.TEST_SETS["test_seen"], 2), "square_unseen": (D.TEST_SETS["square_unseen"], 2),
        "expert_ood": (D.TEST_SETS["expert_ood"], 1), "circle_ood": (D.TEST_SETS["circle_ood"], 1),
    }
    for name, (splits, stride) in todo.items():
        if a.which in ("all", name):
            run(splits, stride, f"cache/{a.prefix}{name}{'_off%d' % a.offset if a.offset else ''}.npz", offset=a.offset)
