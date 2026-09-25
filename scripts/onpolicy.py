"""DAgger-style on-policy data with re-linearisation.

Roll out a trained residual model from true states on the training episodes. At every visited
(predicted) state, record the LCR inputs and re-linearise the one-step map around the model's
own residual dW_ref:
    err(dW)  ~ e0 + J  (dW - dW_ref)     e0 = step(dW_ref) - true next pose
    F/T(dW)  ~ W0 + JW (dW - dW_ref)
Training on these samples teaches the model to correct the drift it produces itself, and the
Jacobians are taken where the model actually operates instead of at dW = 0.
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from lcr import data as D                                  # noqa: E402
from lcr.rollout import sim_velocity_from_history          # noqa: E402
from lcr.sim import AdmSim, SimParams                      # noqa: E402
from evaluate import load_model, pose_feat                 # noqa: E402
from precompute import EPS                                 # noqa: E402

_CACHE = {}
ARGS = {}


def _get(split):
    if split not in _CACHE:
        torch.set_num_threads(1)
        model, _, _ = load_model(ARGS["model"])
        _CACHE[split] = (D.load_episodes(split), AdmSim(split), model)
    return _CACHE[split]


def snapshot(d):
    return d.qpos.copy(), d.qvel.copy(), d.qacc_warmstart.copy(), d.time


def restore(d, s):
    d.qpos[:], d.qvel[:], d.qacc_warmstart[:], d.time = s[0], s[1], s[2], s[3]
    d.xfrc_applied[:] = 0.0


def process(job):
    split, ep_idx, seed = job
    eps, sim, model = _get(split)
    ep = eps[ep_idx]
    ref = sim.socket_ref()
    rng = np.random.default_rng(seed)
    H = ARGS["horizon"]
    T = len(ep["pos"])
    rows = []
    for t0 in rng.choice(np.arange(2, T - H - 2), size=ARGS["starts"], replace=False):
        v, w = D.backward_velocity(ep["pos"], ep["quat"], t0)
        sim.set_state(ep["pos"][t0], ep["quat"][t0], v, w)
        Ph, Qh = [ep["pos"][t0 - 1], ep["pos"][t0]], [ep["quat"][t0 - 1], ep["quat"][t0]]
        for k in range(H):
            t = t0 + k
            a = ep["action"][t]
            vv, ww = sim_velocity_from_history(Ph, Qh)
            sim.forward(a)
            f = sim.features(a, vv, ww)
            p_now, q_now = sim.pose()
            b = {kk: torch.from_numpy(np.asarray(f[kk])[None]) for kk in ["feats", "R", "r", "lam", "mask", "glob"]}
            b["pose"] = torch.from_numpy(pose_feat(p_now, q_now, ref)[None])
            with torch.no_grad():
                dW_ref = model(b)[0].numpy().astype(np.float64)
            s0 = snapshot(sim.d)
            W0 = sim.step(a, dW_ref); bad = sim.unstable
            p0, q0 = sim.pose()
            s_next = snapshot(sim.d)
            J = np.zeros((6, 6)); JW = np.zeros((6, 6))
            for j in range(6):
                restore(sim.d, s0)
                dW = dW_ref.copy(); dW[j] += EPS[j]
                Wj = sim.step(a, dW); bad |= sim.unstable
                pj, qj = sim.pose()
                J[:3, j] = (pj - p0) / EPS[j]
                J[3:, j] = D.rot_delta_world(q0, qj) / EPS[j]
                JW[:, j] = (Wj - W0) / EPS[j]
            restore(sim.d, s_next)                               # continue the model's own rollout
            if bad or not np.all(np.isfinite(J)):
                break
            e0 = np.concatenate([p0 - ep["pos"][t + 1], D.rot_delta_world(ep["quat"][t + 1], q0)])
            delta = np.concatenate([ep["pos"][t + 1] - ep["pos"][t], D.rot_delta_world(ep["quat"][t], ep["quat"][t + 1])])
            rows.append(dict(feats=f["feats"], R=f["R"], r=f["r"], lam=f["lam"], mask=f["mask"], glob=f["glob"],
                             pose=pose_feat(p_now, q_now, ref), W0=W0, JW=JW, e0=e0, J=J, obs=ep["obs"][t],
                             delta=delta, dWref=dW_ref, ep=ep_idx, t=t, k=k, tool={"cylinder": 0, "hexagon": 1, "square": 2}.get(sim.tool_type, -1)))
            Ph.append(p0); Qh.append(q0)
    return rows


def init(args):
    ARGS.update(args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--starts", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_eps", type=int, default=10**9, help="episodes per split (smoke tests)")
    a = ap.parse_args()
    jobs = [(s, i, a.seed * 100000 + si * 1000 + i) for si, s in enumerate(D.TRAIN_SPLITS)
            for i in range(min(a.max_eps, D.manifest()[s]["n_episodes"]))]
    with Pool(4, initializer=init, initargs=(vars(a),)) as pool:
        chunks = pool.map(process, jobs, chunksize=4)
    rows = [r for c in chunks for r in c]
    arr = {k: np.stack([r[k] for r in rows]).astype(np.int32 if k in ("ep", "t", "k", "tool") else np.float32) for k in rows[0]}
    np.savez(a.out, **arr)
    print(f"{a.out}: {len(rows)} on-policy samples; mean |e0| pos {np.linalg.norm(arr['e0'][:, :3], axis=1).mean()*1000:.3f} mm")
