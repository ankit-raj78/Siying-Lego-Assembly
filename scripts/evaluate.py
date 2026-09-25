"""Evaluate all methods on the paper's protocol plus rollout F/T error.

  * 100-step open-loop rollouts from true states (position RMSE mm, orientation RMSE deg,
    F/T error along the rollout), segments sampled with a fixed seed
  * one-step F/T error from the true state (per-axis-mean and vector RMSE)
  * wall-clock time per predicted step (single CPU core)
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lcr import data as D                                         # noqa: E402
from lcr.models import LCR, GlobalResidual, BlackBox, SkinNet     # noqa: E402
from lcr.skin import SkinGeometry                                 # noqa: E402
from lcr.rollout import rollout, pose_errors, wrench_metrics     # noqa: E402
from lcr.sim import AdmSim, SimParams                             # noqa: E402

H = 100
KIND = {"lcr": LCR, "lcr_nogeom": lambda: LCR(use_geom=False), "lcr_noattn": lambda: LCR(attention=False),
        "global": GlobalResidual, "blackbox": BlackBox, "skin": SkinNet, "skin_gated": lambda: SkinNet(gate_tol=5e-4)}


def load_model(name):
    ck = torch.load(f"results/models/{name}.pt", weights_only=False)
    m = KIND[ck["kind"]]()
    m.load_state_dict(ck["state"]); m.eval()
    return m, ck["stats"], ck["kind"]


def pose_feat(p, q, ref):
    R = D.quat_to_mat(q)
    return np.concatenate([(p - ref) / 0.05, R[:, 0], R[:, 1]]).astype(np.float32)


def glob_feat(v, w, a):
    return np.concatenate([v / 5e-3, w / 0.05, a[:3] / 10.0, a[3:]]).astype(np.float32)


def make_residual(model, ref):
    def f(sim, a, v, w):
        ft = sim.features(a, v, w)
        p, q = sim.pose()
        b = {k: torch.from_numpy(np.asarray(ft[k])[None]) for k in ["feats", "R", "r", "lam", "mask", "glob"]}
        b["pose"] = torch.from_numpy(pose_feat(p, q, ref)[None])
        with torch.no_grad():
            return model(b)[0].numpy().astype(np.float64)
    return f


def make_skin_residual(model, geo):
    def f(sim, a, v, w):
        p, q = sim.pose()
        sc = sim.features(a, v, w)                              # solver contacts (after sim.forward(a))
        m = sc["mask"] > 0
        cpos = (p + sc["r"][m]).astype(np.float64)
        cforce = np.einsum("kij,ki->kj", sc["R"][m], sc["lam"][m]).astype(np.float64)   # rows of R are n,t1,t2
        ft = geo.features(p, q, v, w, a, cpos, cforce, sc["R"][m][:, 0, :].astype(np.float64))
        b = {k: torch.from_numpy(np.asarray(ft[k])[None]) for k in ["feats", "mask", "r", "g", "lam_skin", "glob"]}
        b["W_inst"] = torch.from_numpy(sim.contact_wrench()[None].astype(np.float32))
        with torch.no_grad():
            return model(b)[0].numpy().astype(np.float64)
    return f


def bb_rollout(model, st, ep, t0, H, ref):
    Ph, Qh = [ep["pos"][t0 - 1], ep["pos"][t0]], [ep["quat"][t0 - 1], ep["quat"][t0]]
    P, Q, W = [], [], []
    sd, so = st["sd"].numpy(), st["so"].numpy()
    for k in range(H):
        a = ep["action"][t0 + k]
        v = (Ph[-1] - Ph[-2]) / D.DT; w = D.rot_delta_world(Qh[-2], Qh[-1]) / D.DT
        b = {"glob": torch.from_numpy(glob_feat(v, w, a)[None]),
             "pose": torch.from_numpy(pose_feat(Ph[-1], Qh[-1], ref)[None])}
        with torch.no_grad():
            out = model(b)[0].numpy().astype(np.float64)
        dlt = out[:6] * sd
        p = Ph[-1] + dlt[:3]
        q = D.quat_mul(D.rotvec_to_quat(dlt[3:]), Qh[-1]); q /= np.linalg.norm(q)
        P.append(p); Q.append(q); W.append(out[6:] * so)
        Ph.append(p); Qh.append(q)
    return np.array(P), np.array(Q), np.array(W), False


def job(args):
    method, split, n_seg, stride1 = args
    torch.set_num_threads(1)
    eps = D.load_episodes(split)
    sim = AdmSim(split)
    ref = sim.socket_ref()
    rng = np.random.default_rng(12345)
    model = st = None
    residual = None
    kind = "base"
    if method not in ("base",):
        model, st, kind = load_model(method)
        if kind in ("skin", "skin_gated"):
            residual = make_skin_residual(model, SkinGeometry(sim))
        elif kind != "blackbox":
            residual = make_residual(model, ref)
    run = (lambda ep, t0, h: bb_rollout(model, st, ep, t0, h, ref)) if kind == "blackbox" else \
          (lambda ep, t0, h: rollout(sim, ep, t0, h, residual))
    pe, ae, fe_pred, fe_true, n_bad, steps, secs = [], [], [], [], 0, 0, 0.0
    pvec, rvec = [], []
    example = None
    for ei, ep in enumerate(eps):
        T = len(ep["pos"])
        starts = rng.choice(np.arange(2, T - H - 1), size=n_seg, replace=False)
        for t0 in starts:
            a = time.perf_counter()
            P, Q, W, bad = run(ep, t0, H)
            secs += time.perf_counter() - a; steps += H
            e_p, e_a = pose_errors(P, Q, ep, t0)
            pe.append(e_p); ae.append(e_a); n_bad += int(bad)
            pvec.append(P - ep["pos"][t0 + 1:t0 + H + 1])
            rvec.append(D.rot_delta_world(ep["quat"][t0 + 1:t0 + H + 1], Q))
            fe_pred.append(W); fe_true.append(ep["obs"][t0:t0 + H])
            if example is None and ei == 0:
                example = {"t0": int(t0), "P": P, "W": W, "P_true": ep["pos"][t0 + 1:t0 + H + 1],
                           "W_true": ep["obs"][t0:t0 + H]}
    one_pred, one_true = [], []
    for ep in eps:
        for t in range(2, len(ep["pos"]) - 1, stride1):
            _, _, W, _ = run(ep, t, 1)
            one_pred.append(W[0]); one_true.append(ep["obs"][t])
    return {"method": method, "split": split, "pos_err": np.array(pe), "ang_err": np.array(ae),
            "pos_vec": np.array(pvec), "rot_vec": np.array(rvec),
            "W_pred": np.concatenate(fe_pred), "W_true": np.concatenate(fe_true),
            "one_pred": np.array(one_pred), "one_true": np.array(one_true), "n_bad": n_bad,
            "ms_per_step": 1000 * secs / steps, "example": example}


def linear_baseline():
    """Geometry-free admittance law fitted on train (one-step F/T only)."""
    tr = np.load("cache/train.npz")
    X = np.c_[tr["glob"], np.ones(len(tr["glob"]))]
    B, *_ = np.linalg.lstsq(X, tr["obs"], rcond=None)
    out = {}
    for name in D.TEST_SETS:
        z = np.load(f"cache/{name}.npz")
        out[name] = wrench_metrics(np.c_[z["glob"], np.ones(len(z["glob"]))] @ B, z["obs"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="base,global,lcr,lcr_nogeom,blackbox")
    ap.add_argument("--out", default="results/eval.json")
    ap.add_argument("--suffix", default="", help="checkpoint suffix, e.g. _full")
    a = ap.parse_args()
    methods = a.methods.split(",")
    cfg = {"test_seen": (4, 4), "square_unseen": (4, 4), "expert_ood": (2, 2), "circle_ood": (2, 2)}
    methods_ck = [m if m in ("base",) else m + a.suffix for m in methods]
    methods = methods_ck
    jobs = [(m, s, *cfg[ts]) for m in methods for ts, splits in D.TEST_SETS.items() for s in splits]
    jobs.sort(key=lambda j: j[0].startswith("blackbox"))
    with Pool(4) as pool:
        res = pool.map(job, jobs, chunksize=1)
    summary = {"linear_admittance": {ts: {"one_step": m} for ts, m in linear_baseline().items()}}
    examples = {}
    for m in methods:
        summary[m] = {}
        for ts, splits in D.TEST_SETS.items():
            rs = [r for r in res if r["method"] == m and r["split"] in splits]
            pe = np.concatenate([r["pos_err"] for r in rs]); ae = np.concatenate([r["ang_err"] for r in rs])
            summary[m][ts] = {
                "rollout_pos_rmse_mm": float(np.sqrt((pe ** 2).mean()) * 1000),
                "rollout_ori_rmse_deg": float(np.degrees(np.sqrt((ae ** 2).mean()))),
                "final_pos_err_mm": float(pe[:, -1].mean() * 1000),
                "pos_err_curve_mm": [float(x) for x in pe.mean(0) * 1000],
                "ori_err_curve_deg": [float(x) for x in np.degrees(ae.mean(0))],
                "rollout_pos_rmse_axis_mm": float(np.sqrt((np.concatenate([r["pos_vec"] for r in rs]) ** 2).mean()) * 1000),
                "rollout_ori_rmse_axis_deg": float(np.degrees(np.sqrt((np.concatenate([r["rot_vec"] for r in rs]) ** 2).mean()))),
                "rollout_ft": wrench_metrics(np.concatenate([r["W_pred"] for r in rs]),
                                             np.concatenate([r["W_true"] for r in rs])),
                "one_step": wrench_metrics(np.concatenate([r["one_pred"] for r in rs]),
                                           np.concatenate([r["one_true"] for r in rs])),
                "n_segments": int(len(pe)), "n_unstable": int(sum(r["n_bad"] for r in rs)),
                "ms_per_step": float(np.mean([r["ms_per_step"] for r in rs])),
            }
            for r in rs:
                examples[f"{m}|{r['split']}"] = r["example"]
    os.makedirs("results", exist_ok=True)
    json.dump(summary, open(a.out, "w"), indent=1)
    np.save(a.out.replace(".json", "_examples.npy"), examples, allow_pickle=True)
    for m in summary:
        for ts, v in summary[m].items():
            if "rollout_pos_rmse_mm" in v:
                print(f"{m:18s} {ts:14s} pos {v['rollout_pos_rmse_mm']:6.2f} mm (axis {v['rollout_pos_rmse_axis_mm']:5.2f})  ori {v['rollout_ori_rmse_deg']:5.2f} deg (axis {v['rollout_ori_rmse_axis_deg']:4.2f})  "
                      f"rollF {v['rollout_ft']['force_axis_mean']:5.2f} N  1stepF {v['one_step']['force_axis_mean']:5.2f} N "
                      f"/{v['one_step']['torque_axis_mean']:.4f} Nm  unstable {v['n_unstable']}  {v['ms_per_step']:.2f} ms/step")
            else:
                print(f"{m:18s} {ts:14s} 1stepF {v['one_step']['force_axis_mean']:5.2f} N /{v['one_step']['torque_axis_mean']:.4f} Nm")


if __name__ == "__main__":
    main()
