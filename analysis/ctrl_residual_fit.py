"""Feasibility: a linear controller correction f = K [v, w, F_t, F_{t-1}, 1] applied as an extra wrench,
fitted on free-space steps using the precomputed one-step map  err(f) = e0 + J f.
Prints free-space pose RMSE before/after on train and on held-out val."""
import sys, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from lcr import data as D
np.set_printoptions(precision=4, suppress=True, linewidth=160)

def build(name, splits, stride=2, offset=0):
    z = np.load(f"cache/{name}.npz")
    eps = {s: D.load_episodes(s) for s in splits}
    n_per = {s: len(eps[s]) * len(range(2 + offset, 499, stride)) for s in splits}
    split_of = np.concatenate([[s] * n_per[s] for s in splits])
    assert len(split_of) == len(z["obs"]), (len(split_of), len(z["obs"]))
    Fprev = np.array([eps[s][e]["action"][t - 1] for s, e, t in zip(split_of, z["ep"], z["t"])])
    Fcur = np.array([eps[s][e]["action"][t] for s, e, t in zip(split_of, z["ep"], z["t"])])
    g = z["glob"]; v = g[:, :3] * 5e-3; w = g[:, 3:6] * 0.05
    free = z["mask"].sum(1) == 0
    return dict(e0=z["e0"], J=z["J"], v=v, w=w, Ft=Fcur, Fp=Fprev, free=free)

def feats(d, kind):
    cols = {"none": [], "vF": [d["v"], d["w"], d["Ft"]], "vFFp": [d["v"], d["w"], d["Ft"], d["Fp"]]}[kind]
    if not cols:
        return None
    return np.concatenate(cols + [np.ones((len(d["v"]), 1))], 1)

SCALE = np.array([1e3, 1e3, 1e3, 1e3, 1e3, 1e3])   # mm and mrad
def rms(e): return np.sqrt((e ** 2).mean(0)) * SCALE

def fit(d, kind):
    X = feats(d, kind); m = d["free"]
    e0, J = d["e0"][m], d["J"][m]; X = X[m]
    N, p = X.shape
    A = np.einsum("ni,njk->njik", X, J).reshape(N * 6, 6 * p)   # (n,j,i,k): row (n,j), col (k,i)?  fix below
    # err_j = e0_j + sum_k J_jk f_k, f_k = sum_i K_ki x_i  -> coeff of K_ki is J_jk x_i
    A = np.einsum("njk,ni->njki", J, X).reshape(N * 6, 6 * p)
    b = -e0.reshape(-1)
    K, *_ = np.linalg.lstsq(A, b, rcond=None)
    return K.reshape(6, p)

def apply(d, kind, K):
    X = feats(d, kind)
    f = X @ K.T
    return d["e0"] + np.einsum("njk,nk->nj", d["J"], f), f

tr = build("s2_train", D.TRAIN_SPLITS); va = build("s2_val", D.VAL_SPLITS)
print("free-space samples: train", tr["free"].sum(), "val", va["free"].sum())
print("val free-space RMSE [x y z mm | rx ry rz mrad], base sim:", rms(va["e0"][va["free"]]))
for kind in ["vF", "vFFp"]:
    K = fit(tr, kind)
    err_tr, _ = apply(tr, kind, K); err_va, f = apply(va, kind, K)
    print(f"  + linear ctrl correction [{kind}]  train free {rms(err_tr[tr['free']])}  val free {rms(err_va[va['free']])}")
    print(f"      val ALL steps (contact too): base {rms(va['e0']).round(3)} -> {rms(err_va).round(3)};  |f| mean {np.abs(f).mean(0).round(3)}")
    if kind == "vFFp":
        np.save("results/ctrl_K_vFFp.npy", K)
        print("  K rows=wrench out [Fx Fy Fz Tx Ty Tz], cols=[v(3) w(3) Ft(6) Fprev(6) 1]:")
        print(K)
