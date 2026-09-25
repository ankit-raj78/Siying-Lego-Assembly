"""Free-space velocity-space test of command preprocessing variants, all implementable in the sim:
   V1: F_app = F_t                       (current)
   V2: F_app = (1-a) F_t + a F_{t+1}     (command timestamp offset; whole action sequence is known in rollout)
   V3: F_app = F_t + b (F_t - F_{t-1})   (lead term)
   V4: F_app = (1-a) F_t + a F_{t+1} + b (F_t - F_{t-1})
Model per axis: v_{t+1} = p v_t + q F_app  (discrete mass-damper). Fit p,q by LS on train free-space for a grid of a,b;
report held-out val RMSE of v_{t+1} and implied per-step position error."""
import sys, numpy as np
sys.path.insert(0, ".")
from lcr import data as D
np.set_printoptions(precision=4, suppress=True, linewidth=160)

def collect(splits, max_eps=None):
    V0, V1, Fm, F0, Fp = [], [], [], [], []
    for s in splits:
        for e in D.load_episodes(s)[:max_eps]:
            P, Q, A, O = e["pos"], e["quat"], e["action"], e["obs"]
            for t in range(3, len(P) - 2):
                if max(np.linalg.norm(O[t-1:t+2, :3], axis=1)) > 0.3: continue
                v0, w0 = D.backward_velocity(P, Q, t); v1, w1 = D.backward_velocity(P, Q, t + 1)
                V0.append(np.r_[v0, w0]); V1.append(np.r_[v1, w1]); Fm.append(A[t-1]); F0.append(A[t]); Fp.append(A[t+1])
    return [np.array(x) for x in (V0, V1, Fm, F0, Fp)]

tr = collect(D.TRAIN_SPLITS, 200); va = collect(D.VAL_SPLITS)
print("free-space transitions: train", len(tr[0]), "val", len(va[0]))

def fapp(Fm, F0, Fp, a, b): return (1 - a) * F0 + a * Fp + b * (F0 - Fm)
def fit_eval(a, b):
    out = []
    for k in range(6):
        X = np.c_[tr[0][:, k], fapp(tr[2], tr[3], tr[4], a, b)[:, k], np.ones(len(tr[0]))]
        c, *_ = np.linalg.lstsq(X, tr[1][:, k], rcond=None)
        Xv = np.c_[va[0][:, k], fapp(va[2], va[3], va[4], a, b)[:, k], np.ones(len(va[0]))]
        out.append(np.sqrt(((Xv @ c - va[1][:, k]) ** 2).mean()))
    return np.array(out)

base = fit_eval(0, 0)
print("V1 (current) val RMSE of v_{t+1} [m/s x3, rad/s x3]:", base, " -> pos err/step ~", (base[:3] * D.DT * 1e3).round(4), "mm")
best = {}
for name, grid in [("V2 offset a", [(a, 0) for a in np.linspace(0, 1, 11)]),
                   ("V3 lead b", [(0, b) for b in np.linspace(0, 1.5, 16)]),
                   ("V4 both", [(a, b) for a in np.linspace(0, 1, 6) for b in np.linspace(0, 1.5, 7)])]:
    res = [(fit_eval(a, b), a, b) for a, b in grid]
    r, a, b = min(res, key=lambda x: x[0][:3].sum())
    best[name] = (a, b)
    print(f"{name:12s} best a={a:.2f} b={b:.2f}: val RMSE {r}  translational improvement {(1 - r[:3] / base[:3]).round(3)}")
