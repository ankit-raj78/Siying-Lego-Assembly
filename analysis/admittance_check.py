"""How much of the measured wrench is explained by the admittance law
F_ext ~ M*a + D*v - F_cmd (no geometry at all)?"""
import sys, numpy as np
sys.path.insert(0, sys.argv[1])
import load
np.set_printoptions(precision=3, suppress=True, linewidth=160)
dt = 0.01

def feats(eps, use_acc):
    X, Y = [], []
    for e in eps:
        F = e["ft_seq"][:, 1]; V = e["qvel"]; W = e["wrench_data"]
        A = np.gradient(V, dt, axis=0)
        cols = [F, V] + ([A] if use_acc else [])
        X.append(np.concatenate(cols, 1)); Y.append(W)
    return np.concatenate(X), np.concatenate(Y)

def fit_eval(train, test, use_acc):
    Xtr, Ytr = feats(train, use_acc); Xte, Yte = feats(test, use_acc)
    Xtr1 = np.c_[Xtr, np.ones(len(Xtr))]; Xte1 = np.c_[Xte, np.ones(len(Xte))]
    B, *_ = np.linalg.lstsq(Xtr1, Ytr, rcond=None)
    P = Xte1 @ B
    rmse = np.sqrt(((P - Yte) ** 2).mean(0)); r2 = 1 - ((P - Yte) ** 2).sum(0) / ((Yte - Yte.mean(0)) ** 2).sum(0)
    return rmse, r2, B, Yte

tr = load.load_episodes("cylinder_peg/train_200k") + load.load_episodes("hexagon_peg/train_200k")
for split in ["cylinder_peg/test_25k", "hexagon_peg/test_25k", "square_peg/test_25k"]:
    te = load.load_episodes(split)
    for use_acc in (False, True):
        rmse, r2, B, Yte = fit_eval(tr, te, use_acc)
        print(f"{split:24s} linear[F_cmd, v{', a' if use_acc else ''}] RMSE {rmse}  R2 {np.round(r2,3)}")
    print(f"{split:24s} zero-pred RMSE {np.sqrt((Yte**2).mean(0))}")
    # diag coefficient of F_cmd on measured (expect ~ -1 if quasi-static contact)
_, _, B, _ = fit_eval(tr, te, False)
print("coef of F_cmd (diag):", np.diag(B[:6, :6])); print("coef of v (diag):", np.diag(B[6:12, :6]))
# In contact vs free: fraction of steps with |F_meas| > 1 N
W = np.concatenate([e["wrench_data"] for e in tr]); print("frac |F_meas|>1N:", np.mean(np.linalg.norm(W[:, :3], axis=1) > 1.0))
