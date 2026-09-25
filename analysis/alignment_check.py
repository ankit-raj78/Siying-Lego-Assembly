"""Which command index drives which motion interval, and which interval does o_t describe?"""
import sys, numpy as np
sys.path.insert(0, sys.argv[1])
import load
DT = 0.1
eps = load.load_episodes("cylinder_peg/train_200k") + load.load_episodes("hexagon_peg/train_200k")
def r2(X, y):
    X1 = np.c_[X, np.ones(len(X))]; b, *_ = np.linalg.lstsq(X1, y, rcond=None); p = X1 @ b
    return 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
# free-space steps: |o| small at both ends
rows = {k: [] for k in ["dx", "Fm1", "F0", "Fp1", "W0", "Wp1"]}
for e in eps:
    x = e["xpos"][:, 0]; F = e["ft_seq"][:, 1]; W = e["wrench_data"]
    for t in range(1, len(x) - 2):
        rows["dx"].append((x[t + 1] - x[t]) / DT)
        rows["Fm1"].append(F[t - 1, :3]); rows["F0"].append(F[t, :3]); rows["Fp1"].append(F[t + 1, :3])
        rows["W0"].append(W[t, :3]); rows["Wp1"].append(W[t + 1, :3])
R = {k: np.array(v) for k, v in rows.items()}
free = (np.linalg.norm(R["W0"], axis=1) < 0.3) & (np.linalg.norm(R["Wp1"], axis=1) < 0.3)
print("free-space transitions:", free.sum())
for name in ["Fm1", "F0", "Fp1"]:
    print(f"  motion x_t->x_t+1 explained by F_cmd[{name}]: R2(x) = {r2(R[name][free], R['dx'][free][:, 0]):.4f}")
# contact: does (x_t+1 - x_t) relate to W_t or W_t+1 ?  D*v = F + W
allm = np.ones(len(R["dx"]), bool)
for wname in ["W0", "Wp1"]:
    for fname in ["F0", "Fp1"]:
        X = np.c_[R[fname], R[wname]]
        print(f"  all steps: v(t->t+1) ~ F_cmd[{fname}] + W[{wname}]: R2(x) = {r2(X, R['dx'][:, 0]):.4f}  R2(z) = {r2(X, R['dx'][:, 2]):.4f}")
