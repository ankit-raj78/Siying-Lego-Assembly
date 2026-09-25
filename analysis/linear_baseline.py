"""Geometry-free one-step F/T baseline using only past poses (no future leakage):
   W_t ~ A*F_cmd_t + B*v_bwd_t + C*w_bwd_t + c,  v_bwd from x_t - x_{t-1}."""
import sys, numpy as np
sys.path.insert(0, sys.argv[1])
import load
np.set_printoptions(precision=3, suppress=True, linewidth=160)
DT = 0.1

def quat_to_rotvec_delta(q0, q1):
    # relative rotation q1 * conj(q0) -> rotation vector (world frame), quats wxyz
    w0, v0 = q0[:, :1], q0[:, 1:]; w1, v1 = q1[:, :1], q1[:, 1:]
    # conj(q0)
    w0c, v0c = w0, -v0
    w = w1 * w0c - (v1 * v0c).sum(1, keepdims=True)
    v = w1 * v0c + w0c * v1 + np.cross(v1, v0c)
    s = np.sign(w); s[s == 0] = 1; w, v = w * s, v * s
    ang = 2 * np.arctan2(np.linalg.norm(v, axis=1, keepdims=True), w)
    n = np.linalg.norm(v, axis=1, keepdims=True); n[n < 1e-12] = 1
    return v / n * ang

def feats(eps):
    X, Y = [], []
    for e in eps:
        x = e["xpos"][:, 0]; q = e["xquat"][:, 0]; F = e["ft_seq"][:, 1]; W = e["wrench_data"]
        v = (x[1:] - x[:-1]) / DT                     # velocity ending at t (t>=1)
        w = quat_to_rotvec_delta(q[:-1], q[1:]) / DT
        X.append(np.c_[F[1:], v, w]); Y.append(W[1:])
    return np.concatenate(X), np.concatenate(Y)

tr = load.load_episodes("cylinder_peg/train_200k") + load.load_episodes("hexagon_peg/train_200k")
Xtr, Ytr = feats(tr); Xtr1 = np.c_[Xtr, np.ones(len(Xtr))]
B, *_ = np.linalg.lstsq(Xtr1, Ytr, rcond=None)
print("diag coef F_cmd:", np.diag(B[:6, :6]))
print("diag coef v,w  :", np.r_[np.diag(B[6:9, :3]), np.diag(B[9:12, 3:6])])

def report(name, te):
    X, Y = feats(te); P = np.c_[X, np.ones(len(X))] @ B
    E = P - Y
    f_axis = np.sqrt((E[:, :3] ** 2).mean(0)); t_axis = np.sqrt((E[:, 3:] ** 2).mean(0))
    f_vec = np.sqrt((np.linalg.norm(E[:, :3], axis=1) ** 2).mean()); t_vec = np.sqrt((np.linalg.norm(E[:, 3:], axis=1) ** 2).mean())
    print(f"{name:34s} force RMSE/axis {f_axis} (mean {f_axis.mean():.3f}, vec {f_vec:.3f}) N | "
          f"torque RMSE/axis {t_axis} (mean {t_axis.mean():.4f}, vec {t_vec:.4f}) Nm")

test = load.load_episodes("cylinder_peg/test_25k") + load.load_episodes("hexagon_peg/test_25k")
report("test (cyl+hex, seen tools)", test)
report("square_peg test (unseen tool)", load.load_episodes("square_peg/test_25k"))
for s in ["cylinder_peg/expert_10k", "hexagon_peg/expert_10k",
          "cylinder_peg/circle/radius_0.02_numCircles_1_epLen_200_steps_10k"]:
    report(s.split("/")[0] + "/" + s.split("/")[1], load.load_episodes(s))
Y = feats(test)[1]
print("zero predictor on test: force vec RMSE %.3f N, torque vec RMSE %.4f Nm" % (
    np.sqrt((np.linalg.norm(Y[:, :3], axis=1) ** 2).mean()), np.sqrt((np.linalg.norm(Y[:, 3:], axis=1) ** 2).mean())))
