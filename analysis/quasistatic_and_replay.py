import sys, numpy as np, mujoco
sys.path.insert(0, sys.argv[1])
import load
np.set_printoptions(precision=3, suppress=True, linewidth=160)
DT = 0.1
# ---- (1) free-space virtual mass/damping from sysid_no_con: F_cmd = M a + D v + c
eps = load.load_episodes("cylinder_peg/sysid_no_con_10k")
X, Y = [], []
for e in eps:
    v = e["qvel"]; a = np.gradient(v, DT, axis=0); F = e["ft_seq"][:, 1]; W = e["wrench_data"]
    free = np.linalg.norm(W[:, :3], axis=1) < 0.5
    for k in range(6):
        pass
    X.append((v[free], a[free], F[free])); 
V = np.concatenate([x[0] for x in X]); A = np.concatenate([x[1] for x in X]); F = np.concatenate([x[2] for x in X])
print("free-space samples:", len(V))
for k in range(6):
    M_ = np.c_[A[:, k], V[:, k], np.ones(len(V))]
    coef, *_ = np.linalg.lstsq(M_, F[:, k], rcond=None)
    pred = M_ @ coef; r2 = 1 - ((pred - F[:, k]) ** 2).sum() / ((F[:, k] - F[:, k].mean()) ** 2).sum()
    tau = coef[0] / coef[1] if coef[1] != 0 else np.nan
    print(f"axis {k}: M={coef[0]:8.2f}  D={coef[1]:8.2f}  M/D={tau*1000:7.1f} ms  R2={r2:.3f}")

# ---- (2) replay with admittance damper: xfrc = F_cmd - D v ; contact wrench as F/T prediction
Dv = np.array([915, 898, 1122, 12.4, 13.0, 11.7])
model, test = load.load_split("cylinder_peg/test_25k")
model.opt.gravity[:] = 0
tool = 1
model.body_mass[tool] = 5.0; model.body_inertia[tool] = [0.05, 0.05, 0.05]
data = mujoco.MjData(model)
tool_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == tool}
c6 = np.zeros(6)
def contact_wrench():
    f_w = np.zeros(3); t_w = np.zeros(3)
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 in tool_geoms or c.geom2 in tool_geoms:
            mujoco.mj_contactForce(model, data, i, c6)
            f = c.frame.reshape(3, 3).T @ c6[:3]
            if c.geom2 not in tool_geoms: f = -f
            f_w += f; t_w += np.cross(c.pos - data.xpos[tool], f)
    return np.r_[f_w, t_w]
P, Wm = [], []
for e in test[:20]:
    for t in range(1, e["qpos"].shape[0]):
        data.qpos[:] = e["qpos"][t]; data.qvel[:] = e["qvel"][t]
        data.xfrc_applied[tool] = e["ft_seq"][t, 1] - Dv * e["qvel"][t]
        mujoco.mj_forward(model, data)
        P.append(contact_wrench()); Wm.append(e["wrench_data"][t])
P = np.array(P); Wm = np.array(Wm)
print("replay WITH admittance damper, per-axis RMSE:", np.sqrt(((P - Wm) ** 2).mean(0)))
print("  per-axis corr:", np.round([np.corrcoef(P[:, k], Wm[:, k])[0, 1] for k in range(6)], 3))
print("  zero-pred RMSE:", np.sqrt((Wm ** 2).mean(0)))
