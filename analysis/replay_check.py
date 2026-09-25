"""Teacher-forced replay: put measured tool pose into the abstract MuJoCo model,
read the contact wrench on the tool, and compare it with the measured F/T."""
import sys, numpy as np, mujoco
sys.path.insert(0, sys.argv[1])
import load

np.set_printoptions(precision=3, suppress=True, linewidth=160)
split = sys.argv[2]
n_eps = int(sys.argv[3]) if len(sys.argv) > 3 else 5
model, eps = load.load_split(split)
data = mujoco.MjData(model)
tool = model.body(1).id
tool_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == tool}
print("bodies:", [model.body(i).name for i in range(model.nbody)], "timestep", model.opt.timestep)

def contact_wrench(m, d):
    """Sum of contact forces acting ON the tool, about the tool origin, world frame."""
    f_w = np.zeros(3); t_w = np.zeros(3); n = 0; depths = []
    c6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        if c.geom1 in tool_geoms or c.geom2 in tool_geoms:
            mujoco.mj_contactForce(m, d, i, c6)
            R = c.frame.reshape(3, 3)          # rows: normal, t1, t2 (world)
            f = R.T @ c6[:3]                    # force on geom2 from geom1, world
            if c.geom2 not in tool_geoms:       # tool is geom1 -> flip
                f = -f
            r = c.pos - d.xpos[tool]
            f_w += f; t_w += np.cross(r, f); n += 1; depths.append(c.dist)
    return f_w, t_w, n, depths

rows = []
ncons = []
for e in eps[:n_eps]:
    for t in range(e["qpos"].shape[0]):
        data.qpos[:] = e["qpos"][t]; data.qvel[:] = e["qvel"][t]
        data.xfrc_applied[:] = e["ft_seq"][t]
        mujoco.mj_forward(model, data)
        f_w, t_w, n, depths = contact_wrench(model, data)
        R = data.xmat[tool].reshape(3, 3)       # body->world
        f_b, t_b = R.T @ f_w, R.T @ t_w         # express in tool/sensor frame
        rows.append(np.concatenate([f_w, t_w, f_b, t_b, e["wrench_data"][t]]))
        ncons.append(n)
rows = np.array(rows); ncons = np.array(ncons)
W = rows[:, 12:]
print(f"steps {len(rows)}  frac with tool contact {np.mean(ncons>0):.3f}  contacts/step mean {ncons.mean():.2f} max {ncons.max()}")
for name, S in [("world", rows[:, 0:6]), ("tool frame", rows[:, 6:12])]:
    corr = [np.corrcoef(S[:, k], W[:, k])[0, 1] if S[:, k].std() > 1e-9 else np.nan for k in range(6)]
    print(f"corr(sim contact wrench [{name}], measured) per axis:", np.round(corr, 3))
S = rows[:, 6:12]
for sign in (+1, -1):
    rmse = np.sqrt(((sign * S - W) ** 2).mean(0))
    print(f"RMSE sign={sign:+d} tool frame:", rmse)
print("measured std:", W.std(0))
print("zero-predictor RMSE:", np.sqrt((W ** 2).mean(0)))
